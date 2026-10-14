# ============================================================
# SECTION 7 - TRAINING & VALIDATION LOOP (AMP + EMA + CV)
# ============================================================
from __future__ import annotations

from sklearn.model_selection import StratifiedGroupKFold


def make_fold_labels(records: pd.DataFrame, cfg: Config = CFG) -> np.ndarray:
    """Multi-label stratification key for StratifiedGroupKFold.

    Encodes the 12-bit label pattern as an integer so rare positive combinations
    are spread evenly across folds; studies with no labels get key 0.
    """
    cols = [c for c in cfg.TARGET_COLS if c in records.columns]
    mat = records[cols].fillna(0.0).to_numpy(dtype=np.float32) if cols else \
        np.zeros((len(records), cfg.NUM_TARGETS), np.float32)
    bits = (mat > 0.5).astype(np.int64)
    if bits.shape[1] < 12:
        pad = np.zeros((bits.shape[0], 12 - bits.shape[1]), np.int64)
        bits = np.concatenate([bits, pad], axis=1)
    keys = np.zeros(len(bits), dtype=np.int64)
    for j in range(12):
        keys += bits[:, j] * (2 ** j)
    # Keep key cardinality manageable: bucket by positive-count when patterns are unique.
    npos = bits.sum(1)
    uniq = np.unique(keys).size
    if uniq > max(32, len(bits) // 8):
        keys = npos.astype(np.int64)
    return keys


def build_folds(records: pd.DataFrame, cfg: Config = CFG) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return list of (train_idx, valid_idx) grouped by StudyInstanceUID."""
    groups = records[cfg.GROUP_COL].astype(str).to_numpy()
    y = make_fold_labels(records, cfg)
    n_groups = int(pd.Series(groups).nunique())
    k = min(cfg.N_FOLDS, max(2, n_groups))
    try:
        sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=cfg.SEED)
        splits = list(sgkf.split(np.zeros(len(records)), y, groups))
    except Exception as exc:
        print(f"[cv] StratifiedGroupKFold failed ({exc}); falling back to GroupKFold")
        from sklearn.model_selection import GroupKFold
        gkf = GroupKFold(n_splits=k)
        splits = list(gkf.split(np.zeros(len(records)), groups=groups))
    print(f"[cv] {k} folds over {n_groups} study groups")
    return splits


# --------------------------------------------------------------- scheduler
class WarmupCosineScheduler:
    """Linear warmup for `warmup_steps`, then cosine decay to min_lr (per-step)."""

    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int,
                 total_steps: int, base_lr: float, min_lr: float = 1e-6) -> None:
        self.opt = optimizer
        self.warmup = max(1, int(warmup_steps))
        self.total = max(self.warmup + 1, int(total_steps))
        self.base_lr, self.min_lr = base_lr, min_lr
        self.step_num = 0

    def step(self) -> float:
        self.step_num += 1
        if self.step_num <= self.warmup:
            lr = self.base_lr * self.step_num / self.warmup
        else:
            prog = (self.step_num - self.warmup) / max(1, self.total - self.warmup)
            prog = min(1.0, prog)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * prog))
        for g in self.opt.param_groups:
            g["lr"] = lr
        return lr


# ------------------------------------------------------------------ trainer
class KneeTrainer:
    """Fold-level trainer with AMP, gradient accumulation, EMA and OOF collection."""

    def __init__(self, model: nn.Module, cfg: Config = CFG, device: Optional[torch.device] = None) -> None:
        self.cfg = cfg
        self.device = device or DEVICE
        self.model = model.to(self.device)
        self.criterion = make_loss(cfg).to(self.device)
        self.ema = EMAModel(self.model, cfg.EMA_DECAY)
        self.scaler = torch.amp.GradScaler(
            device_type=self.device.type, enabled=(cfg.MIXED_PRECISION and self.device.type == "cuda"))
        self.amp_dtype = resolve_amp_dtype(cfg)
        self.history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------ optimiser
    def configure_optimiser(self, steps_per_epoch: int, epochs: int) -> None:
        total = max(1, steps_per_epoch * epochs)
        self.optimizer = torch.optim.AdamW(
            self.model.param_groups(self.cfg.WEIGHT_DECAY),
            lr=self.cfg.LR, betas=self.cfg.BETAS, weight_decay=self.cfg.WEIGHT_DECAY,
            eps=1e-8, foreach=False if self.cfg.DETERMINISTIC else True,
        )
        self.scheduler = WarmupCosineScheduler(
            self.optimizer, self.cfg.WARMUP_STEPS, total, self.cfg.LR, self.cfg.MIN_LR)

    # ------------------------------------------------------------- one epoch
    def _prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        vol = batch["volume"].to(self.device, non_blocking=True)          # (B,P,C,K,H,W)
        meta = batch["meta_flags"].to(self.device, non_blocking=True)
        mask = batch["mask"].to(self.device, non_blocking=True)
        labels = batch["labels"].to(self.device, non_blocking=True)
        weight = batch["weight"].to(self.device, non_blocking=True)
        return {"volume": vol, "meta": meta, "mask": mask,
                "plane_ids": batch["plane_ids"].to(self.device, non_blocking=True),
                "labels": labels, "weight": weight}

    @staticmethod
    def _normalise_tensor(vol: torch.Tensor, cfg: Config) -> torch.Tensor:
        """Map [0,1] MRI intensities into ImageNet-ish normalisation window."""
        mean = torch.tensor(cfg.NORM_MEAN, device=vol.device, dtype=vol.dtype).view(1, 1, 3, 1, 1, 1)
        std = torch.tensor(cfg.NORM_STD, device=vol.device, dtype=vol.dtype).view(1, 1, 3, 1, 1, 1)
        return (vol - mean) / std

    def train_one_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.train()
        running = 0.0
        n_batches = 0
        self.optimizer.zero_grad(set_to_none=True)
        pbar = _tqdm(loader, desc=f"train ep{epoch}", leave=False)
        for it, batch in enumerate(pbar):
            if self.cfg.time_left_for_training() <= 0:
                print("[time] training wall-clock budget exhausted -> stopping epoch")
                break
            tb = self._prepare_batch(batch)
            x = self._normalise_tensor(tb["volume"], self.cfg)
            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype,
                                enabled=self.cfg.MIXED_PRECISION and self.device.type == "cuda"):
                logits = self.model(x, tb["meta"], tb["mask"], tb["plane_ids"], return_logits=True)
                loss = self.criterion(logits, tb["labels"], sample_weight=tb["weight"])
                loss = loss / self.cfg.ACCUM_STEPS
            self.scaler.scale(loss).backward()
            running += float(loss.item()) * self.cfg.ACCUM_STEPS
            n_batches += 1
            if (it + 1) % self.cfg.ACCUM_STEPS == 0 or (it + 1) == len(loader):
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.GRAD_CLIP_NORM)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.ema.update(self.model)
                lr = self.scheduler.step()
                if hasattr(pbar, "set_postfix"):
                    pbar.set_postfix(loss=running / max(1, n_batches), lr=f"{lr:.2e}")
        return running / max(1, n_batches)

    # ------------------------------------------------------------- validation
    @torch.no_grad()
    def evaluate(self, loader: DataLoader, use_ema: bool = True) -> Tuple[float, Dict[str, float], pd.DataFrame]:
        weights_backup: Optional[Dict[str, torch.Tensor]] = None
        if use_ema:
            weights_backup = self.ema.apply_to(self.model)
        self.model.eval()
        logger = MetricLogger(self.cfg.TARGET_COLS)
        pbar = _tqdm(loader, desc="valid", leave=False)
        for batch in pbar:
            tb = self._prepare_batch(batch)
            x = self._normalise_tensor(tb["volume"], self.cfg)
            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype,
                                enabled=self.cfg.MIXED_PRECISION and self.device.type == "cuda"):
                probs = self.model(x, tb["meta"], tb["mask"], tb["plane_ids"])
            probs = probs.clamp(1e-6, 1 - 1e-6)
            logger.update(batch["uid"], probs, tb["labels"])
        macro, per, frame = logger.compute()
        if use_ema and weights_backup is not None:
            self.ema.restore(self.model, weights_backup)
        return macro, per, frame

    # ------------------------------------------------------------ checkpoint
    def save_best(self, path: Path, macro: float, fold: int, extra: Dict[str, Any]) -> None:
        payload = {
            "model_state": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
            "ema_state": {k: v.cpu() for k, v in self.ema.shadow.items()},
            "ema_buffers": {k: v.cpu() for k, v in self.ema.buffers.items()},
            "macro_auc": float(macro), "fold": int(fold),
            "config": {**self.cfg.to_dict(), **extra},
            "backbone": extra.get("backbone", self.cfg.BACKBONE),
        }
        torch.save(payload, path)
        print(f"[ckpt] saved fold={fold} macro_auc={macro:.5f} -> {path}")

    def load_weights(self, path: Path, prefer_ema: bool = True) -> float:
        obj = torch.load(path, map_location=self.device, weights_only=False)
        sd = obj["ema_state"] if (prefer_ema and "ema_state" in obj) else obj["model_state"]
        msd = self.model.state_dict()
        fixed: Dict[str, torch.Tensor] = {}
        for k, v in sd.items():
            kk = k.replace("module.", "")
            if kk in msd and msd[kk].shape == v.shape:
                fixed[kk] = v.to(msd[kk].dtype)
        msd.update(fixed)
        self.model.load_state_dict(msd, strict=False)
        if "ema_state" in obj:
            self.ema.load_state_dict({"shadow": obj["ema_state"],
                                      "buffers": obj.get("ema_buffers", {}),
                                      "decay": self.cfg.EMA_DECAY})
        print(f"[ckpt] loaded {path.name} (ema={prefer_ema}) auc={obj.get('macro_auc', float('nan')):.5f}")
        return float(obj.get("macro_auc", 0.0))


# ---------------------------------------------------------------- fold driver
def run_fold(trainer: KneeTrainer, tr_loader: DataLoader, va_loader: DataLoader,
             fold: int, epochs: int, out_dir: Path, backbone_name: str) -> Dict[str, Any]:
    """Train/validate a single fold, keep best checkpoint, return OOF frame + score."""
    trainer.configure_optimiser(len(tr_loader) // max(1, trainer.cfg.ACCUM_STEPS), epochs)
    best_auc, best_epoch, best_frame = -1.0, -1, None
    ckpt = out_dir / f"knee_{backbone_name.replace('.', '_')}_fold{fold}.pt"
    for ep in range(epochs):
        t0 = time.time()
        tr_loss = trainer.train_one_epoch(tr_loader, ep)
        macro, per, frame = trainer.evaluate(va_loader, use_ema=True)
        dt = time.time() - t0
        trainer.history.append({"fold": fold, "epoch": ep, "loss": tr_loss,
                                "macro_auc": macro, "sec": dt})
        print(MetricLogger.format(per, macro) + f" | ep={ep} loss={tr_loss:.4f} {dt:.1f}s")
        if macro > best_auc:
            best_auc, best_epoch, best_frame = macro, ep, frame
            trainer.save_best(ckpt, macro, fold, {"backbone": backbone_name, "epoch": ep})
        if trainer.cfg.time_left_for_training() <= 0:
            print("[time] budget guard triggered -> early stop this fold")
            break
    return {"fold": fold, "best_auc": best_auc, "best_epoch": best_epoch,
            "oof": best_frame, "ckpt": str(ckpt)}


def train_cross_backbone_cv(
    records: pd.DataFrame,
    series_dirs: Sequence[Path],
    backbones: Sequence[str],
    cfg: Config = CFG,
) -> Dict[str, Any]:
    """Full multi-backbone, multi-fold training orchestration with OOF aggregation."""
    splits = build_folds(records, cfg)
    results: Dict[str, Any] = {"folds": [], "oof": {}, "auc": {}}
    out_dir = Path(cfg.ARTIFACT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    for bb in backbones:
        fold_frames: List[pd.DataFrame] = []
        aucs: List[float] = []
        for fi, (tr_idx, va_idx) in enumerate(splits):
            if cfg.time_left_for_training() <= 30 * 60:
                print("[time] skipping remaining folds to protect inference budget")
                break
            tr_df = records.iloc[tr_idx]
            va_df = records.iloc[va_idx]
            tr_ds = RSNAKneeDataset(tr_df, series_dirs, cfg, train=True)
            va_ds = RSNAKneeDataset(va_df, series_dirs, cfg, train=False)
            tr_loader = DataLoader(tr_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                                   num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY,
                                   collate_fn=knee_collate, drop_last=False,
                                   prefetch_factor=cfg.PREFETCH_FACTOR if cfg.NUM_WORKERS else None,
                                   persistent_workers=cfg.PERSISTENT_WORKERS and cfg.NUM_WORKERS > 0)
            va_loader = DataLoader(va_ds, batch_size=cfg.EVAL_BATCH_SIZE, shuffle=False,
                                   num_workers=max(1, cfg.NUM_WORKERS // 2), pin_memory=cfg.PIN_MEMORY,
                                   collate_fn=knee_collate)
            model = KneeAbnormalityModel(cfg, backbone_name=bb, pretrained=True)
            trainer = KneeTrainer(model, cfg)
            res = run_fold(trainer, tr_loader, va_loader, fi, cfg.EPOCHS, out_dir, bb)
            aucs.append(res["best_auc"])
            if res["oof"] is not None:
                fold_frames.append(res["oof"])
            del model, trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        oof = pd.concat(fold_frames, ignore_index=True) if fold_frames else None
        results["folds"].append({"backbone": bb, "aucs": aucs,
                                 "mean_auc": float(np.mean(aucs)) if aucs else 0.5})
        results["oof"][bb] = oof
        print(f"[cv] backbone={bb} mean OOF macro-AUC={np.mean(aucs) if aucs else 0:.5f}")
    return results


print("[train] trainer + StratifiedGroupKFold + warmup/cosine + EMA ready")
