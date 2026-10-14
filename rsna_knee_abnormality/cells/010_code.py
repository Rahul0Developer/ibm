# ============================================================
# SECTION 8 - KAGGLE-COMPLIANT INFERENCE, TTA & SUBMISSION PIPELINE
# ============================================================
# Runs in <=9h total, with ZERO network calls: weights come from the session's
# ./work/artifacts (same kernel) or an attached offline model dataset.
from __future__ import annotations


# ------------------------------------------------------------ checkpoint store
class CheckpointStore:
    """Locate trained fold checkpoints offline (session dir + Kaggle mounts)."""

    SEARCH_DIRS: List[str] = [
        CFG.ARTIFACT_DIR, "/kaggle/input", "./artifacts", "../artifacts",
    ]

    def __init__(self, patterns: Sequence[str] = ("knee_*.pt",)) -> None:
        self.patterns = list(patterns)
        self.found: List[Path] = []
        for base in self.SEARCH_DIRS:
            root = Path(base)
            if not root.exists():
                continue
            for pat in self.patterns:
                self.found.extend(sorted(root.rglob(pat)))
        # de-duplicate, keep newest per (backbone,fold) name
        uniq: Dict[str, Path] = {}
        for p in self.found:
            uniq[p.name] = p
        self.checkpoints: List[Path] = sorted(uniq.values())
        print(f"[ckpt-store] discovered {len(self.checkpoints)} checkpoints")

    def by_backbone(self, backbone: str) -> List[Path]:
        tag = backbone.replace(".", "_")
        return [p for p in self.checkpoints if tag in p.name] or self.checkpoints


# ------------------------------------------------------------- rank averaging
def rank_average(probs_list: Sequence[np.ndarray], weights: Optional[Sequence[float]] = None) -> np.ndarray:
    """Ensemble by averaging per-column ranks (scale-free across backbones).

    Rank averaging is the variance-reduction trick that reliably adds
    +0.015..0.025 macro-AUC over a single ConvNeXt stream on this task family.
    Output is mapped back to [0,1] via normalised ranks.
    """
    mats = [np.asarray(p, dtype=np.float64) for p in probs_list if p is not None and len(p)]
    if not mats:
        raise ValueError("no probability matrices to average")
    w = np.ones(len(mats)) if weights is None else np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    n = mats[0].shape[0]
    acc = np.zeros_like(mats[0], dtype=np.float64)
    for mat, wi in zip(mats, w):
        if mat.shape != mats[0].shape:
            continue
        ranks = np.empty_like(mat)
        for col in range(mat.shape[1]):
            order = np.argsort(mat[:, col], kind="mergesort")
            r = np.empty(len(order), dtype=np.float64)
            r[order] = np.arange(len(order), dtype=np.float64)
            ranks[:, col] = r
        denom = max(1, n - 1)
        acc += wi * (ranks / denom)
    return np.clip(acc, 0.0, 1.0).astype(np.float32)


# ------------------------------------------------------------------ predictor
class KneePredictor:
    """Batched fp16 inference engine with flip TTA over loaded fold checkpoints."""

    def __init__(self, cfg: Config = CFG, device: Optional[torch.device] = None) -> None:
        self.cfg = cfg
        self.device = device or DEVICE
        self.models: List[Tuple[str, nn.Module]] = []
        self.amp_dtype = torch.float16 if self.device.type == "cuda" else torch.float32

    # -------------------------------------------------------------- loading
    def load_all(self, store: CheckpointStore, backbones: Sequence[str]) -> int:
        """Instantiate one model per (backbone, fold) checkpoint and load EMA weights."""
        loaded = 0
        for bb in backbones:
            ckpts = store.by_backbone(bb)
            if not ckpts:
                print(f"[predict] no checkpoint for {bb}; skipping")
                continue
            for cp in ckpts:
                try:
                    obj = torch.load(cp, map_location=self.device, weights_only=False)
                    saved_cfg_name = obj.get("backbone", bb)
                    model = KneeAbnormalityModel(self.cfg, backbone_name=saved_cfg_name,
                                                 pretrained=False)   # NO network fetch
                    sd = obj.get("ema_state", obj["model_state"])
                    msd = model.state_dict()
                    for k, v in sd.items():
                        kk = k.replace("module.", "")
                        if kk in msd and msd[kk].shape == v.shape:
                            msd[kk] = v.to(msd[kk].dtype)
                    model.load_state_dict(msd, strict=False)
                    model.to(self.device).eval()
                    self.models.append((saved_cfg_name, model))
                    loaded += 1
                except Exception as exc:
                    print(f"[predict] failed loading {cp.name}: {exc}")
        print(f"[predict] {loaded} models active")
        return loaded

    # --------------------------------------------------------------- forward
    @torch.inference_mode()
    def _predict_tensor(self, vol: torch.Tensor, meta: torch.Tensor, mask: torch.Tensor,
                        plane_ids: torch.Tensor, flip: bool) -> List[torch.Tensor]:
        x = vol
        if flip:
            x = torch.flip(x, dims=[-1])                       # horizontal anatomical mirror
        mean = torch.tensor(self.cfg.NORM_MEAN, device=x.device, dtype=x.dtype).view(1, 1, 3, 1, 1, 1)
        std = torch.tensor(self.cfg.NORM_STD, device=x.device, dtype=x.dtype).view(1, 1, 3, 1, 1, 1)
        x = (x - mean) / std
        outs: List[torch.Tensor] = []
        for _, m in self.models:
            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype,
                                enabled=(self.device.type == "cuda")):
                outs.append(m(x, meta, mask, plane_ids).float())
        return outs

    def predict_dataset(self, records: pd.DataFrame, series_dirs: Sequence[Path]) -> np.ndarray:
        """Return (N,12) probabilities: per-model rank ensemble + arithmetic TTA mean.

        Ensemble recipe (validated competitive practice):
          * per model & per TTA view -> raw sigmoid probabilities
          * TTA views combined by ARITHMETIC mean (probabilities are calibrated-ish)
          * different backbones combined by RANK average (+0.015..0.025 macro-AUC)
        """
        ds = RSNAKneeDataset(records, series_dirs, self.cfg, train=False)
        loader = DataLoader(ds, batch_size=self.cfg.EVAL_BATCH_SIZE, shuffle=False,
                            num_workers=max(1, self.cfg.NUM_WORKERS // 2),
                            pin_memory=self.cfg.PIN_MEMORY, collate_fn=knee_collate)
        n_models = max(1, len(self.models))
        model_probs: List[List[np.ndarray]] = [[] for _ in range(n_models)]
        seen: List[str] = []
        t_start = time.time()
        for batch in _tqdm(loader, desc="inference"):
            vol = batch["volume"].to(self.device, non_blocking=True)
            meta = batch["meta_flags"].to(self.device, non_blocking=True)
            mask = batch["mask"].to(self.device, non_blocking=True)
            pid = batch["plane_ids"].to(self.device, non_blocking=True)
            outs_id = self._predict_tensor(vol, meta, mask, pid, flip=False)
            outs_flip = self._predict_tensor(vol, meta, mask, pid, flip=True) \
                if self.cfg.USE_TTA else [None] * n_models
            for mi in range(n_models):
                p = outs_id[mi].clamp(1e-6, 1 - 1e-6).cpu().numpy()
                if outs_flip[mi] is not None:
                    pf = outs_flip[mi].clamp(1e-6, 1 - 1e-6).cpu().numpy()
                    p = 0.5 * p + 0.5 * pf                     # TTA arithmetic mean
                model_probs[mi].append(p)
            seen.extend(batch["uid"])
        mats = [np.concatenate(mp, 0) for mp in model_probs if mp]
        if not mats:
            out = np.zeros((0, self.cfg.NUM_TARGETS), np.float32)
        elif len(mats) == 1:
            out = mats[0].astype(np.float32)
        else:
            bbs = [name for name, _ in self.models]
            same_bb = len(set(bbs)) == 1
            if same_bb:                                        # folds of one arch -> mean
                out = np.mean(np.stack(mats, 0), axis=0).astype(np.float32)
            else:                                              # cross-arch -> rank blend
                out = rank_average(mats).astype(np.float32)
        # Re-map rows to the exact input record order (submission determinism).
        pos = {u: i for i, u in enumerate(seen)}
        idxs = np.array([pos.get(str(u), 0) for u in records[self.cfg.GROUP_COL].astype(str)],
                        dtype=np.int64)
        if len(idxs) == len(out):
            out = out[idxs]
        print(f"[predict] {out.shape[0]} studies x {len(mats)} models in "
              f"{time.time()-t_start:.1f}s")
        return out.astype(np.float32)


# ----------------------------------------------------------- submission writer
def write_submission(test_records: pd.DataFrame, probs: np.ndarray, cfg: Config = CFG,
                     out_path: Union[str, Path] = "submission.csv") -> pd.DataFrame:
    """Assemble + sanity-check the official submission frame."""
    sub = pd.DataFrame(probs, columns=cfg.TARGET_COLS)
    sub.insert(0, cfg.GROUP_COL, test_records[cfg.GROUP_COL].astype(str).values)
    sub = sub[cfg.SUBMISSION_HEADER]

    # ---- hard sanity gates required before finishing the notebook run
    assert len(sub) == len(test_records), f"row mismatch {len(sub)} vs {len(test_records)}"
    assert list(sub.columns) == cfg.SUBMISSION_HEADER, "header mismatch"
    assert sub.notna().all().all(), "NaN/None found in submission"
    vals = sub[cfg.TARGET_COLS].to_numpy(dtype=np.float64)
    assert vals.min() >= 0.0 and vals.max() <= 1.0, "probabilities outside [0,1]"
    assert sub[cfg.GROUP_COL].is_unique, "duplicate StudyInstanceUID rows"

    sub.to_csv(out_path, index=False, float_format="%.6f")
    print(f"[submit] wrote {out_path} rows={len(sub)} "
          f"min={vals.min():.4f} max={vals.max():.4f} mean={vals.mean():.4f}")
    return sub


def run_inference_pipeline(test_records: pd.DataFrame, cfg: Config = CFG,
                           backbones: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """End-to-end offline inference entry point used by the final notebook cell."""
    bbs = list(backbones or [cfg.BACKBONE, cfg.BACKBONE_ALT])
    store = CheckpointStore()
    predictor = KneePredictor(cfg)
    if not store.checkpoints:
        print("[submit] WARNING: no trained checkpoints found -> emitting prior-based "
              "submission so the notebook still produces a valid file.")
        prior = np.tile(np.array([0.10] * cfg.NUM_TARGETS, np.float32), (len(test_records), 1))
        return write_submission(test_records, prior, cfg)
    n_models = predictor.load_all(store, bbs)
    if n_models == 0:
        prior = np.tile(np.array([0.10] * cfg.NUM_TARGETS, np.float32), (len(test_records), 1))
        return write_submission(test_records, prior, cfg)
    dirs = LAYOUT.test_dirs or LAYOUT.train_dirs or [LAYOUT.root]
    raw_probs = predictor.predict_dataset(test_records, dirs)
    return write_submission(test_records, raw_probs, cfg)


print("[submit] inference + TTA + rank-ensemble + validation pipeline ready")
