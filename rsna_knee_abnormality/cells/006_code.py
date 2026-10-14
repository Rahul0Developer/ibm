# ============================================================
# SECTION 4 - DATASET, MULTI-VIEW COLLATOR & AUGMENTATION
# ============================================================
from __future__ import annotations


# ------------------------------------------------------------ augmentation ops
def _to_uint8(vol: np.ndarray) -> np.ndarray:
    """[0,1] float volume -> uint8 for albumentations speed/compat."""
    return np.clip(vol * 255.0, 0, 255).astype(np.uint8)


def _from_uint8(arr: np.ndarray) -> np.ndarray:
    return arr.astype(np.float32) / 255.0


class SliceConsistentAug:
    """Apply the SAME spatial transform to every slice of a series (depth coherence).

    Albumentations is instantiated once per call with a shared RNG seed so that all
    K frames receive identical RandomAffine / ShiftScaleRotate / HorizontalFlip /
    CoarseDropout parameters. Photometric noise is applied independently per slice
    (physically realistic: acquisition noise varies through depth).
    """

    def __init__(self, cfg: Config = CFG, strong: bool = True) -> None:
        self.cfg = cfg
        self.strong = strong and cfg.AUG_STRONG
        self._have_alb = False
        try:
            import albumentations as A
            self.A = A
            self._have_alb = True
        except Exception:
            self.A = None
        if not self._have_alb:
            print("[aug] albumentations unavailable -> numpy-only geometric fallback")

    # -------------------------------------------------- geometric transforms
    def _numpy_affine(self, vol: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        h, w = vol.shape[-2:]
        scale = rng.uniform(*self.cfg.AFFINE_SCALE)
        ang = math.radians(rng.uniform(-self.cfg.AFFINE_ROTATE, self.cfg.AFFINE_ROTATE))
        dx = rng.uniform(-self.cfg.AFFINE_SHIFT, self.cfg.AFFINE_SHIFT) * w
        dy = rng.uniform(-self.cfg.AFFINE_SHIFT, self.cfg.AFFINE_SHIFT) * h
        cos, sin = math.cos(ang), math.sin(ang)
        cx, cy = w / 2.0, h / 2.0
        xs = np.arange(w, dtype=np.float32) - cx
        ys = np.arange(h, dtype=np.float32) - cy
        X, Y = np.meshgrid(xs, ys)
        map_x = (cos * X + sin * Y) / scale + cx - dx
        map_y = (-sin * X + cos * Y) / scale + cy - dy
        out = np.empty_like(vol)
        for i in range(vol.shape[0]):
            xi = np.clip(map_x, 0, w - 1).astype(np.int32)
            yi = np.clip(map_y, 0, h - 1).astype(np.int32)
            out[i] = vol[i][yi, xi]
        return out

    def _numpy_coarse_dropout(self, vol: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        h, w = vol.shape[-2:]
        n_boxes = int(rng.integers(1, 4))
        out = vol.copy()
        gh = max(8, h // self.cfg.DROPOUT_GRID)
        gw = max(8, w // self.cfg.DROPOUT_GRID)
        for _ in range(n_boxes):
            y0 = int(rng.integers(0, max(1, h - gh)))
            x0 = int(rng.integers(0, max(1, w - gw)))
            bh = int(rng.integers(gh // 2, gh + 1))
            bw = int(rng.integers(gw // 2, gw + 1))
            out[:, y0:y0 + bh, x0:x0 + bw] = 0.0
        return out

    # -------------------------------------------------------------- public API
    def __call__(self, vol: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """vol: (K,H,W) float32 in [0,1]. Returns augmented (K,H,W) float32."""
        rng = rng or np.random.default_rng()
        if vol.size == 0:
            return vol
        if self._have_alb:
            A = self.A
            ops = [
                A.HorizontalFlip(p=self.cfg.HFLIP_PROB),
                A.ShiftScaleRotate(
                    shift_limit=self.cfg.AFFINE_SHIFT,
                    scale_limit=(self.cfg.AFFINE_SCALE[0] - 1.0, self.cfg.AFFINE_SCALE[1] - 1.0),
                    rotate_limit=self.cfg.AFFINE_ROTATE,
                    border_mode=0, value=0, p=1.0,
                ),
                A.CoarseDropout(
                    max_holes=3,
                    max_height=self.cfg.IMAGE_SIZE[0] // self.cfg.DROPOUT_GRID,
                    max_width=self.cfg.IMAGE_SIZE[1] // self.cfg.DROPOUT_GRID,
                    min_holes=1, fill_value=0, p=0.5,
                ),
            ]
            pipeline = A.ReplayCompose(ops)
            u8 = _to_uint8(vol)
            data = {"images": [u8[i] for i in range(u8.shape[0])]}
            try:
                out = pipeline(**data, replay=None)
                frames = [d["image"] for d in out["images"]]
                arr = np.stack(frames, axis=0)
                arr = _from_uint8(arr)
            except Exception:
                arr = self._numpy_affine(vol, rng)
        else:
            arr = self._numpy_affine(vol, rng)

        if rng.random() < self.cfg.HFLIP_PROB:      # belt-and-braces flip (non-alb path)
            if not self._have_alb:
                arr = arr[:, :, ::-1].copy()
        arr = self._numpy_coarse_dropout(arr, rng)
        # photometric jitter, independent per slice
        noise = rng.normal(0.0, self.cfg.NOISE_STD, size=arr.shape).astype(np.float32)
        bright = rng.uniform(*self.cfg.BRIGHTNESS)
        arr = np.clip(arr + noise + bright, 0.0, 1.0)
        return arr.astype(np.float32)


# ------------------------------------------------------------------- dataset
LABEL_PSEUDO_COLS: List[str] = [f"{t}_pseudo" for t in TARGET_COLS]


class RSNAKneeDataset(Dataset):
    """Study-level multi-plane dataset.

    Each item = one StudyInstanceUID with up to 3 plane streams
    (Sagittal, Coronal, Axial); missing planes are zero-padded and flagged
    by `mask` so the model can gate them out.
    """

    PLANES_WANTED: Tuple[int, ...] = (PLANE_SAGITTAL, PLANE_CORONAL, PLANE_AXIAL)

    def __init__(
        self,
        records: pd.DataFrame,
        series_dirs: Sequence[Path],
        cfg: Config = CFG,
        train: bool = True,
        reader: Optional[DicomSeriesReader] = None,
    ) -> None:
        self.records = records.reset_index(drop=True)
        self.series_dirs = [Path(p) for p in series_dirs]
        self.cfg = cfg
        self.train = train
        self.reader = reader or READER
        self.aug = SliceConsistentAug(cfg, strong=train) if train else None
        self._mem_cache: Dict[str, Dict[str, Any]] = {}
        # label columns actually present
        if all(c in self.records.columns for c in cfg.TARGET_COLS):
            self.label_cols = list(cfg.TARGET_COLS)
        elif all(c in self.records.columns for c in LABEL_PSEUDO_COLS):
            self.label_cols = LABEL_PSEUDO_COLS
        else:
            self.label_cols = []
        self.has_labels = len(self.label_cols) == cfg.NUM_TARGETS

    # ------------------------------------------------------------------ utils
    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _study_dirs(root: Path, uid: str) -> List[Path]:
        """Return candidate folders holding this study's DICOMs (both layouts)."""
        out: List[Path] = []
        direct = root / uid
        if direct.is_dir():
            out.append(direct)
        # flat layout: files named <uid>_*.dcm directly under root
        if not out and root.is_dir():
            out.append(root)
        return out

    def _zero_pack(self) -> Dict[str, Any]:
        k, (h, w) = self.cfg.NUM_SLICES, self.cfg.IMAGE_SIZE
        p = self.cfg.NUM_PLANES
        return {
            "volumes": np.zeros((p, self.cfg.IN_CHANNELS, k, h, w), np.float32),
            "plane_ids": np.asarray(self.PLANES_WANTED, np.int64),
            "meta_flags": np.zeros((p, 3), np.float32),
            "mask": np.zeros((p,), np.float32),
        }

    # --------------------------------------------------------------- getitem
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.records.iloc[idx]
        uid = str(row[self.cfg.GROUP_COL])
        rng = np.random.default_rng(self.cfg.SEED + idx * 977 + (0 if not self.train else 1_000_003))

        pack = self._zero_pack()
        rec: Optional[Dict[str, Any]] = None
        if self.cfg.ENABLE_CACHING and uid in self._mem_cache:
            rec = self._mem_cache[uid]
        else:
            dirs: List[Path] = []
            for root in self.series_dirs:
                dirs.extend(self._study_dirs(root, uid))
            if dirs:
                try:
                    rec = self.reader.load_study(dirs, uid, self.PLANES_WANTED)
                except Exception as exc:                       # corrupt study -> zeros
                    print(f"[dataset] load failed {uid}: {exc}")
                    rec = None
            if rec is not None and self.cfg.ENABLE_CACHING:
                self._mem_cache[uid] = rec

        if rec is not None:
            vols = rec["volumes"]                              # (P,K,H,W)
            if self.aug is not None:
                vols = np.stack([self.aug(vols[i], rng) for i in range(vols.shape[0])], 0)
            pack["volumes"] = np.repeat(vols[:, None], self.cfg.IN_CHANNELS, axis=1)  # (P,3,K,H,W)
            pack["plane_ids"] = np.asarray(rec["plane_ids"], np.int64)[: self.cfg.NUM_PLANES]
            pack["meta_flags"] = np.asarray(rec["meta_flags"], np.float32)[: self.cfg.NUM_PLANES]
            pack["mask"] = np.asarray(rec["mask"], np.float32)[: self.cfg.NUM_PLANES]

        labels = np.zeros(self.cfg.NUM_TARGETS, np.float32)
        weight = 0.0
        if self.has_labels:
            labels = row[self.label_cols].astype(np.float32).to_numpy()
            weight = float(row.get("label_weight", 1.0)) if "label_weight" in row else 1.0

        return {
            "uid": uid,
            "volume": torch.from_numpy(pack["volumes"]),       # (P,C,K,H,W)
            "plane_ids": torch.from_numpy(pack["plane_ids"]),
            "meta_flags": torch.from_numpy(pack["meta_flags"]),
            "mask": torch.from_numpy(pack["mask"]),
            "labels": torch.from_numpy(labels),
            "weight": torch.tensor(weight, dtype=torch.float32),
        }


# ----------------------------------------------------------------- collator
def knee_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Stack fixed-geometry tensors; keep UIDs as python list for OOF bookkeeping."""
    return {
        "uid": [b["uid"] for b in batch],
        "volume": torch.stack([b["volume"] for b in batch], 0),      # (B,P,C,K,H,W)
        "plane_ids": torch.stack([b["plane_ids"] for b in batch], 0),
        "meta_flags": torch.stack([b["meta_flags"] for b in batch], 0),
        "mask": torch.stack([b["mask"] for b in batch], 0),
        "labels": torch.stack([b["labels"] for b in batch], 0),
        "weight": torch.stack([b["weight"] for b in batch], 0),
    }


# ------------------------------------------------------------- demo/synthetic
def make_synthetic_records(cfg: Config = CFG, n_train: int = 24, n_test: int = 6,
                           gt_fraction: float = 0.35, seed: int = 7) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build fake train/test frames when no competition mount exists.

    Lets the FULL pipeline (loader->model->ASL->CV->submission) execute end-to-end
    on CPU/laptop so the notebook is verifiably runnable before Kaggle attach.
    """
    rng = np.random.default_rng(seed)
    reports = [
        "No evidence of ACL tear. Menisci are intact. No effusion. Mild medial compartment osteoarthritis.",
        "Complete tear of the ACL. Large joint effusion with Baker's cyst posteriorly. Bone marrow edema consistent with contusion.",
        "MCL grade 2 sprain. Medial meniscus posterior horn tear. Small suprapatellar effusion. Patellofemoral joint space narrowing.",
        "Unremarkable knee MRI. Intact ligaments and menisci. No fracture or bone marrow lesion.",
        "Non-displaced tibial plateau fracture with extensive marrow edema. Lateral meniscus root avulsion. Synovitis noted.",
        "Moderate lateral compartment osteoarthritis with osteophytes. Lateral meniscus complex tear. No ACL tear.",
    ]
    def frame(n: int, labelled: bool) -> Tuple[pd.DataFrame, np.ndarray]:
        uids = [f"synth_{i:05d}" for i in range(n)]
        df = pd.DataFrame({cfg.GROUP_COL: uids})
        lab = rng.binomial(1, 0.18, size=(n, cfg.NUM_TARGETS)).astype(np.float32)
        for i, t in enumerate(cfg.TARGET_COLS):
            df[t] = lab[:, i] if labelled else np.nan
        df["Report"] = [reports[int(rng.integers(0, len(reports)))] for _ in range(n)]
        return df, lab

    train_df, lab = frame(n_train, labelled=False)
    gt_rows = max(1, int(n_train * gt_fraction))
    for i in range(gt_rows):                     # human-annotated subset
        for j, t in enumerate(cfg.TARGET_COLS):
            train_df.loc[i, t] = float(lab[i, j])
    test_df = pd.DataFrame({cfg.GROUP_COL: [f"synth_t_{i:05d}" for i in range(n_test)]})
    return train_df, test_df


print("[dataset] RSNAKneeDataset + ReplayCompose slice-consistent augmentation ready")
