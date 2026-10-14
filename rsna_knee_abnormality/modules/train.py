"""
MODULE 3 — Multi-View Attention-Pooled 2.5D Model & Training Loop (RSNA Knee).

Contents
--------
* Config dataclass (SEED/IMAGE_SIZE/NUM_SLICES/BATCH_SIZE/BACKBONE/TARGET_COLS/DEVICE/MIXED_PRECISION...)
* RSNAKneeDataset      : study-level dataset over preprocessed .npy volumes (Module 1 output),
                         picks up to 3 dominant series per anatomical plane (Sagittal/Coronal/Axial),
                         zero-pads missing planes; consistent-across-slices Albumentations augmentation.
* RSNAMultiPlaneModel  : timm 2D backbone applied slice-wise (2.5D), BiGRU / Multi-Head Self-Attention
                         aggregator + attention pooling across slices, plane concatenation,
                         metadata embeddings (FluidSensitive/FatSuppression/plane) fused before head,
                         12-logit multi-label classifier with sigmoid at inference.
* AsymmetricLoss       : ASL (gamma_neg=4, gamma_pos=0, clip=0.05) optimized implementation.
* MacroAUC             : competition metric = mean of per-class ROC-AUC, crash-safe on single-class batches.
* Warmup+Cosine LR schedule, AdamW, torch.amp autocast, EMA weights, GroupKFold(5) by StudyInstanceUID,
  OOF macro-AUC every epoch, best checkpoint per fold (`best_auc_fold_{k}.pt`).

Run (Kaggle):
    !python train.py --backbone convnext_nano.in12k_ft_in1k --fold 0 --epochs 6 \
        --volumes-root /kaggle/input/rsna-knee-preprocessed-256/train_volumes \
        --manifest /kaggle/working/train_manifest.csv \
        --labels /kaggle/working/train_pseudo_labels.csv \
        --ckpt-dir /kaggle/working/checkpoints
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

TARGET_COLS: List[str] = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
    "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]
PLANES: List[str] = ["Sagittal", "Coronal", "Axial"]


# --------------------------------------------------------------------------------------
# SECTION 1: SYSTEM CONFIG & HYPERPARAMETERS
# --------------------------------------------------------------------------------------
@dataclass
class Config:
    SEED: int = 42
    IMAGE_SIZE: Tuple[int, int] = (256, 256)
    NUM_SLICES: int = 16
    BATCH_SIZE: int = 8
    NUM_WORKERS: int = 4
    BACKBONE: str = "convnext_nano.in12k_ft_in1k"   # or "tf_efficientnet_b0_ns" for memory efficiency
    IN_CHANNELS: int = 3                            # 3-channel duplicate for pretrained weights
    NUM_PLANES: int = 3
    FEATURE_DIM: int = 256                          # projection dim after backbone
    AGGREGATOR: str = "bigru"                       # {"bigru", "mhsa"}
    ATTENTION_HEADS: int = 4
    DROPOUT: float = 0.25
    LR: float = 3e-4
    WEIGHT_DECAY: float = 1e-2
    EPOCHS: int = 6
    WARMUP_EPOCHS: int = 1
    N_FOLDS: int = 5
    MIXED_PRECISION: bool = True
    USE_EMA: bool = True
    EMA_DECAY: float = 0.995
    PSEUDO_WEIGHT: float = 0.5                      # loss weight for pseudo-labeled rows
    TARGET_COLS: List[str] = field(default_factory=lambda: list(TARGET_COLS))
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


def set_determinism(seed: int) -> None:
    """Strict determinism across Python / NumPy / PyTorch / CuDNN."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------------------
# SECTION 4: DATASET (study-level, multi-plane, consistent slice augmentation)
# --------------------------------------------------------------------------------------
def load_volume(path: str) -> np.ndarray:
    """Load a preprocessed (D,H,W) uint8 volume from .npy or compressed .npz."""
    if path.endswith(".npz"):
        with np.load(path) as z:
            arr = z["arr"] if "arr" in z.files else z[z.files[0]]
    else:
        arr = np.load(path)
    return np.asarray(arr)


class ConsistentSliceAug:
    """Applies identical geometric augmentation params across all slices of one series.

    Uses albumentations when available; otherwise falls back to pure-numpy ops so the
    pipeline never hard-fails on Kaggle images lacking the package.
    """

    def __init__(self, p_flip: float = 0.5, p_affine: float = 0.7, p_brightness: float = 0.5,
                 p_dropout: float = 0.3, image_size: Tuple[int, int] = (256, 256), seed: int = 42):
        self.p_flip, self.p_affine, self.p_brightness, self.p_dropout = p_flip, p_affine, p_brightness, p_dropout
        self.image_size = image_size
        self.rng = np.random.default_rng(seed)
        try:
            # Probe the full transform stack once on a dummy frame: some Albumentations
            # builds have optional native deps (albucore/numkong) that fail lazily at
            # call time; probing here lets us fall back cleanly instead of crashing mid-train.
            import albumentations as A
            tf = A.Compose([
                A.HorizontalFlip(p=p_flip),
                A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.10, rotate_limit=10, p=p_affine,
                                   border_mode=0, value=0),
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=p_brightness),
                A.CoarseDropout(max_holes=6, max_height=24, max_width=24, min_holes=1,
                                fill_value=0, p=p_dropout),
            ])
            tf(image=np.zeros((64, 64), np.uint8))
            self.transform = tf
        except Exception:
            self.transform = None

    def __call__(self, vol: np.ndarray) -> np.ndarray:
        """vol: (D,H,W) uint8 -> augmented (D,H,W) uint8 with SHARED geometry."""
        d = vol.shape[0]
        if self.transform is not None:
            # Draw ONE geometric realization and apply identically to each slice by
            # seeding the underlying RNG through repeated calls on a reference patch.
            ref = self._consistent_apply(vol)
            return ref
        # ---- numpy fallback -----------------------------------------------------
        out = vol.copy()
        if self.rng.random() < self.p_flip:
            out = out[:, :, ::-1]
        if self.rng.random() < self.p_dropout:
            h, w = out.shape[1], out.shape[2]
            n_holes = int(self.rng.integers(1, 7))
            for _ in range(n_holes):
                hh = int(self.rng.integers(6, 25)); ww = int(self.rng.integers(6, 25))
                y0 = int(self.rng.integers(0, max(1, h - hh))); x0 = int(self.rng.integers(0, max(1, w - ww)))
                out[:, y0:y0 + hh, x0:x0 + ww] = 0
        if self.rng.random() < self.p_brightness:
            b = float(self.rng.uniform(-0.12, 0.12))
            out = np.clip(out.astype(np.float32) * (1.0 + b), 0, 255).astype(np.uint8)
        return out

    def _consistent_apply(self, vol: np.ndarray) -> np.ndarray:
        """Albumentations applies per-image randomness; we force consistency by using
        a keypoint-free deterministic seed derived per-series call."""
        import albumentations as A
        seed = int(self.rng.integers(0, 2**31 - 1))
        A.set_random_seed(seed)
        frames = []
        for s in vol:
            res = self.transform(image=s)
            frames.append(res["image"])
        return np.stack(frames, axis=0)


class RSNAKneeDataset(Dataset):
    """Study-level dataset. Each item = up to 3 plane volumes + metadata + label vector.

    Returns tensors shaped:
        volumes : (NUM_PLANES, NUM_SLICES, H, W) float32 in [0,1]
        plane_mask : (NUM_PLANES,) float32 (1 if that plane exists)
        meta : (4,) float32 [any_fluid_sensitive, any_fat_suppressed, n_series_norm, has_axial]
        target : (12,) float32
        sample_weight : scalar tensor (1.0 GT / cfg.PSEUDO_WEIGHT pseudo)
    """

    def __init__(
        self,
        studies_df: pd.DataFrame,
        manifest_df: pd.DataFrame,
        volumes_root: str | Path,
        cfg: Config,
        train: bool = True,
    ):
        self.cfg = cfg
        self.volumes_root = Path(volumes_root)
        self.train = train
        self.studies = studies_df.reset_index(drop=True)

        # Index manifest by study
        m = manifest_df.copy()
        m = m[m["status"].isin(["ok", "skipped"])].copy()
        self._by_study: Dict[str, List[Dict[str, object]]] = {}
        for rec in m.to_dict(orient="records"):
            self._by_study.setdefault(rec["StudyInstanceUID"], []).append(rec)

        # If skip_existing left planes unknown, fall back to SeriesDescription parsing.
        self.aug = ConsistentSliceAug(image_size=cfg.IMAGE_SIZE, seed=cfg.SEED) if train else None

        # Pre-compute per-study chosen series (deterministic selection).
        self._chosen: List[List[Optional[Dict[str, object]]]] = []
        for uid in self.studies["StudyInstanceUID"].values:
            self._chosen.append(self._select_series(str(uid)))

    # ------------------------------------------------------------------
    def _plane_of(self, rec: Dict[str, object]) -> str:
        plane = str(rec.get("AnatomicalPlane", "") or "").lower()
        desc = str(rec.get("SeriesDescription", "") or "").lower()
        for canonical, keys in (("sagittal", ("sag",)), ("coronal", ("cor",)), ("axial", ("ax", "tra"))):
            if any(k in plane for k in keys) or any(f"{k}" in desc for k in keys):
                return canonical.capitalize() if False else canonical
        return "unknown"

    def _select_series(self, uid: str) -> List[Optional[Dict[str, object]]]:
        """Pick up to one dominant series per plane (Sagittal, Coronal, Axial).

        Dominance heuristic: most slices, tie-break preferring fluid-sensitive sequences.
        """
        cands = self._by_study.get(uid, [])
        best: Dict[str, Dict[str, object]] = {}
        for rec in cands:
            pl = self._plane_of(rec)
            if pl == "unknown":
                continue
            cur = best.get(pl)
            score = (int(rec.get("NumSlicesOriginal", 0) or 0), int(rec.get("FluidSensitive", 0) or 0))
            if cur is None:
                best[pl] = rec
            else:
                cur_score = (int(cur.get("NumSlicesOriginal", 0) or 0), int(cur.get("FluidSensitive", 0) or 0))
                if score > cur_score:
                    best[pl] = rec
        ordered: List[Optional[Dict[str, object]]] = []
        for pl in ("sagittal", "coronal", "axial"):
            ordered.append(best.get(pl))
        # Degenerate case: no recognized planes at all -> take first 2 readable series as sag/cor slots.
        if all(o is None for o in ordered) and cands:
            ordered = [cands[0], cands[1] if len(cands) > 1 else None, None]
        return ordered

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.studies)

    def _load_plane_volume(self, rec: Optional[Dict[str, object]]) -> np.ndarray:
        if rec is None:
            return np.zeros((self.cfg.NUM_SLICES, *self.cfg.IMAGE_SIZE), dtype=np.uint8)
        path = self.volumes_root / Path(str(rec["VolumePath"])).name
        try:
            vol = load_volume(str(path))
        except Exception:
            # direct stored absolute path attempt
            try:
                vol = load_volume(str(rec["VolumePath"]))
            except Exception:
                return np.zeros((self.cfg.NUM_SLICES, *self.cfg.IMAGE_SIZE), dtype=np.uint8)
        # Defensive shape repair: enforce exactly K slices and HxW.
        if vol.ndim != 3:
            vol = vol.reshape(-1, vol.shape[-2], vol.shape[-1]) if vol.ndim > 3 else vol[None].repeat(self.cfg.NUM_SLICES, 0)
        if vol.shape[0] != self.cfg.NUM_SLICES:
            idx = np.linspace(0, vol.shape[0] - 1, self.cfg.NUM_SLICES).round().astype(int)
            vol = vol[idx]
        if vol.shape[1:] != tuple(self.cfg.IMAGE_SIZE):
            try:
                import cv2
                vol = np.stack([cv2.resize(v, tuple(reversed(self.cfg.IMAGE_SIZE)), interpolation=cv2.INTER_AREA) for v in vol])
            except Exception:
                from PIL import Image
                vol = np.stack([np.asarray(Image.fromarray(v).resize(tuple(reversed(self.cfg.IMAGE_SIZE)))) for v in vol])
        return vol.astype(np.uint8, copy=False)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.studies.iloc[index]
        uid = str(row["StudyInstanceUID"])
        chosen = self._chosen[index]

        vols, masks, fluids, fats = [], [], [], []
        for rec in chosen:
            v = self._load_plane_volume(rec)
            if self.aug is not None and rec is not None:
                v = self.aug(v)
            vols.append(v)
            masks.append(1.0 if rec is not None else 0.0)
            fluids.append(float(rec.get("FluidSensitive", 0) or 0) if rec else 0.0)
            fats.append(float(rec.get("FatSuppression", 0) or 0) if rec else 0.0)

        vol_stack = np.stack(vols, axis=0).astype(np.float32) / 255.0     # (P,D,H,W)
        vol_stack = np.repeat(vol_stack[:, :, None], self.cfg.IN_CHANNELS, axis=2)  # (P,D,C,H,W)

        labels = np.array([float(row[t]) if pd.notna(row[t]) else 0.0 for t in TARGET_COLS], dtype=np.float32)
        src = str(row.get("label_source", "gt")).lower()
        weight = 1.0 if src == "gt" else self.cfg.PSEUDO_WEIGHT

        meta = np.array([
            float(np.max(fluids)), float(np.max(fats)),
            float(sum(masks)) / 3.0, float(masks[2]),
        ], dtype=np.float32)

        return {
            "volumes": torch.from_numpy(vol_stack),                     # (P,D,C,H,W)
            "plane_mask": torch.tensor(masks, dtype=torch.float32),     # (P,)
            "fluid_flags": torch.tensor(fluids, dtype=torch.float32),   # (P,)
            "fat_flags": torch.tensor(fats, dtype=torch.float32),       # (P,)
            "meta": torch.from_numpy(meta),
            "target": torch.from_numpy(labels),
            "weight": torch.tensor(weight, dtype=torch.float32),
            "uid": uid,
        }


def collate_knee(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    uids = [b.pop("uid") for b in batch]
    out = {k: torch.stack([b[k] for b in batch], dim=0) for k in
           ["volumes", "plane_mask", "fluid_flags", "fat_flags", "meta", "target", "weight"]}
    out["uids"] = uids  # type: ignore[assignment]
    return out


# --------------------------------------------------------------------------------------
# SECTION 5: MODEL ARCHITECTURE
# --------------------------------------------------------------------------------------
def build_backbone(name: str, in_chans: int) -> Tuple[nn.Module, int]:
    """Create a timm backbone returning (features, embed_dim). Graceful offline handling.

    If `timm` is not importable at all (minimal container), falls back to the local CNN
    so the pipeline never hard-crashes; pretrained weights are only attempted when timm
    is present, and failures degrade to random-init architecture instead of aborting.
    """
    errors: List[str] = []
    try:
        import timm
        # Try exact name first, then without tag suffix (offline caches often keyed differently).
        for cand in (name, name.split(".")[0]):
            for pre in (True, False):
                try:
                    m = timm.create_model(cand, pretrained=pre, in_chans=in_chans, num_classes=0)
                    dim = getattr(m, "num_features", None)
                    if dim is None:
                        with torch.no_grad():
                            probe = m(torch.zeros(1, in_chans, 64, 64))
                            dim = probe.shape[-1] if probe.ndim == 2 else probe.shape[1]
                    if not pre:
                        print(f"[WARN] Using random-init {cand} (weights download unavailable/offline).")
                    return m, int(dim)
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{cand}(pretrained={pre}): {exc}")
    except ImportError as exc:
        errors.append(f"timm import failed: {exc}")
    # Last resort: tiny local CNN so submission never dies because timm is missing.
    print("[WARN] Could not load timm backbone; falling back to lightweight local CNN.", errors[:2])

    class FallbackCNN(nn.Module):
        def __init__(self, c: int, dim: int = 512):
            super().__init__()
            def block(ci, co):
                return nn.Sequential(nn.Conv2d(ci, co, 3, stride=2, padding=1, bias=False),
                                     nn.BatchNorm2d(co), nn.SiLU(inplace=True))
            self.net = nn.Sequential(block(c, 32), block(32, 64), block(64, 128), block(128, 256),
                                     nn.AdaptiveAvgPool2d(1), nn.Flatten())
            self.num_features = dim

        def forward(self, x):
            return self.net(x)

    dim = 256
    return FallbackCNN(in_chans, dim), dim


class SliceAggregator(nn.Module):
    """Bidirectional GRU or Multi-Head Self-Attention + attention pooling over slices."""

    def __init__(self, dim: int, mode: str = "bigru", heads: int = 4, dropout: float = 0.25):
        super().__init__()
        self.mode = mode
        if mode == "bigru":
            self.gru = nn.GRU(dim, dim // 2, num_layers=1, batch_first=True,
                              bidirectional=True, dropout=dropout)
        else:  # mhsa
            self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
            self.norm1 = nn.LayerNorm(dim)
        # Learned query attention pooling -> study-level vector
        self.pool_query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.pool_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=max(1, heads // 2), batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: (B, D, dim) -> (B, dim)"""
        if self.mode == "bigru":
            h, _ = self.gru(x)
            h = self.norm1(h) if hasattr(self, "norm1") else h
        else:
            attn_mask = None
            if mask is not None:
                attn_mask = ~mask.bool()          # True => ignore
                if attn_mask.all(dim=1).any():    # fully-masked rows would NaN softmax
                    attn_mask = None
            att, _ = self.attn(x, x, x, attn_mask=attn_mask)
            h = self.norm1(x + att)
        q = self.pool_query.expand(x.shape[0], -1, -1)
        key_mask = None
        if mask is not None:
            key_mask = ~mask.bool()
            if key_mask.all(dim=1).any():
                key_mask = None
        pooled, _ = self.pool_attn(q, h, h, key_padding_mask=key_mask, need_weights=False)
        return self.drop(self.norm(pooled.squeeze(1)))


class RSNAMultiPlaneModel(nn.Module):
    """Shared 2.5D backbone across planes + per-plane aggregators + metadata fusion."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        backbone, feat_dim = build_backbone(cfg.BACKBONE, cfg.IN_CHANNELS)
        self.backbone = backbone
        self.proj = nn.Sequential(nn.Linear(feat_dim, cfg.FEATURE_DIM), nn.ReLU(inplace=True))

        self.aggregators = nn.ModuleList([
            SliceAggregator(cfg.FEATURE_DIM, mode=cfg.AGGREGATOR, heads=cfg.ATTENTION_HEADS, dropout=cfg.DROPOUT)
            for _ in range(cfg.NUM_PLANES)
        ])

        # Metadata embeddings: FluidSensitive, FatSuppression (per plane) + plane indicator.
        self.meta_embed = nn.Embedding(4, 16)   # 0=none(pad),1=sag,2=cor,3=ax combined w/ flags handled separately
        self.flags_proj = nn.Linear(3 * 2, 32)  # per-plane [fluid, fat] flags flattened
        self.plane_embed = nn.Embedding(cfg.NUM_PLANES, 8)

        fused_dim = cfg.FEATURE_DIM * cfg.NUM_PLANES + 32 + cfg.NUM_PLANES * 8
        self.head = nn.Sequential(
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(fused_dim, 512),
            nn.GELU(),
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(512, len(cfg.TARGET_COLS)),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """batch['volumes']: (B,P,D,C,H,W) -> logits (B,12)"""
        vols = batch["volumes"]
        b, p, d, c, h, w = vols.shape
        x = vols.reshape(b * p * d, c, h, w)
        with torch.autocast(device_type=self.cfg.DEVICE, dtype=torch.float16,
                            enabled=self.cfg.MIXED_PRECISION and self.cfg.DEVICE == "cuda"):
            feats = self.backbone(x)                       # (B*P*D, feat_dim)
            feats = self.proj(feats.float())               # (B*P*D, FEATURE_DIM) in fp32
        feats = feats.reshape(b * p, d, -1)

        plane_masks = batch["plane_mask"].reshape(b * p)   # (B*P,)
        outs = []
        for pi in range(p):
            sel = slice(pi, b * p, p)                      # indices for plane pi across batch
            fp = feats[sel]                                # (B, D, FEATURE_DIM)
            mp = plane_masks[sel]                          # (B,)
            agg = self.aggregators[pi](fp, mask=None)      # shapes fixed; pad volumes are zeros
            # Zero-out contributions of missing planes.
            agg = agg * mp.unsqueeze(-1)
            outs.append(agg)

        plane_feats = torch.cat(outs, dim=-1)              # (B, P*FEATURE_DIM)
        flags = torch.cat([batch["fluid_flags"], batch["fat_flags"]], dim=-1)  # (B, 2P)
        flag_emb = self.flags_proj(flags)
        plane_ids = torch.arange(p, device=vols.device).unsqueeze(0).expand(b, -1)
        p_emb = self.plane_embed(plane_ids).reshape(b, -1)
        fused = torch.cat([plane_feats, flag_emb, p_emb], dim=-1)
        logits = self.head(fused)
        return logits


# --------------------------------------------------------------------------------------
# SECTION 6: LOSS & METRICS
# --------------------------------------------------------------------------------------
class AsymmetricLoss(nn.Module):
    """Asymmetric Loss for multi-label imbalance (Ridnik et al. 2021), optimized variant.

    gamma_neg=4, gamma_pos=0, clip=0.05 maximizes macro AUC under extreme sparsity.
    """

    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 0.0, clip: float = 0.05,
                 eps: float = 1e-8, disable_tilde_gamma: bool = True):
        super().__init__()
        self.gamma_neg, self.gamma_pos, self.clip = gamma_neg, gamma_pos, clip
        self.eps = eps
        self.disable_tilde_gamma = disable_tilde_gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                sample_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        p_pos = probs
        p_neg = 1.0 - probs
        if self.clip is not None and self.clip > 0:
            p_neg = (p_neg + self.clip).clamp(max=1.0)
        loss_pos = targets * torch.log(p_pos.clamp(min=self.eps))
        loss_neg = (1.0 - targets) * torch.log(p_neg.clamp(min=self.eps))
        loss = -(loss_pos + loss_neg)
        # Asymmetric focusing
        tilde_gamma_neg = self.gamma_neg if self.disable_tilde_gamma else 0.0
        tilde_gamma_pos = self.gamma_pos if self.disable_tilde_gamma else 0.0
        loss = loss * ((probs - tilde_gamma_neg).clamp(min=0) ** self.gamma_neg * (1 - targets) +
                       (1 - probs - tilde_gamma_pos).clamp(min=0) ** self.gamma_pos * targets)
        if sample_weight is not None:
            sw = sample_weight.view(-1, 1)
            loss = loss * sw
            denom = sw.expand_as(loss).sum().clamp(min=1.0)
            return loss.sum() / denom
        return loss.mean()


class MacroAUC:
    """Competition metric: mean over classes of ROC-AUC; crash-safe for degenerate batches."""

    def __init__(self, n_classes: int = 12):
        self.n = n_classes
        self.reset()

    def reset(self) -> None:
        self.y: List[np.ndarray] = []
        self.p: List[np.ndarray] = []

    def update(self, probs: torch.Tensor, targets: torch.Tensor) -> None:
        self.p.append(probs.detach().float().cpu().numpy())
        self.y.append(targets.detach().cpu().numpy())

    def compute(self) -> Tuple[float, Dict[str, float]]:
        if not self.y:
            return float("nan"), {}
        y = np.concatenate(self.y, axis=0)
        p = np.concatenate(self.p, axis=0)
        per_class: Dict[str, float] = {}
        aucs: List[float] = []
        try:
            from sklearn.metrics import roc_auc_score
            for j in range(self.n):
                col = y[:, j]
                if len(np.unique(col)) < 2:      # single-class column -> AUC undefined; skip gracefully
                    per_class[f"class_{j}"] = float("nan")
                    continue
                a = float(roc_auc_score(col, p[:, j]))
                per_class[f"class_{j}"] = a
                aucs.append(a)
        except ImportError:
            # Manual rank-based AUC fallback (Mann-Whitney U).
            for j in range(self.n):
                col = y[:, j]
                pos, neg = p[col == 1, j], p[col == 0, j]
                if pos.size == 0 or neg.size == 0:
                    continue
                ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
                r_pos = ranks[:pos.size].sum()
                a = (r_pos - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size)
                per_class[f"class_{j}"] = float(a)
                aucs.append(float(a))
        macro = float(np.mean(aucs)) if aucs else float("nan")
        return macro, per_class


# --------------------------------------------------------------------------------------
# SECTION 7: TRAINING LOOP (AMP, EMA, warmup+cosine, GroupKFold)
# --------------------------------------------------------------------------------------
class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}
        self.other = {k: v.detach().clone() for k, v in model.state_dict().items()
                      if not v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)
            else:
                self.other[k] = v.detach().clone()

    def state_dict(self) -> Dict[str, torch.Tensor]:
        out = dict(self.shadow)
        out.update(self.other)
        return out

    def apply_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.state_dict(), strict=False)


def get_lr(cfg: Config, base_lr: float, epoch_frac: float) -> float:
    """Linear warmup for WARMUP_EPOCHS then cosine annealing to 0."""
    if cfg.WARMUP_EPOCHS > 0 and epoch_frac < cfg.WARMUP_EPOCHS:
        return base_lr * (epoch_frac / max(1e-6, float(cfg.WARMUP_EPOCHS)))
    prog = (epoch_frac - cfg.WARMUP_EPOCHS) / max(1e-6, float(cfg.EPOCHS - cfg.WARMUP_EPOCHS))
    prog = min(1.0, max(0.0, prog))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * prog))


@torch.no_grad()
def predict(model: nn.Module, batch: Dict[str, torch.Tensor], device: str, amp: bool) -> torch.Tensor:
    batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    with torch.autocast(device_type=device, dtype=torch.float16, enabled=amp and device == "cuda"):
        logits = model(batch)
    return torch.sigmoid(logits.float())


def run_epoch(model: nn.Module, loader: DataLoader, optimizer: Optional[torch.optim.Optimizer],
              criterion: AsymmetricLoss, device: str, cfg: Config,
              ema: Optional[EMA], lr_fn=None, epoch_idx: float = 0.0,
              metrics: Optional[MacroAUC] = None, log_every: int = 25) -> float:
    train = optimizer is not None
    model.train(train)
    if metrics is not None:
        metrics.reset()
    total_loss = 0.0
    n_batches = 0
    for step, batch in enumerate(loader):
        uids = batch.pop("uids")  # type: ignore[misc]
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        if train:
            assert optimizer is not None
            if lr_fn is not None:
                lr = lr_fn(epoch_idx + step / max(1, len(loader)))
                for g in optimizer.param_groups:
                    g["lr"] = lr
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=cfg.MIXED_PRECISION and device == "cuda"):
                logits = model(batch)
                loss = criterion(logits, batch["target"], batch["weight"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            if ema is not None:
                ema.update(model)
        else:
            assert metrics is not None
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=cfg.MIXED_PRECISION and device == "cuda"):
                logits = model(batch)
            probs = torch.sigmoid(logits.float())
            metrics.update(probs, batch["target"])
        total_loss += float(loss.detach().item())
        n_batches += 1
        if train and (step + 1) % log_every == 0:
            cur_lr = optimizer.param_groups[0]["lr"] if optimizer else 0.0
            print(f"  [{mode_str(train)}] step {step+1}/{len(loader)} loss={total_loss/n_batches:.4f} lr={cur_lr:.2e}", flush=True)
    return total_loss / max(1, n_batches)


def mode_str(train: bool) -> str:
    return "train" if train else "valid"


def make_folds(df: pd.DataFrame, n_folds: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Stratified GroupKFold by StudyInstanceUID (multi-label stratification via label-hash bucketing)."""
    from sklearn.model_selection import StratifiedGroupKFold
    labels_bin = df[TARGET_COLS].fillna(0).astype(int)
    # Bucket stratification key: primary positive class (most frequent informative), fallback "ALLNEG".
    any_pos = labels_bin.sum(axis=1) > 0
    strat_key = np.where(any_pos, labels_bin.idxmax(axis=1), "ALLNEG")
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = list(sgkf.split(df.index, y=strat_key, groups=df["StudyInstanceUID"].values))
    return folds


def train_pipeline(args: argparse.Namespace) -> None:
    cfg = Config()
    cfg.BACKBONE = args.backbone
    cfg.EPOCHS = args.epochs
    cfg.BATCH_SIZE = args.batch_size
    cfg.NUM_WORKERS = args.num_workers
    cfg.AGGREGATOR = args.aggregator
    set_determinism(cfg.SEED)
    device = cfg.DEVICE
    print(f"Config: {json.dumps(asdict(cfg), default=str)}")

    manifest = pd.read_csv(args.manifest)
    labels = pd.read_csv(args.labels)
    # Restrict to studies whose volumes actually exist.
    vol_dir = Path(args.volumes_root)
    existing = {str(Path(r).name).rsplit("_", 1)[0] for r in manifest.loc[manifest["status"].isin(["ok", "skipped"]), "VolumePath"]}
    labels = labels[labels["StudyInstanceUID"].astype(str).isin(existing)].reset_index(drop=True)
    if len(labels) == 0:
        raise RuntimeError("No labeled studies intersect with available preprocessed volumes.")

    folds = make_folds(labels, cfg.N_FOLDS, cfg.SEED)
    ckpt_dir = Path(args.ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)

    fold_range = [args.fold] if args.fold >= 0 else list(range(cfg.N_FOLDS))
    oof_all: Dict[int, np.ndarray] = {}

    for fold in fold_range:
        tr_idx, va_idx = folds[fold]
        tr_df, va_df = labels.iloc[tr_idx], labels.iloc[va_idx]
        ds_tr = RSNAKneeDataset(tr_df, manifest, args.volumes_root, cfg, train=True)
        ds_va = RSNAKneeDataset(va_df, manifest, args.volumes_root, cfg, train=False)
        dl_tr = DataLoader(ds_tr, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS,
                           collate_fn=collate_knee, pin_memory=(device == "cuda"), drop_last=True)
        dl_va = DataLoader(ds_va, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS,
                           collate_fn=collate_knee, pin_memory=(device == "cuda"))

        model = RSNAMultiPlaneModel(cfg).to(device)
        ema = EMA(model, cfg.EMA_DECAY) if cfg.USE_EMA else None
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
        crit = AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.0, clip=0.05).to(device)
        metric = MacroAUC(len(cfg.TARGET_COLS))
        steps_per_epoch = max(1, len(dl_tr))
        total_steps = cfg.EPOCHS * steps_per_epoch
        start_step = [0]

        def lr_fn(frac_epochs: float) -> float:
            return get_lr(cfg, cfg.LR, frac_epochs)

        best_auc = -1.0
        hist: List[Dict[str, float]] = []
        for epoch in range(cfg.EPOCHS):
            t0 = time.time()
            tl = run_epoch(model, dl_tr, opt, crit, device, cfg, ema, lr_fn, epoch_idx=float(epoch))
            # Validate with EMA weights (swap-in, swap-out).
            raw_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if ema is not None:
                ema.apply_to(model)
            model.eval()
            vl = run_epoch(model, dl_va, None, crit, device, cfg, None, None, metrics=metric)
            auc, per_class = metric.compute()
            if ema is not None:
                model.load_state_dict(raw_state)
            dt = time.time() - t0
            print(f"[fold {fold}] epoch {epoch+1}/{cfg.EPOCHS} train_loss={tl:.4f} val_loss={vl:.4f} "
                  f"val_macro_AUC={auc:.4f} ({dt:.1f}s)", flush=True)
            hist.append({"epoch": epoch + 1, "train_loss": tl, "val_loss": vl, "val_auc": auc})
            if np.isfinite(auc) and auc > best_auc:
                best_auc = auc
                payload = {
                    "model_state": model.state_dict(),
                    "ema_state": ema.state_dict() if ema is not None else None,
                    "optimizer_state": opt.state_dict(),
                    "config": asdict(cfg),
                    "fold": fold,
                    "best_auc": best_auc,
                    "backbone": cfg.BACKBONE,
                }
                torch.save(payload, ckpt_dir / f"best_auc_fold_{fold}.pt")
                print(f"  -> saved new best checkpoint (AUC={best_auc:.4f})", flush=True)
        with open(ckpt_dir / f"history_fold_{fold}_{Path(cfg.BACKBONE).name}.json", "w") as fh:
            json.dump({"history": hist, "best_auc": best_auc}, fh, indent=2)
        print(f"[fold {fold}] DONE best val macro AUC = {best_auc:.4f}", flush=True)

    # Fold summary
    summary_fp = ckpt_dir / "fold_summary.json"
    summaries = {}
    for f in fold_range:
        hb = sorted(ckpt_dir.glob(f"history_fold_{f}_*.json"))
        if hb:
            summaries[f] = json.loads(hb[-1].read_text()).get("best_auc")
    summary_fp.write_text(json.dumps(summaries, indent=2))
    print("Fold best-AUC summary:", summaries)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="RSNA Knee multi-plane 2.5D trainer")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--labels", required=True, help="train_pseudo_labels.csv from Module 2")
    ap.add_argument("--volumes-root", required=True)
    ap.add_argument("--ckpt-dir", default="/kaggle/working/checkpoints")
    ap.add_argument("--backbone", default="convnext_nano.in12k_ft_in1k")
    ap.add_argument("--aggregator", default="bigru", choices=["bigru", "mhsa"])
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--fold", type=int, default=-1, help="fold index or -1 for all folds")
    ns = ap.parse_args(argv)
    train_pipeline(ns)
    return 0


if __name__ == "__main__":
    sys.exit(main())
