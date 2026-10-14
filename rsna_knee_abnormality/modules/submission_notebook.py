"""
MODULE 4 — Kaggle-Compliant Offline Inference & Ensemble Submission (RSNA Knee).

Design goals
------------
1. STRICTLY OFFLINE: no pip installs, no HTTP calls; all weights come from attached datasets.
2. FAST DYNAMIC DICOM READER for test studies (no preprocessed .npy available at submit time):
   slice sorting -> percentile clip -> linspace resample to K=16 -> resize 256x256 -> tensor.
3. DEFENSIVE: any unreadable/corrupt study falls back to training-set prior probabilities.
4. TTA: original + horizontal-flip averaged.
5. ENSEMBLE: 5 fold checkpoints x {convnext_nano, tf_efficientnet_b0_ns} rank-averaged predictions.
6. VALIDATION: exact row count vs test.csv, exact header order, values in [0,1], non-null.

Paste this into the Kaggle "submission" notebook cells (Internet OFF, GPU ON, <=9h budget).
It is also importable as a module (`python submission_notebook.py --dry-run` for smoke tests).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

# --------------------------------------------------------------------------------------
# Shared constants / config (must match Module 3 exactly)
# --------------------------------------------------------------------------------------
TARGET_COLS: List[str] = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
    "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]
SUBMISSION_HEADER: List[str] = ["StudyInstanceUID"] + TARGET_COLS


@dataclass
class InferConfig:
    SEED: int = 42
    IMAGE_SIZE: Tuple[int, int] = (256, 256)
    NUM_SLICES: int = 16
    BATCH_SIZE: int = 8                       # studies per batch during inference
    IN_CHANNELS: int = 3
    NUM_PLANES: int = 3
    FEATURE_DIM: int = 256
    AGGREGATOR: str = "bigru"
    ATTENTION_HEADS: int = 4
    DROPOUT: float = 0.25
    MIXED_PRECISION: bool = True
    BACKBONES: List[str] = field(default_factory=lambda: [
        "convnext_nano.in12k_ft_in1k", "tf_efficientnet_b0.ns_jft_in1k",
    ])
    N_FOLDS: int = 5
    TIME_BUDGET_HOURS: float = 7.5            # stop early with partial ensemble if exceeded
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


np.random.seed(InferConfig.SEED)
torch.manual_seed(InferConfig.SEED)

# --------------------------------------------------------------------------------------
# Import model/dataset classes from train.py when available; otherwise define inline.
# The submission notebook must be self-contained if train.py isn't attached, so we
# re-export via try/except with full inline fallback definitions.
# --------------------------------------------------------------------------------------
_INLINE_DEFINITIONS = False
try:
    from train import (  # type: ignore
        Config as TrainConfig, RSNAMultiPlaneModel, RSNAKneeDataset, collate_knee, build_backbone,
    )
except Exception:
    _INLINE_DEFINITIONS = True

if _INLINE_DEFINITIONS:
    @dataclass
    class TrainConfig:
        SEED: int = 42
        IMAGE_SIZE: Tuple[int, int] = (256, 256)
        NUM_SLICES: int = 16
        BATCH_SIZE: int = 8
        NUM_WORKERS: int = 4
        BACKBONE: str = "convnext_nano.in12k_ft_in1k"
        IN_CHANNELS: int = 3
        NUM_PLANES: int = 3
        FEATURE_DIM: int = 256
        AGGREGATOR: str = "bigru"
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
        PSEUDO_WEIGHT: float = 0.5
        TARGET_COLS: List[str] = field(default_factory=lambda: list(TARGET_COLS))
        DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    def build_backbone(name: str, in_chans: int) -> Tuple[nn.Module, int]:
        """Offline-safe timm backbone builder with local CNN fallback."""
        try:
            import timm
            errors: List[str] = []
            for cand in (name, name.split(".")[0]):
                try:
                    m = timm.create_model(cand, pretrained=False, in_chans=in_chans, num_classes=0)
                    dim = getattr(m, "num_features", None) or 512
                    return m, int(dim)
                except Exception as exc:
                    errors.append(str(exc))
            raise RuntimeError("; ".join(errors[:2]))
        except Exception:
            class FallbackCNN(nn.Module):
                def __init__(self, c: int, dim: int = 256):
                    super().__init__()
                    def block(ci, co):
                        return nn.Sequential(nn.Conv2d(ci, co, 3, stride=2, padding=1, bias=False),
                                             nn.BatchNorm2d(co), nn.SiLU(inplace=True))
                    self.net = nn.Sequential(block(c, 32), block(32, 64), block(64, 128),
                                             block(128, 256), nn.AdaptiveAvgPool2d(1), nn.Flatten())
                    self.num_features = dim
                def forward(self, x):
                    return self.net(x)
            return FallbackCNN(in_chans, 256), 256

    class SliceAggregator(nn.Module):
        def __init__(self, dim: int, mode: str = "bigru", heads: int = 4, dropout: float = 0.25):
            super().__init__()
            self.mode = mode
            if mode == "bigru":
                self.gru = nn.GRU(dim, dim // 2, num_layers=1, batch_first=True, bidirectional=True, dropout=dropout)
            else:
                self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
                self.norm1 = nn.LayerNorm(dim)
            self.pool_query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
            self.pool_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=max(1, heads // 2),
                                                   batch_first=True, dropout=dropout)
            self.norm = nn.LayerNorm(dim)
            self.drop = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            if self.mode == "bigru":
                h, _ = self.gru(x)
            else:
                att, _ = self.attn(x, x, x)
                h = self.norm1(x + att)
            q = self.pool_query.expand(x.shape[0], -1, -1)
            pooled, _ = self.pool_attn(q, h, h, need_weights=False)
            return self.drop(self.norm(pooled.squeeze(1)))

    class RSNAMultiPlaneModel(nn.Module):
        def __init__(self, cfg: TrainConfig):
            super().__init__()
            self.cfg = cfg
            backbone, feat_dim = build_backbone(cfg.BACKBONE, cfg.IN_CHANNELS)
            self.backbone = backbone
            self.proj = nn.Sequential(nn.Linear(feat_dim, cfg.FEATURE_DIM), nn.ReLU(inplace=True))
            self.aggregators = nn.ModuleList([
                SliceAggregator(cfg.FEATURE_DIM, mode=cfg.AGGREGATOR, heads=cfg.ATTENTION_HEADS, dropout=cfg.DROPOUT)
                for _ in range(cfg.NUM_PLANES)])
            self.meta_embed = nn.Embedding(4, 16)
            self.flags_proj = nn.Linear(3 * 2, 32)
            self.plane_embed = nn.Embedding(cfg.NUM_PLANES, 8)
            fused_dim = cfg.FEATURE_DIM * cfg.NUM_PLANES + 32 + cfg.NUM_PLANES * 8
            self.head = nn.Sequential(
                nn.Dropout(cfg.DROPOUT), nn.Linear(fused_dim, 512), nn.GELU(),
                nn.Dropout(cfg.DROPOUT), nn.Linear(512, len(cfg.TARGET_COLS)))

        def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
            vols = batch["volumes"]
            b, p, d, c, h, w = vols.shape
            x = vols.reshape(b * p * d, c, h, w)
            feats = self.backbone(x)
            feats = self.proj(feats.float())
            feats = feats.reshape(b * p, d, -1)
            plane_masks = batch["plane_mask"].reshape(b * p)
            outs = []
            for pi in range(p):
                sel = slice(pi, b * p, p)
                fp, mp = feats[sel], plane_masks[sel]
                agg = self.aggregators[pi](fp, mask=None)
                outs.append(agg * mp.unsqueeze(-1))
            plane_feats = torch.cat(outs, dim=-1)
            flags = torch.cat([batch["fluid_flags"], batch["fat_flags"]], dim=-1)
            flag_emb = self.flags_proj(flags)
            plane_ids = torch.arange(p, device=vols.device).unsqueeze(0).expand(b, -1)
            p_emb = self.plane_embed(plane_ids).reshape(b, -1)
            fused = torch.cat([plane_feats, flag_emb, p_emb], dim=-1)
            return self.head(fused)

    def collate_knee(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        uids = [b.pop("uid") for b in batch]
        out = {k: torch.stack([b[k] for b in batch], dim=0) for k in
               ["volumes", "plane_mask", "fluid_flags", "fat_flags", "meta", "target", "weight"]}
        out["uids"] = uids  # type: ignore[assignment]
        return out


# --------------------------------------------------------------------------------------
# Fast dynamic DICOM reader (test-time, on-the-fly)
# --------------------------------------------------------------------------------------
def _safe_get(ds, name, default=None):
    try:
        v = getattr(ds, name, default)
    except Exception:
        return default
    return default if v is None else v


def read_series_onthefly(series_dir: Path, num_slices: int, image_size: Tuple[int, int],
                         p_low: float = 0.5, p_high: float = 99.5) -> Tuple[np.ndarray, Dict[str, float]]:
    """Decode one series dir -> (uint8 volume (K,H,W), metadata dict {fluid, fat}).

    Raises RuntimeError on total decode failure (caller imputes priors).
    """
    import pydicom
    files = [Path(e.path) for e in os.scandir(str(series_dir)) if e.is_file()]
    if not files:
        raise RuntimeError(f"empty series {series_dir}")
    dss = []
    n_err = 0
    for fp in files:
        try:
            ds = pydicom.dcmread(str(fp), force=True)
            if "PixelData" in ds or hasattr(ds, "pixel_data_array"):
                dss.append(ds)
            else:
                n_err += 1
        except Exception:
            n_err += 1
    if not dss:
        raise RuntimeError(f"no readable slices in {series_dir} ({n_err} errors)")

    def zkey(ds):
        ipp = _safe_get(ds, "ImagePositionPatient", None)
        try:
            return float(ipp[2])
        except Exception:
            pass
        try:
            return float(_safe_get(ds, "SliceLocation", float(_safe_get(ds, "InstanceNumber", 0))))
        except Exception:
            return 0.0
    try:
        dss = sorted(dss, key=zkey)
    except Exception:
        pass

    planes = str(_safe_get(dss[0], "AnatomicalPlaneType", "") or "").upper()
    desc_blob = (str(_safe_get(dss[0], "SeriesDescription", "") or "") + " " +
                 str(_safe_get(dss[0], "SequenceName", "") or "")).upper()
    if not planes or planes == "OTHER":
        iop = _safe_get(dss[0], "ImageOrientationPatient", None)
        try:
            row = np.array([float(x) for x in iop[:3]]); col = np.array([float(x) for x in iop[3:]])
            n = np.cross(row, col); ax = int(np.argmax(np.abs(n)))
            # Normal along x <=> slices stacked left-right <=> SAGITTAL view.
            planes = {0: "SAGITTAL", 1: "CORONAL", 2: "AXIAL"}.get(ax, "UNKNOWN")
            # normal direction convention varies; fall back to description keywords
            if "SAG" in desc_blob: planes = "SAGITTAL"
            elif "COR" in desc_blob: planes = "CORONAL"
            elif "AXI" in desc_blob or "TRA" in desc_blob: planes = "AXIAL"
        except Exception:
            planes = "UNKNOWN"
    fluid = float(any(t in desc_blob for t in ("PDFS", "T2 FS", "T2FS", "STIR", "FLAIR", "FS", "FATSAT", "SPIR")))
    fat = float(any(t in desc_blob for t in ("FS", "FATSAT", "SPIR", "STIR")))

    arrs: List[np.ndarray] = []
    decode_fail = 0
    for ds in dss:
        try:
            a = ds.pixel_array
            if a.ndim == 3:
                a = a.mean(axis=0) if a.shape[0] <= 16 else a.reshape(a.shape[0], -1).mean(axis=1).reshape(a.shape[1:])
            a = a.astype(np.float32)
            slope = _safe_get(ds, "RescaleSlope", 1.0); inter = _safe_get(ds, "RescaleIntercept", 0.0)
            try:
                a = a * float(slope) + float(inter)
            except Exception:
                pass
            arrs.append(a)
        except Exception:
            decode_fail += 1
    if decode_fail > max(1, len(dss) // 2) or not arrs:
        raise RuntimeError(f"{decode_fail}/{len(dss)} undecodable slices in {series_dir} (missing pylibjpeg decoders?)")
    while len(arrs) < len(dss):
        arrs.append(np.zeros_like(arrs[0]))

    vol = np.stack(arrs, axis=0).astype(np.float32)
    ref = arrs[0]
    finite = vol[np.isfinite(vol)]
    lo = float(np.percentile(finite, p_low)) if finite.size else 0.0
    hi = float(np.percentile(finite, p_high)) if finite.size else 1.0
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = -1.0, 1.0
    vol = np.nan_to_num((vol - lo) / (hi - lo), nan=0.0, posinf=1.0, neginf=0.0).clip(0, 1)
    vol = (vol * 255.0).round().astype(np.uint8)

    # depth resample to K
    d = vol.shape[0]
    if d != num_slices:
        idx = np.linspace(0, d - 1, num_slices).round().astype(int) if d > 1 else np.zeros(num_slices, dtype=int)
        vol = vol[idx]
    # spatial resize
    if vol.shape[1:] != tuple(image_size):
        try:
            import cv2
            vol = np.stack([cv2.resize(v, (image_size[1], image_size[0]), interpolation=cv2.INTER_AREA) for v in vol])
        except Exception:
            from PIL import Image
            vol = np.stack([np.asarray(Image.fromarray(v).resize((image_size[1], image_size[0]))) for v in vol])
    meta = {"plane": planes, "fluid": fluid, "fat": fat}
    assert vol.shape[0] == num_slices and vol.shape[1] == image_size[0] and vol.shape[2] == image_size[1]
    return vol, meta


def load_study_volumes(study_series_dir: Path, cfg: InferConfig) -> Dict[str, object]:
    """Assemble the 3-plane tensor bundle for one study with defensive zero-padding."""
    slots: Dict[str, Tuple[Optional[np.ndarray], Dict[str, float]]] = {
        "SAGITTAL": (None, {}), "CORONAL": (None, {}), "AXIAL": (None, {})}
    order = ["SAGITTAL", "CORONAL", "AXIAL"]
    extras: List[Tuple[np.ndarray, Dict[str, float]]] = []
    entries = [e for e in os.scandir(str(study_series_dir)) if e.is_dir()]
    for e in entries:
        try:
            vol, meta = read_series_onthefly(Path(e.path), cfg.NUM_SLICES, cfg.IMAGE_SIZE)
        except Exception:
            continue
        pl = str(meta.get("plane", "UNKNOWN")).upper()
        placed = False
        for key in order:
            if key in pl and slots[key][0] is None:
                slots[key] = (vol, meta); placed = True; break
        if not placed:
            extras.append((vol, meta))
    # Fill empty slots with leftover series (keeps model fed even if plane tags are bad).
    for key in order:
        if slots[key][0] is None and extras:
            slots[key] = extras.pop(0)

    vols, masks, fluids, fats = [], [], [], []
    for key in order:
        v, m = slots[key]
        if v is None:
            v = np.zeros((cfg.NUM_SLICES, *cfg.IMAGE_SIZE), dtype=np.uint8)
            m = {"fluid": 0.0, "fat": 0.0}
            masks.append(0.0)
        else:
            masks.append(1.0)
        vols.append(v)
        fluids.append(float(m.get("fluid", 0.0)))
        fats.append(float(m.get("fat", 0.0)))
    stack = np.stack(vols, axis=0).astype(np.float32) / 255.0
    stack = np.repeat(stack[:, :, None], cfg.IN_CHANNELS, axis=2)     # (P,D,C,H,W)
    any_readable = max(masks) > 0.0
    return {"volumes": stack, "plane_mask": np.array(masks, np.float32),
            "fluid_flags": np.array(fluids, np.float32), "fat_flags": np.array(fats, np.float32),
            "meta": np.array([max(fluids), max(fats), float(sum(masks)) / 3.0, masks[2]], np.float32),
            "readable": bool(any_readable)}


# --------------------------------------------------------------------------------------
# Checkpoint loading & ensemble inference
# --------------------------------------------------------------------------------------
def find_checkpoints(weights_root: str | Path, backbones: List[str], n_folds: int) -> List[Path]:
    root = Path(weights_root)
    found: List[Path] = []
    for bb in backbones:
        stem = bb.split(".")[0].replace("-", "_")
        for f in range(n_folds):
            cands = list(root.rglob(f"best_auc_fold_{f}.pt")) + \
                    list(root.rglob(f"*{stem}*fold_{f}*.pt")) + \
                    list(root.rglob(f"*fold_{f}*{stem}*.pt"))
            uniq = {c.resolve(): c for c in cands}
            for c in uniq.values():
                if c not in found:
                    found.append(c)
                    break
    return found


def make_train_cfg_from_ckpt(payload: Dict[str, object], fallback_backbone: str) -> TrainConfig:
    cfg_d = payload.get("config", {}) if isinstance(payload, dict) else {}
    tc = TrainConfig()
    for k, v in (cfg_d or {}).items():
        if hasattr(tc, k) and k not in ("DEVICE", "TARGET_COLS"):
            setattr(tc, k, v)
    tc.BACKBONE = str(payload.get("backbone", fallback_backbone)) if isinstance(payload, dict) else fallback_backbone
    return tc


@torch.inference_mode()
def predict_one_model(model: nn.Module, loader_iter, device: str, amp: bool,
                      flip: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference over pre-assembled study tensors; optionally horizontally flip volumes."""
    model.eval()
    probs_all: List[np.ndarray] = []
    uids_all: List[str] = []
    for uids, bundle in loader_iter:
        vols = bundle["volumes"]
        if flip:
            vols = vols[..., ::-1]
        x = {
            "volumes": torch.from_numpy(vols).to(device, non_blocking=True),
            "plane_mask": torch.from_numpy(bundle["plane_mask"]).to(device),
            "fluid_flags": torch.from_numpy(bundle["fluid_flags"]).to(device),
            "fat_flags": torch.from_numpy(bundle["fat_flags"]).to(device),
            "meta": torch.from_numpy(bundle["meta"]).to(device),
            "target": torch.zeros((len(uids), len(TARGET_COLS))),
            "weight": torch.ones(len(uids)),
        }
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=amp and device == "cuda"):
            logits = model(x)
        probs_all.append(torch.sigmoid(logits.float()).cpu().numpy())
        uids_all.extend(uids)
    return np.concatenate(probs_all, axis=0) if probs_all else np.zeros((0, len(TARGET_COLS))), np.array(uids_all)


def build_study_batches(studies: List[str], series_root: Path, cfg: InferConfig,
                        chunk: int = 48) -> List[Tuple[List[str], Dict[str, np.ndarray]]]:
    """Group raw DICOM studies into inference batches; failed reads flagged for imputation."""
    batches: List[Tuple[List[str], Dict[str, np.ndarray]]] = []
    buf_u: List[str] = []
    buf_v: List[np.ndarray] = []
    buf_pm: List[np.ndarray] = []
    buf_fl: List[np.ndarray] = []
    buf_fa: List[np.ndarray] = []
    buf_me: List[np.ndarray] = []
    for uid in studies:
        sdir = series_root / uid
        ok = False
        try:
            if sdir.is_dir():
                bundle = load_study_volumes(sdir, cfg)
                ok = bundle["readable"]
                if ok:
                    buf_v.append(bundle["volumes"]); buf_pm.append(bundle["plane_mask"])
                    buf_fl.append(bundle["fluid_flags"]); buf_fa.append(bundle["fat_flags"])
                    buf_me.append(bundle["meta"]); buf_u.append(uid)
        except Exception:
            ok = False
        if not ok:
            buf_u.append(uid)
            buf_v.append(np.zeros((cfg.NUM_PLANES, cfg.NUM_SLICES, cfg.IN_CHANNELS, *cfg.IMAGE_SIZE), np.float32))
            buf_pm.append(np.zeros(cfg.NUM_PLANES, np.float32))
            buf_fl.append(np.zeros(cfg.NUM_PLANES, np.float32))
            buf_fa.append(np.zeros(cfg.NUM_PLANES, np.float32))
            buf_me.append(np.zeros(4, np.float32))
        if len(buf_u) >= chunk:
            batches.append((buf_u, {"volumes": np.stack(buf_v), "plane_mask": np.stack(buf_pm),
                                    "fluid_flags": np.stack(buf_fl), "fat_flags": np.stack(buf_fa),
                                    "meta": np.stack(buf_me)}))
            buf_u, buf_v, buf_pm, buf_fl, buf_fa, buf_me = [], [], [], [], [], []
    if buf_u:
        batches.append((buf_u, {"volumes": np.stack(buf_v), "plane_mask": np.stack(buf_pm),
                                "fluid_flags": np.stack(buf_fl), "fat_flags": np.stack(buf_fa),
                                "meta": np.stack(buf_me)}))
    return batches


def rank_average(pred_stack: np.ndarray) -> np.ndarray:
    """Per-model, per-column rank normalization followed by averaging.

    Rank-averaging removes calibration drift between fold/backbone models while the
    probability-mean (applied separately and blended by the caller) preserves magnitude.
    pred_stack: (M, N, C) -> (N, C) in [0, 1].
    """
    m, n, c = pred_stack.shape
    if m == 1:
        return pred_stack[0]
    out = np.empty_like(pred_stack, dtype=np.float64)
    for mi in range(m):
        for cj in range(c):
            col = pred_stack[mi, :, cj]
            order = np.argsort(col, kind="stable")
            ranks = np.empty(n, dtype=np.float64)
            ranks[order] = np.arange(n, dtype=np.float64)
            out[mi, :, cj] = ranks / max(1.0, float(n - 1))
    return out.mean(axis=0)


def compute_training_priors(labels_csv: Optional[str], manifest_status_csv: Optional[str]) -> np.ndarray:
    """Fallback prior probabilities per target used to impute unreadable test studies."""
    if labels_csv and Path(labels_csv).exists():
        df = pd.read_csv(labels_csv)
        cols = [c for c in TARGET_COLS if c in df.columns]
        if cols:
            priors = df[cols].fillna(0).mean(axis=0).values.astype(np.float64)
            full = np.zeros(len(TARGET_COLS), dtype=np.float64)
            for i, c in enumerate(cols):
                full[TARGET_COLS.index(c)] = priors[i]
            return np.clip(full, 0.01, 0.99)
    return np.full(len(TARGET_COLS), 0.10, dtype=np.float64)  # generic knee-abnormality base rate


def run_submission(
    test_csv: str,
    test_series_root: str,
    weights_root: str,
    out_csv: str,
    labels_for_priors: Optional[str] = None,
    dry_run: bool = False,
    max_studies: Optional[int] = None,
) -> pd.DataFrame:
    cfg = InferConfig()
    t_start = time.time()
    deadline = t_start + cfg.TIME_BUDGET_HOURS * 3600.0

    test_df = pd.read_csv(test_csv)
    if "StudyInstanceUID" not in test_df.columns:
        raise ValueError("test.csv missing StudyInstanceUID column")
    study_ids = test_df["StudyInstanceUID"].astype(str).tolist()
    submission_mode = len(test_df) > 3 or Path(test_series_root).exists()
    print(f"[INFO] rows={len(test_df)} | submission_mode={submission_mode} | device={cfg.DEVICE}")
    if max_studies:
        study_ids = study_ids[:max_studies]

    priors = compute_training_priors(labels_for_priors, None)
    series_root = Path(test_series_root)

    # ---- assemble inference batches (CPU-bound DICOM decode happens once) ----------
    print("[INFO] Decoding test DICOM series (on-the-fly)...", flush=True)
    if dry_run:
        batches = []
    else:
        batches = build_study_batches(study_ids, series_root, cfg, chunk=48)
    print(f"[INFO] Built {len(batches)} study batches.", flush=True)

    ckpts = find_checkpoints(weights_root, cfg.BACKBONES, cfg.N_FOLDS)
    if not ckpts:
        print("[WARN] No checkpoints found under", weights_root, "— submitting calibrated priors.")
        preds = np.tile(priors, (len(study_ids), 1))
        uids = np.array(study_ids)
    else:
        print(f"[INFO] Found {len(ckpts)} checkpoints: {[c.name for c in ckpts[:6]]}...")
        model_preds: List[np.ndarray] = []
        uid_order_ref: Optional[np.ndarray] = None
        for ci, ck in enumerate(ckpts):
            if time.time() > deadline:
                print("[WARN] Time budget reached; finalizing with partial ensemble.")
                break
            try:
                payload = torch.load(ck, map_location="cpu", weights_only=False)
            except TypeError:
                payload = torch.load(ck, map_location="cpu")
            tc = make_train_cfg_from_ckpt(payload, cfg.BACKBONES[ci % len(cfg.BACKBONES)])
            model = RSNAMultiPlaneModel(tc)
            state = payload.get("ema_state") or payload.get("model_state")
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                print(f"[WARN] {ck.name}: missing keys {list(missing)[:4]} ...")
            model.to(cfg.DEVICE).eval()

            # Preload batches to tensors lazily per model to bound RAM.
            iters_orig = [(u, {k: v for k, v in b.items()}) for (u, b) in batches]
            probs_o, uids_o = predict_one_model(model, iters_orig, cfg.DEVICE, cfg.MIXED_PRECISION, flip=False)
            probs_f, _ = predict_one_model(model, iters_orig, cfg.DEVICE, cfg.MIXED_PRECISION, flip=True)
            probs_tta = (probs_o + probs_f) / 2.0
            del model
            if cfg.DEVICE == "cuda":
                torch.cuda.empty_cache()

            # Map back to the canonical study order; impute training priors for any missing rows.
            ordered = np.full((len(study_ids), len(TARGET_COLS)), np.nan)
            pos = {u: i for i, u in enumerate(study_ids)}
            for i, u in enumerate(uids_o):
                if u in pos:
                    ordered[pos[u]] = probs_tta[i]
            n_missing = int(np.isnan(ordered).any(axis=1).sum())
            if n_missing:
                ordered[np.isnan(ordered).any(axis=1)] = priors
                print(f"[INFO] Imputed training priors for {n_missing} unreadable studies.")
            model_preds.append(ordered)
            print(f"[INFO] Model {ci+1}/{len(ckpts)} ({Path(ck).parent.name}/{ck.name}) done "
                  f"in {(time.time()-t_start)/60:.1f} min elapsed", flush=True)

        if model_preds:
            stacked = np.stack(model_preds, axis=0)
            preds_rank = rank_average(stacked)
            preds_mean = stacked.mean(axis=0)
            # Blend rank-average with probability-mean (rank fixes calibration drift across folds).
            preds = np.clip(0.5 * preds_rank + 0.5 * preds_mean, 0.0, 1.0)
        else:
            preds = np.tile(priors, (len(study_ids), 1))

    sub = pd.DataFrame(preds, columns=TARGET_COLS)
    sub.insert(0, "StudyInstanceUID", study_ids)

    # ----------------------- SECTION 8 sanity checks --------------------------------
    expected_header = SUBMISSION_HEADER
    assert list(sub.columns) == expected_header, f"Header mismatch: {list(sub.columns)}"
    assert len(sub) == len(test_df), f"Row count mismatch: {len(sub)} vs {len(test_df)}"
    assert sub.notna().all().all(), "NaNs detected in submission"
    assert ((sub[TARGET_COLS] >= 0.0) & (sub[TARGET_COLS] <= 1.0)).all().all(), "Probabilities outside [0,1]"

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False)
    mins = (time.time() - t_start) / 60.0
    print(f"[OK] Wrote {out_path} | shape={sub.shape} | mean prob={sub[TARGET_COLS].values.mean():.4f} | elapsed={mins:.1f} min")
    return sub


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="RSNA Knee offline ensemble submission pipeline")
    ap.add_argument("--test-csv", default="/kaggle/input/rsna-str-pkc-dataset/test.csv")
    ap.add_argument("--test-series-root", default="/kaggle/input/rsna-str-pkc-dataset/test_series")
    ap.add_argument("--weights-root", default="/kaggle/input/rsna-knee-weights")
    ap.add_argument("--labels-for-priors", default=None)
    ap.add_argument("--out-csv", default="submission.csv")
    ap.add_argument("--dry-run", action="store_true", help="Skip DICOM/model work; emit priors-only submission")
    ap.add_argument("--max-studies", type=int, default=None)
    ns = ap.parse_args(argv)
    run_submission(ns.test_csv, ns.test_series_root, ns.weights_root, ns.out_csv,
                   labels_for_priors=ns.labels_for_priors, dry_run=ns.dry_run, max_studies=ns.max_studies)
    return 0


if __name__ == "__main__":
    sys.exit(main())
