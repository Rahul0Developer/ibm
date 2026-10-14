# ============================================================
# SECTION 1 - SYSTEM CONFIG & HYPERPARAMETERS
# ============================================================
# Single source of truth for every magic number in this notebook.
# Everything downstream (data, model, training, inference) reads from CFG.
from __future__ import annotations

import os
import random
import sys
import time
import json
import math
import hashlib
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DistributedDataParallel  # noqa: F401 (kept for reference)

warnings.filterwarnings("ignore")

try:  # tqdm is present on Kaggle; fall back to a no-op shim locally.
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover
    def _tqdm(iterable=None, *a, **kw):  # type: ignore[misc]
        return iterable if iterable is not None else []


@dataclass
class Config:
    """Unified experiment configuration."""

    # ------------------------------------------------------------------ seeds
    SEED: int = 42
    DETERMINISTIC: bool = True          # strict determinism (reproducible OOF)

    # -------------------------------------------------------------- tensor geo
    IMAGE_SIZE: Tuple[int, int] = (256, 256)   # (H, W) after resample
    NUM_SLICES: int = 16                        # K = uniform depth samples / series
    IN_CHANNELS: int = 3                        # single-channel MRI tiled x3 for ImageNet weights
    NUM_PLANES: int = 3                         # Sagittal, Coronal, Axial
    NUM_TARGETS: int = 12

    # ------------------------------------------------------------ optimisation
    BATCH_SIZE: int = 8                         # per-GPU train batch (3 planes x 16 slices)
    EVAL_BATCH_SIZE: int = 8
    NUM_WORKERS: int = 4
    PIN_MEMORY: bool = True
    PREFETCH_FACTOR: int = 2
    PERSISTENT_WORKERS: bool = True

    EPOCHS: int = 8                             # labelled epoch budget
    PSEUDO_EPOCHS: int = 3                      # extra epochs on pseudo-labelled studies
    LR: float = 3e-4
    MIN_LR: float = 1e-6
    WEIGHT_DECAY: float = 1e-2
    BETAS: Tuple[float, float] = (0.9, 0.999)
    WARMUP_STEPS: int = 400                     # linear LR warmup
    GRAD_CLIP_NORM: float = 5.0
    ACCUM_STEPS: int = 2                        # effective batch = BATCH_SIZE * ACCUM_STEPS

    # ----------------------------------------------------------------- model
    BACKBONE: str = "convnext_nano.in12k_ft_in1k"   # primary backbone
    BACKBONE_ALT: str = "tf_efficientnet_b0.ns_jft_in1k"  # ensemble partner
    FEATURE_DIM: int = 768                      # slice-level feature dim (auto-corrected)
    ATTENTION_HEADS: int = 8
    ATTENTION_LAYERS: int = 1
    GRU_HIDDEN: int = 256                       # bidirectional -> 512 out
    GRU_LAYERS: int = 1
    META_DIM: int = 8                           # learned sequence-metadata embedding width
    DROPOUT: float = 0.10
    HEAD_HIDDEN: int = 512

    # ------------------------------------------------------------------- loss
    ASL_GAMMA_NEG: float = 4.0
    ASL_GAMMA_POS: float = 0.0
    ASL_CLIP: float = 0.05
    ASL_MARGIN_NEG: float = 0.0
    ASL_MARGIN_POS: float = 0.0
    # Weak-supervision trust: pseudo-labelled rows get a down-weighted gradient.
    PSEUDO_LABEL_WEIGHT: float = 0.55
    REPORT_ONLY_WEIGHT: float = 0.35            # report-derived labels w/o human GT

    # ---------------------------------------------------------------- regular.
    EMA_DECAY: float = 0.995
    MIXED_PRECISION: bool = True
    AMP_DTYPE: str = "bf16"                     # auto-downgraded to fp16 on < Ampere
    LABEL_SMOOTH_NOISE: float = 0.02            # tiny logit noise on negatives (ASL-safe)

    # -------------------------------------------------------------------- CV
    N_FOLDS: int = 5
    GROUP_COL: str = "StudyInstanceUID"         # fold groups == studies (no leakage)

    # ------------------------------------------------------------------- paths
    DATA_ROOT: str = "./input"
    WORK_DIR: str = "./work"
    ARTIFACT_DIR: str = "./work/artifacts"
    VOLUME_CACHE_DIR: str = "./work/volcache"
    MODEL_DIR: str = "./work/models"

    # ----------------------------------------------------------- Kaggle limits
    MAX_TRAIN_SECONDS: float = 6.0 * 3600.0     # hard wall-clock guard (< 9h total)
    RESERVE_INFERENCE_SECONDS: float = 45 * 60.0
    START_TS: float = field(default_factory=time.time)

    # ------------------------------------------------------------ competition
    TARGET_COLS: List[str] = field(
        default_factory=lambda: [
            "ACL",
            "MCL",
            "Medial Meniscus",
            "Lateral Meniscus",
            "Medial OA",
            "Lateral OA",
            "PF OA",
            "Effusion",
            "Synovitis",
            "Baker's",
            "Contusion",
            "Fracture",
        ]
    )
    SUBMISSION_HEADER: List[str] = field(
        default_factory=lambda: ["StudyInstanceUID"] + [
            "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
            "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
            "Contusion", "Fracture",
        ]
    )

    # --------------------------------------------------------------- runtime
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # --------------------------------------------------------------- flags
    USE_PSEUDO_LABELS: bool = True
    USE_TTA: bool = True                        # original + horizontal flip
    ENABLE_CACHING: bool = True                 # cache decoded volumes to disk
    FAST_DEV_RUN: bool = False                  # subsample for smoke tests

    # ------------------------------------------------------------ augmentation
    AUG_STRONG: bool = True
    AFFINE_SCALE: Tuple[float, float] = (0.90, 1.12)
    AFFINE_ROTATE: float = 12.0
    AFFINE_SHIFT: float = 0.05
    HFLIP_PROB: float = 0.5
    DROPOUT_GRID: int = 4                       # CoarseDropout max_h/w = H/W / grid
    NOISE_STD: float = 0.02
    BRIGHTNESS: Tuple[float, float] = (-0.10, 0.15)

    # ------------------------------------------------------------- normalisation
    PERC_LOW: float = 0.5                       # 0.5th percentile clip
    PERC_HIGH: float = 99.5                     # 99.5th percentile clip
    NORM_MEAN: Tuple[float, ...] = (0.449, 0.449, 0.449)
    NORM_STD: Tuple[float, ...] = (0.221, 0.221, 0.221)

    def elapsed(self) -> float:
        """Seconds since the notebook kernel started."""
        return time.time() - self.START_TS

    def time_left_for_training(self) -> float:
        """Wall-clock seconds still usable for training given the inference reserve."""
        return self.MAX_TRAIN_SECONDS - self.elapsed() - self.RESERVE_INFERENCE_SECONDS

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


CFG = Config()

# Alias used by the spec's prose (TARGET_COLS lives on the config instance).
TARGET_COLS: List[str] = CFG.TARGET_COLS
DEVICE: torch.device = torch.device(CFG.DEVICE)
NUM_WORKERS: int = CFG.NUM_WORKERS


def set_deterministic_seed(seed: int = CFG.SEED) -> None:
    """Pin RNGs across Python / NumPy / PyTorch / CuDNN for reproducible OOF scores."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if CFG.DETERMINISTIC:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf16_reduced_precision_reduction = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:  # older torch signature
            torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True


set_deterministic_seed(CFG.SEED)


def resolve_amp_dtype(cfg: Config = CFG) -> torch.dtype:
    """Pick the widest safe autocast dtype supported by the active GPU."""
    if cfg.DEVICE == "cpu":
        return torch.float32
    major: int = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
    if major >= 8 and cfg.AMP_DTYPE == "bf16":      # Ampere+ (A100/L4/T4->no bf16)
        return torch.bfloat16
    return torch.float16                            # Pascal / Volta / Turing fallback


def make_dirs(cfg: Config = CFG) -> None:
    """Create all writable artefact directories (idempotent)."""
    for d in (cfg.WORK_DIR, cfg.ARTIFACT_DIR, cfg.VOLUME_CACHE_DIR, cfg.MODEL_DIR):
        Path(d).mkdir(parents=True, exist_ok=True)


make_dirs(CFG)

print(f"[config] device={CFG.DEVICE} torch={torch.__version__} "
      f"amp={resolve_amp_dtype(CFG)} folds={CFG.N_FOLDS} K={CFG.NUM_SLICES} "
      f"img={CFG.IMAGE_SIZE} backbone={CFG.BACKBONE}")
print(f"[config] targets={len(CFG.TARGET_COLS)} -> {CFG.TARGET_COLS}")
print(f"[config] work dirs ready: {CFG.WORK_DIR}")
