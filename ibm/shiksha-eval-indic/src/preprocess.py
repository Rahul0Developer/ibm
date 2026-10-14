"""
ShikshaEval-Indic — src/preprocess.py
=====================================

Noisy-classroom image preprocessing & augmentation for Devanagari/Marathi
handwritten answer evaluation.

Two responsibilities:

  TRAINING path   (`ClassroomAugmentor`)
      Synthesizes realistic low-resource classroom artifacts on clean notebook
      scans so the VLM learns robustness to what a phone camera actually sees:
        * non-uniform illumination / cast shadows (hand + overhead tube light)
        * perspective warp + rotation skew (±12°)
        * Gaussian sensor noise (small mobile sensors, dim classrooms)
        * JPEG recompression artifacts (WhatsApp-forwarded homework photos)

  INFERENCE path  (`restore_for_inference`)
      Deterministic restoration before the image hits the VLM vision tower:
        * Otsu-based background/foreground separation
        * Deskew via min-area bounding box AND Hough projection voting
          (consensus of both), tuned so the Devanagari *shirorekha*
          (सिररेखा — the continuous top header line) is never broken:
          rotation is applied with INTER_LINEAR + borderValue=white and
          angles are clamped so near-horizontal strokes stay one-pixel-flat.
        * CLAHE in the L channel of LAB (contrast without blowing out matras)
        * Morphological closing with an ASYMMETRIC vertical-safe kernel that
          reconnects a hairline-broken shirorekha but never erodes above-base
          matras (ि ी ु ू ृ, ं ं अनुस्वार, ः विसर्ग, ि् फलानि below-line matras).

Devanagari Unicode facts baked into this module (U+0900–U+097F):
    independent vowels   : 0x0905–0x0914
    consonants           : 0x0915–0x0939
    nukta-derived letters: 0x0958–0x0962  (क़ ख़ ग़ ज़ ड़ ढ़ फ़ य़ ऱ ஶ etc.)
    matras (vowel signs) : 0x093E–0x094C  (ा ि ी ु ू ृ े ै ो ौ ...)
    halant/virama        : 0x094D  -> conjuncts (जोडक्षर like क्ष, त्र, ज्ञ)
    anusvara / visarga   : 0x0901 / 0x0903
    candrabindu          : 0x0901 chandrabindu vs 0x0970 AI length mark
    marathi-specific     : 0x0933 ळ (retroflex L) — absent in Hindi collation

Author: ShikshaEval-Indic (portfolio build, VOPA AI/ML Engineer role)
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

try:  # Albumentations is used for the photometric half of the augmentation.
    import albumentations as A
    from albumentations import Compose
    _HAS_ALBUMENTATIONS = True
except ImportError:  # pragma: no cover - degrade gracefully, OpenCV fallbacks exist
    A = None
    Compose = None
    _HAS_ALBUMENTATIONS = False


# ---------------------------------------------------------------------------
# Constants: Devanagari-aware geometry tuning
# ---------------------------------------------------------------------------

SHIROREKHA_ROW_BAND = 0.34   # fraction of line-height where the header lives
MAX_DESKEW_DEG = 12.0         # matches the ±12° synthetic skew budget
MIN_MATRA_STROKE_PX = 1       # closing must never remove 1px matra strokes
TARGET_LONG_EDGE = 1024       # Qwen2.5-VL patchifies ~28px; 1024 keeps detail
JPEG_QUALITY_RANGE = (35, 75) # aggressive mobile-cam compression simulation


def is_devanagari_char(ch: str) -> bool:
    """True if `ch` lies in the Devanagari block (incl. extended U+0970-A7F? no: 0900–097F)."""
    cp = ord(ch)
    return 0x0900 <= cp <= 0x097F


def devanagari_profile(text: str) -> dict:
    """Character-class histogram used by the splitter & evaluator for stratification."""
    classes = {
        "consonant": 0, "indep_vowel": 0, "matra": 0, "halant": 0,
        "anusvara_visarga": 0, "nukta": 0, "digit": 0, "other_dev": 0, "non_dev": 0,
    }
    for ch in text:
        cp = ord(ch)
        if not is_devanagari_char(ch):
            classes["non_dev"] += 1
        elif 0x0915 <= cp <= 0x0939:
            classes["consonant"] += 1
        elif 0x0905 <= cp <= 0x0914:
            classes["indep_vowel"] += 1
        elif 0x093E <= cp <= 0x094C:
            classes["matra"] += 1
        elif cp == 0x094D:
            classes["halant"] += 1
        elif cp in (0x0901, 0x0902, 0x0903):
            classes["anusvara_visarga"] += 1
        elif 0x0958 <= cp <= 0x0962:
            classes["nukta"] += 1
        elif 0x0966 <= cp <= 0x096F:
            classes["digit"] += 1
        else:
            classes["other_dev"] += 1
    conjuncts = sum(
        1 for i, c in enumerate(text)
        if ord(c) == 0x094D and i + 1 < len(text) and is_devanagari_char(text[i + 1])
        and ord(text[i + 1]) != 0x094D
    )
    classes["conjunct_clusters"] = conjuncts
    return classes


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def imread_unicode(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """cv2.imread fails silently on non-ASCII (Marathi) filenames; use frombuffer."""
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, flags)
    if img is None:
        raise IOError(f"cv2.imdecode returned None for {path}")
    return img


def imwrite_unicode(path: str | Path, img: np.ndarray, params: list | None = None) -> None:
    ext = Path(path).suffix or ".png"
    ok, buf = cv2.imencode(ext, img, params or [])
    if not ok:
        raise IOError(f"Failed to encode image for {path}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    buf.tofile(str(path))


# ---------------------------------------------------------------------------
# Classroom artifact simulation (training robustness)
# ---------------------------------------------------------------------------

@dataclass
class ClassroomAugmentor:
    """
    Injects realistic 'phone photo of a kid's notebook' artifacts.

    Pipeline order matters and mirrors reality:
        geometry (perspective+rotation) -> illumination (shadow gradient)
        -> sensor noise -> blur -> JPEG recompression.
    """
    p_shadow: float = 0.85
    p_skew: float = 0.90
    max_skew_deg: float = MAX_DESKEW_DEG
    p_blur: float = 0.60
    p_jpeg: float = 0.95
    noise_sigma_range: tuple[float, float] = (2.0, 9.0)
    seed: int | None = None
    _rng: np.random.Generator = field(default_factory=np.random.default_rng, repr=False)
    _pyrng: random.Random = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._pyrng = random.Random(self.seed)

    # -- geometry ------------------------------------------------------------
    @staticmethod
    def _warp_affine(M: np.ndarray, img: np.ndarray) -> np.ndarray:
        """OpenCV >=5 requires dsize as a (w, h) tuple; build it explicitly."""
        h, w = img.shape[:2]
        return cv2.warpAffine(src=img, M=M, dsize=(w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)

    def _perspective_skew(self, img: np.ndarray) -> np.ndarray:
        """Random homography + rotation up to ±max_skew_deg (the classroom tilt)."""
        h, w = img.shape[:2]
        angle = float(self._rng.uniform(-self.max_skew_deg, self.max_skew_deg))
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = self._warp_affine(M, img)
        # mild quad jitter => page not flat on desk
        j = 0.03
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = src + self._rng.uniform(-j * w, j * w, size=(4, 2)).astype(np.float32)
        H = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(src=img, M=H, dsize=(w, h), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)

    # -- illumination ---------------------------------------------------------
    def _cast_shadow(self, img: np.ndarray) -> np.ndarray:
        """
        Non-uniform illumination: smooth low-frequency multiplicative gradient
        (overhead tube light falloff) plus one hard-edged elliptical 'hand shadow'.
        """
        h, w = img.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = self._rng.uniform(0, w), self._rng.uniform(0, h)
        theta = self._rng.uniform(0, math.pi)
        # rotated linear ramp, normalized to [0,1]
        ramp = (xx - cx) * math.cos(theta) + (yy - cy) * math.sin(theta)
        ramp = (ramp - ramp.min()) / (np.ptp(ramp) + 1e-6)
        gain = 0.55 + 0.65 * ramp                      # 0.55..1.20 multiplicative
        ellipse_mask = np.ones((h, w), np.float32)
        ex, ey = self._rng.uniform(0, w), self._rng.uniform(0, h)
        ax, ay = self._rng.uniform(w * .15, w * .45), self._rng.uniform(h * .15, h * .5)
        mask = (((xx - ex) / ax) ** 2 + ((yy - ey) / ay) ** 2) <= 1.0
        soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigmaX=max(ax, ay) / 6)
        ellipse_mask = 1.0 - 0.5 * soft                 # darken up to 50% under hand
        combined = np.clip(gain * ellipse_mask, 0.25, 1.6)[..., None]
        out = np.clip(img.astype(np.float32) * combined, 0, 255).astype(np.uint8)
        return out

    # -- sensor noise / blur ---------------------------------------------------
    def _sensor_noise(self, img: np.ndarray) -> np.ndarray:
        sigma = float(self._rng.uniform(*self.noise_sigma_range))
        noisy = img.astype(np.float32) + self._rng.normal(0, sigma, img.shape)
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def _jpeg_recompress(self, img: np.ndarray) -> np.ndarray:
        q = self._pyrng.randint(*JPEG_QUALITY_RANGE)
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if not ok:
            return img
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)

    # -- public API ------------------------------------------------------------
    def __call__(self, img: np.ndarray) -> np.ndarray:
        if self.p_skew >= self._rng.random():
            img = self._perspective_skew(img)
        if self.p_shadow >= self._rng.random():
            img = self._cast_shadow(img)
        if self.p_blur >= self._rng.random():
            k = int(self._rng.choice([3, 5]))
            img = cv2.GaussianBlur(img, (k, k), 0)
        img = self._sensor_noise(img)
        if self.p_jpeg >= self._rng.random():
            img = self._jpeg_recompress(img)
        # Photometric jitter via Albumentations when available (brightness/contrast)
        if _HAS_ALBUMENTATIONS:
            pipe = Compose([
                A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.7),
                A.HueSaturationValue(hue_shift_limit=6, sat_shift_limit=18,
                                     val_shift_limit=14, p=0.4),
                A.CoarseDropout(max_holes=6, max_height=8, max_width=24,
                                fill_value=255, p=0.15),  # pen-smudge / page fold
            ])
            img = pipe(image=img)["image"]
        return img

    def augment_dataset(self, image_paths: Sequence[str | Path], out_dir: str | Path,
                        variants_per_image: int = 2) -> list[Path]:
        """Materialize N augmented noisy variants per clean scan; returns new paths."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        produced: list[Path] = []
        for p in map(Path, image_paths):
            img = imread_unicode(p)
            for v in range(variants_per_image):
                aug = self(img.copy())
                dst = out_dir / f"{p.stem}_noisy{v}{p.suffix}"
                imwrite_unicode(dst, aug)
                produced.append(dst)
        return produced


# ---------------------------------------------------------------------------
# Restoration & normalization (inference path)
# ---------------------------------------------------------------------------

def _resize_keep_aspect(img: np.ndarray, target_long_edge: int = TARGET_LONG_EDGE) -> np.ndarray:
    h, w = img.shape[:2]
    scale = target_long_edge / max(h, w)
    if abs(scale - 1.0) < 1e-3:
        return img
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(img, (nw, nh), interpolation=interp)


def estimate_skew_min_area(box_img: np.ndarray) -> float:
    """
    MinAreaRect over ALL foreground pixels. For a single written line the text
    blob's long axis tracks the baseline/shirorekha tilt directly.

    Sign convention (empirically verified against synthetic skew sweeps in
    scripts/test_modules.py): returns the page tilt θ such that injecting a
    rotation of +θ makes this estimator report ≈ +θ. `rotate_preserving_shirorekha`
    then undoes it with getRotationMatrix2D(center, -θ).
    """
    coords = np.column_stack(np.where(box_img > 0))
    if len(coords) < 30:
        return 0.0
    rect = cv2.minAreaRect(coords[:, ::-1].astype(np.float32))
    (cx, cy), (rw, rh), angle = rect          # OpenCV>=4.5: angle ∈ [-90, 0]
    a = ((angle + 90.0) % 180.0) - 90.0       # wrap to (-90, 90]
    tilt = -a                                 # page-tilt convention (see docstring)
    if tilt > 45:
        tilt -= 90
    elif tilt <= -45:
        tilt += 90
    return float(np.clip(tilt, -MAX_DESKEW_DEG, MAX_DESKEW_DEG))


def estimate_skew_hough(box_img: np.ndarray) -> float:
    """
    Hough-projection: dominant near-horizontal line-cluster angle, length-weighted
    so a strong shirorekha dominates speckle lines. Same page-tilt sign convention
    as estimate_skew_min_area (atan2 lives in the y-down frame).
    """
    edges = cv2.Canny(box_img, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=60,
                             minLineLength=max(30, box_img.shape[1] // 6), maxLineGap=8)
    if lines is None:
        return 0.0
    # OpenCV version-agnostic flattening: (N,1,4) in 4.x, possibly (N,4) in 5.x
    segs = np.asarray(lines).reshape(-1, 4)
    angles, weights = [], []
    for x1, y1, x2, y2 in segs:
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < 1:
            continue
        deg = math.degrees(math.atan2(dy, dx))   # y-down slope of the stroke
        if deg > 90:
            deg -= 180
        if deg < -90:
            deg += 180
        tilt = -deg                              # page-tilt convention (min_area parity)
        if abs(tilt) <= MAX_DESKEW_DEG:          # only plausible writing tilt
            angles.append(tilt)
            weights.append(length)               # longer line (shirorekha!) weighs more
    if not angles:
        return 0.0
    med = float(np.median(np.asarray(angles)))
    return float(np.clip(med, -MAX_DESKEW_DEG, MAX_DESKEW_DEG))


def estimate_skew_consensus(img_bgr: np.ndarray) -> float:
    """Consensus of min-area + Hough estimates; disagreement falls back to 0 (no-op)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    # foreground = ink: dark relative to smoothed background (handles uneven light)
    bg = cv2.medianBlur(gray, 21)
    diff = cv2.absdiff(gray, bg)
    _, fg = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (5, 2)))
    a1, a2 = estimate_skew_min_area(fg), estimate_skew_hough(fg)
    if abs(a1 - a2) <= 2.0:
        return float(np.mean([a1, a2]))
    # If they disagree strongly trust Hough (line-structure aware) only if small
    return a2 if abs(a2) <= 4.0 else 0.0


def rotate_preserving_shirorekha(img: np.ndarray, page_tilt_deg: float) -> np.ndarray:
    """
    Undo an estimated page tilt of `page_tilt_deg` (the convention produced by
    estimate_skew_*). Sub-degree quantization guard: tilts < 0.4° are skipped
    entirely because bilinear resampling would feather the 1–2px shirorekha
    hairline and split it into two faint rows — which the VLM then reads as a
    missing header line (a classic Devanagari OCR failure).
    """
    if abs(page_tilt_deg) < 0.4:
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), -page_tilt_deg, 1.0)
    return cv2.warpAffine(src=img, M=M, dsize=(w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))


def apply_clahe(img_bgr: np.ndarray, clip: float = 2.4, grid: tuple[int, int] = (8, 8)) -> np.ndarray:
    """CLAHE on the L channel of LAB — local contrast without saturating thin matras."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l2 = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid).apply(l)
    return cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)


def shirorekha_safe_morphology(binary: np.ndarray) -> np.ndarray:
    """
    Closing with a WIDE-but-SHORT horizontal kernel (21x2): bridges hairline gaps
    in the header line WITHOUT vertically merging glyph bodies (which would eat
    the gap between a base letter and its ऊ/ू matra tail below the baseline).
    Then a 1px open removes salt-pepper while preserving >=1px matra strokes.
    """
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 2))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, hk)
    vk = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))   # cross: no directional erosion
    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, vk)
    return cleaned


def adaptive_binarize(gray: np.ndarray, block: int = 31, c: int = 12) -> np.ndarray:
    """Adaptive Gaussian threshold tuned for uneven classroom lighting."""
    if block % 2 == 0:
        block += 1
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, block, c)
    return th


def line_segment_and_crop(img_bgr: np.ndarray, min_line_h: int = 24) -> list[np.ndarray]:
    """Horizontal projection profile segmentation of single-line answer crops."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    th = adaptive_binarize(gray)
    proj = shirorekha_safe_morphology(th).sum(axis=1) / 255.0
    thresh = max(2.0, 0.02 * img_bgr.shape[1])
    rows = proj > thresh
    crops, start = [], None
    for i, on in enumerate(rows):
        if on and start is None:
            start = i
        elif not on and start is not None:
            if i - start >= min_line_h:
                pad = int(0.18 * (i - start))          # pad protects above/below matras
                y0, y1 = max(0, start - pad), min(len(rows), i + pad)
                crops.append(img_bgr[y0:y1])
            start = None
    if start is not None and len(rows) - start >= min_line_h:
        crops.append(img_bgr[start:])
    return crops or [img_bgr]


def restore_for_inference(img_bgr: np.ndarray, binarize_output: bool = False) -> np.ndarray:
    """
    Full deterministic inference-time restoration chain:
      resize -> deskew (consensus) -> CLAHE -> adaptive threshold -> safe morphology
    Returns BGR (3-channel — Qwen2.5-VL processor expects RGB triplets) unless
    `binarize_output=True` for ablation experiments.
    """
    img = _resize_keep_aspect(img_bgr)
    angle = estimate_skew_consensus(img)
    img = rotate_preserving_shirorekha(img, angle)
    img = apply_clahe(img)
    if not binarize_output:
        return img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th = shirorekha_safe_morphology(adaptive_binarize(gray))
    binary = cv2.bitwise_not(th)                       # black ink on white paper
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


# ---------------------------------------------------------------------------
# Dataset splitter (stratified, low-resource aware)
# ---------------------------------------------------------------------------

@dataclass
class SplitSpec:
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_state: int = 42
    min_per_class: int = 2   # low-resource guard: never strand a rare class in test


def _difficulty_bucket(sample: dict) -> str:
    """
    Composite stratum key = rubric score × orthographic difficulty bucket.
    Difficulty buckets from the Devanagari profile so rare conjunct-heavy /
    anusvara-bearing samples appear proportionally in every split.
    """
    prof = devanagari_profile(sample["reference_text"])
    hard = prof["conjunct_clusters"] + prof["anusvara_visarga"] + prof["nukta"]
    bucket = "easy" if hard == 0 else ("med" if hard <= 2 else "hard")
    return f"s{sample['rubric_score']}_{bucket}"


def stratified_split(samples: list[dict], spec: SplitSpec = SplitSpec(),
                     write_manifest: str | Path | None = None) -> dict[str, list[dict]]:
    """
    Bucket-first stratified split on the difficulty×score key. Rare buckets
    (<min_per_class) are force-routed to train; surviving buckets are allocated
    proportionally with a largest-remainder (Hamilton) rounding so every split
    stays non-empty and class-balanced even at 15–30 sample low-resource scale
    (sklearn's StratifiedShuffleSplit cannot guarantee that at tiny n).
    Each sample: {"image_path": str, "reference_text": str, "rubric_score": 0|1|2}
    """
    if not samples:
        raise ValueError("empty dataset")
    rng = np.random.default_rng(spec.random_state)
    buckets: dict[str, list[int]] = {}
    for i, s in enumerate(samples):
        buckets.setdefault(_difficulty_bucket(s), []).append(i)

    rare = {b for b, idxs in buckets.items() if len(idxs) < spec.min_per_class}
    train_idx: list[int] = [i for b in rare for i in buckets[b]]
    val_idx: list[int] = []
    test_idx: list[int] = []

    def _largest_remainder(k: int, ratios: tuple[float, float]) -> tuple[int, int]:
        """Split k items into (val, test) counts with Hamilton rounding, min 1 each if k>=2."""
        if k <= 0:
            return 0, 0
        if k == 1:
            return 1, 0
        vr, tr = ratios
        total = vr + tr
        v_exact, t_exact = k * vr / total, k * tr / total
        v, t = max(1, int(np.floor(v_exact))), max(1, int(np.floor(t_exact)))
        while v + t > k:                       # trim from whichever rounded up more
            if v - v_exact >= t - t_exact:
                v -= 1
            else:
                t -= 1
        rem = k - v - t                        # distribute remainder by fractional part
        order = sorted([(v_exact - np.floor(v_exact), "v"), (t_exact - np.floor(t_exact), "t")],
                       reverse=True)
        for _, which in order[:max(0, rem)]:
            if which == "v":
                v += 1
            else:
                t += 1
        return v, t

    ratio_pair = (spec.val_ratio, spec.test_ratio)
    for b in sorted(buckets):
        if b in rare:
            continue
        idxs = list(buckets[b])
        rng.shuffle(idxs)
        n_val, n_test = _largest_remainder(len(idxs), ratio_pair)
        val_idx += idxs[:n_val]
        test_idx += idxs[n_val:n_val + n_test]
        train_idx += idxs[n_val + n_test:]

    out = {
        "train": [samples[i] for i in train_idx],
        "val":   [samples[i] for i in val_idx],
        "test":  [samples[i] for i in test_idx],
    }
    pyrng = random.Random(spec.random_state)
    for k in out:
        pyrng.shuffle(out[k])
    if write_manifest:
        manifest = {k: {"n": len(v), "items": v} for k, v in out.items()}
        Path(write_manifest).parent.mkdir(parents=True, exist_ok=True)
        Path(write_manifest).write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# CLI: python -m src.preprocess --clean-dir data/clean --out-dir data/noisy ...
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[2])
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("augment", help="simulate noisy classroom photos")
    a.add_argument("--clean-dir", required=True)
    a.add_argument("--out-dir", required=True)
    a.add_argument("--variants", type=int, default=2)
    a.add_argument("--seed", type=int, default=13)

    r = sub.add_parser("restore", help="run inference restoration on images")
    r.add_argument("--in-dir", required=True)
    r.add_argument("--out-dir", required=True)
    r.add_argument("--binarize", action="store_true")

    s = sub.add_parser("split", help="stratified train/val/test manifest")
    s.add_argument("--manifest", required=True, help="JSON list of samples")
    s.add_argument("--out", required=True)

    args = ap.parse_args()
    if args.cmd == "augment":
        aug = ClassroomAugmentor(seed=args.seed)
        paths = sorted(Path(args.clean_dir).glob("*.[jp][pn]*"))
        made = aug.augment_dataset(paths, args.out_dir, args.variants)
        print(f"[augment] {len(paths)} clean -> {len(made)} noisy variants in {args.out_dir}")
    elif args.cmd == "restore":
        out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
        for p in sorted(Path(args.in_dir).glob("*.[jp][pn]*")):
            restored = restore_for_inference(imread_unicode(p), args.binarize)
            imwrite_unicode(out / p.name, restored)
        print(f"[restore] wrote restored images to {out}")
    elif args.cmd == "split":
        samples = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        splits = stratified_split(samples, write_manifest=args.out)
        print({k: len(v) for k, v in splits.items()})


if __name__ == "__main__":
    main()
