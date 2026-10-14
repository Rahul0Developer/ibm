"""
MODULE 1 — Offline DICOM Preprocessor for RSNA Knee Abnormality Detection.

Reads raw `.dcm` files from `train_series/` (and optionally `test_series/`) on CPU,
using a process pool to avoid I/O / decode bottlenecks, and persists each MRI series
as a fixed-shape ``uint8`` volume:

    (NUM_SLICES, H, W) == (16, 256, 256), values in [0, 255]

Pipeline per series:
    1. Robustly read every slice with pydicom (graceful fallbacks for JPEG Lossless,
       JPEG-2000, RLE — requires pylibjpeg-openjpeg / pylibjpeg-libjpeg / gdcm if the
       native decoder cannot unpack the transfer syntax).
    2. Physically sort slices by ImagePositionPatient[2] -> SliceLocation -> InstanceNumber.
    3. Percentile intensity clipping (0.5% / 99.5%) followed by Min-Max scaling to uint8.
    4. Uniform depth resampling of variable slice counts (typically 20-45) to K=16 slices
       using linear (linspace) interpolation along the depth axis.
    5. Spatial resize to 256x256.
    6. Save as .npy keyed by f"{StudyInstanceUID}_{SeriesInstanceUID}.npy".
    7. Emit a metadata manifest (parquet/csv) linking StudyInstanceUID, SeriesInstanceUID,
       AnatomicalPlane, FluidSensitive, FatSuppression, Rows, Columns, NumberSlicesAvailable.

Usage (Kaggle):
    !python preprocess_dataset.py --series-root /kaggle/input/rsna-str-pkc-dataset/train_series \
        --out-root /kaggle/working/preprocessed/train_volumes --manifest /kaggle/working/train_manifest.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import pydicom
    from pydicom.errors import InvalidDicomError
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"pydicom is required for this module: {exc}")

# Optional fast decoders. pydicom >= 2.2 auto-routes through these when installed.
_DECODER_HINT = ""
try:  # noqa: SIM105
    import pylibjpeg  # type: ignore  # noqa: F401
    _DECODER_HINT += " pylibjpeg"
except Exception:
    pass
try:  # noqa: SIM105
    import pylibjpeg_openjpeg  # type: ignore  # noqa: F401
    _DECODER_HINT += " openjpeg"
except Exception:
    pass
try:  # noqa: SIM105
    import pylibjpeg_libjpeg  # type: ignore  # noqa: F401
    _DECODER_HINT += " libjpeg"
except Exception:
    pass
try:  # noqa: SIM105
    import gdcm  # type: ignore  # noqa: F401
    _DECODER_HINT += " gdcm"
except Exception:
    pass

try:
    import cv2  # OpenCV is preinstalled on Kaggle; used for fast spatial resizing.
    _HAVE_CV2 = True
except Exception:  # pragma: no cover
    _HAVE_CV2 = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(processName)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("preprocess")


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PreprocessConfig:
    num_slices: int = 16
    image_size: Tuple[int, int] = (256, 256)
    p_low: float = 0.5           # lower percentile clip (%)
    p_high: float = 99.5         # upper percentile clip (%)
    save_dtype: str = "uint8"    # persisted dtype
    compress: bool = False       # use .npz (compressed) instead of raw .npy
    max_workers: int = 0         # 0 => cpu_count
    skip_existing: bool = True


# --------------------------------------------------------------------------------------
# DICOM reading helpers (thread/process safe)
# --------------------------------------------------------------------------------------
def _safe_get(ds: "pydicom.Dataset", name: str, default: Any = None) -> Any:
    """Fetch a DICOM attribute defensively (missing tags are extremely common)."""
    try:
        val = getattr(ds, name, default)
    except Exception:
        return default
    if val is None or (isinstance(val, (list, tuple)) and len(val) == 0):
        return default
    return val


def _sort_key_factory(ds_list: List["pydicom.Dataset"]) -> Tuple[str, Any]:
    """Choose the most reliable physical-ordering key available across the series."""
    def _ipp_z(ds: "pydicom.Dataset") -> float:
        ipp = _safe_get(ds, "ImagePositionPatient", None)
        try:
            return float(ipp[2])
        except Exception:
            return float("nan")

    def _slice_loc(ds: "pydicom.Dataset") -> float:
        sl = _safe_get(ds, "SliceLocation", None)
        try:
            return float(sl)
        except Exception:
            return float("nan")

    def _inst_num(ds: "pydicom.Dataset") -> float:
        n = _safe_get(ds, "InstanceNumber", None)
        try:
            return float(n)
        except Exception:
            return float("nan")

    for fn, label in ((_ipp_z, "ImagePositionPatient"), (_slice_loc, "SliceLocation"), (_inst_num, "InstanceNumber")):
        vals = [fn(d) for d in ds_list]
        finite = [v for v in vals if np.isfinite(v)]
        # Require that at least 80% of slices carry the tag and values vary.
        if len(finite) >= max(2, int(0.8 * len(vals))) and len(set(finite)) > 1:
            return label, fn
    return "InstanceNumber", _inst_num


def load_series_slices(series_dir: Path) -> Tuple[List["pydicom.Dataset"], str]:
    """Load all DICOM files inside a series directory, sorted physically.

    Returns:
        (sorted datasets, sort_key_used)
    Raises:
        RuntimeError if no readable slices exist.
    """
    files: List[Path] = []
    try:
        entries = list(os.scandir(str(series_dir)))
    except OSError:
        entries = []
    for e in entries:
        if e.is_file():
            files.append(Path(e.path))
    if not files:
        raise RuntimeError(f"No files found under {series_dir}")

    datasets: List["pydicom.Dataset"] = []
    n_fail = 0
    for fp in files:
        try:
            # force=True tolerates missing preamble / non-standard headers.
            ds = pydicom.dcmread(str(fp), force=True)
            if "PixelData" not in ds and not hasattr(ds, "pixel_data_array"):
                n_fail += 1
                continue
            datasets.append(ds)
        except (InvalidDicomError, Exception) as exc:  # noqa: BLE001
            n_fail += 1
            logger.debug("Unreadable file %s: %s", fp, exc)
    if not datasets:
        raise RuntimeError(f"Zero readable slices in {series_dir} ({n_fail} failures)")

    key_label, key_fn = _sort_key_factory(datasets)
    # Stable sort; NaN keys pushed to the end but replaced with interpolated ordering later.
    try:
        datasets_sorted = sorted(datasets, key=lambda d: (key_fn(d) if np.isfinite(key_fn(d)) else float("inf")))
    except Exception:
        datasets_sorted = datasets
    return datasets_sorted, key_label


def dataset_to_float_array(ds: "pydicom.Dataset") -> np.ndarray:
    """Convert one slice's pixel data to a 2D float32 array with rescale applied.

    Handles multi-frame images and applies RescaleSlope/Intercept defensively.
    """
    try:
        arr = ds.pixel_array
    except Exception as exc:  # Transfer syntax not natively supported.
        msg = str(exc)
        raise RuntimeError(
            f"Pixel decode failed (transfer syntax={_safe_get(ds,'TransferSyntaxUID','?')})."
            f" Install offline wheels pylibjpeg-openjpeg/pylibjpeg-libjpeg/gdcm.{_DECODER_HINT}. Err: {msg}"
        ) from exc

    arr = np.asarray(arr)
    if arr.ndim == 3:  # Multi-frame: take mean over frames (rare for knee MRI single slices).
        arr = arr.reshape(arr.shape[0], -1).mean(axis=1).reshape(arr.shape[1:])
    arr = arr.astype(np.float32)

    slope = _safe_get(ds, "RescaleSlope", 1.0)
    intercept = _safe_get(ds, "RescaleIntercept", 0.0)
    try:
        arr = arr * float(slope) + float(intercept)
    except Exception:
        pass
    return arr


# --------------------------------------------------------------------------------------
# Normalization & resampling
# --------------------------------------------------------------------------------------
def percentile_normalize_stack(vol: np.ndarray, p_low: float, p_high: float) -> np.ndarray:
    """Percentile-clip then min-max scale a (D,H,W) float volume into uint8 [0,255].

    The percentile bounds are computed globally over the whole series so that relative
    tissue contrast between slices is preserved (scanner-specific outlier suppression).
    """
    finite = vol[np.isfinite(vol)]
    if finite.size == 0:
        return np.zeros_like(vol, dtype=np.uint8)
    lo = float(np.percentile(finite, p_low))
    hi = float(np.percentile(finite, p_high))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        # Degenerate constant series -> center it.
        med = float(np.median(finite))
        lo, hi = med - 1.0, med + 1.0
    out = (vol - lo) / (hi - lo)
    out = np.clip(out, 0.0, 1.0)
    out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
    return (out * 255.0).round().astype(np.uint8)


def resample_depth(vol: np.ndarray, k: int) -> np.ndarray:
    """Uniform 1D linspace resampling of the depth axis from D slices to exactly k slices.

    Uses linear interpolation between adjacent slices (float math on uint8 input).
    """
    d = vol.shape[0]
    if d == k:
        return vol
    if d == 1:
        return np.repeat(vol, k, axis=0)
    src = np.arange(d, dtype=np.float64)
    dst = np.linspace(0.0, d - 1.0, k)
    x0 = np.floor(dst).astype(np.int64)
    x1 = np.minimum(x0 + 1, d - 1)
    w = (dst - x0).astype(np.float32)[:, None, None]
    v0 = vol[x0].astype(np.float32)
    v1 = vol[x1].astype(np.float32)
    interp = v0 * (1.0 - w) + v1 * w
    return np.clip(interp, 0, 255).astype(np.uint8)


def resize_slice(slice_img: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    h, w = target_hw
    if _HAVE_CV2:
        return cv2.resize(slice_img, (w, h), interpolation=cv2.INTER_AREA)
    # Pure-numpy bilinear fallback (slower but dependency-free).
    from PIL import Image
    im = Image.fromarray(slice_img)
    im = im.resize((w, h), resample=Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def resize_volume(vol: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    if vol.shape[1] == target_hw[0] and vol.shape[2] == target_hw[1]:
        return vol
    return np.stack([resize_slice(s, target_hw) for s in vol], axis=0)


# --------------------------------------------------------------------------------------
# Per-series worker (executed in a separate process)
# --------------------------------------------------------------------------------------
def process_one_series(args: Tuple[str, str, str, Dict[str, Any]]) -> Dict[str, Any]:
    study_uid, series_uid, series_path_str, cfg_dict = args
    cfg_dict = dict(cfg_dict)  # defensive copy: never mutate the shared job payload
    out_root = Path(cfg_dict.pop("_out_root"))
    cfg = PreprocessConfig(**cfg_dict)
    series_path = Path(series_path_str)
    ext = "npz" if cfg.compress else "npy"
    out_fp = out_root / f"{study_uid}_{series_uid}.{ext}"

    record: Dict[str, Any] = {
        "StudyInstanceUID": study_uid,
        "SeriesInstanceUID": series_uid,
        "VolumePath": str(out_fp),
        "status": "failed",
    }
    t0 = time.time()
    try:
        if cfg.skip_existing and out_fp.exists():
            record["status"] = "skipped"
            record["AnatomicalPlane"] = "Unknown"
            return record

        ds_list, sort_key = load_series_slices(series_path)
        ref = ds_list[0]

        planes = str(_safe_get(ref, "AnatomicalPlaneType", _safe_get(ref, "AnatomicalOrientationType", "Unknown")))
        planes = planes.upper() if planes else "UNKNOWN"
        # Infer plane from ImageOrientationPatient when the tag is absent/unhelpful.
        if planes == "UNKNOWN" or planes == "OTHER":
            iop = _safe_get(ref, "ImageOrientationPatient", None)
            try:
                row_cos = np.array([float(x) for x in iop[:3]])
                col_cos = np.array([float(x) for x in iop[3:]])
                n = np.cross(row_cos, col_cos)          # slice-direction unit normal
                ax = int(np.argmax(np.abs(n)))
                # A normal along x  <=> slices stacked left-right  <=> SAGITTAL view.
                planes = {0: "SAGITTAL", 1: "CORONAL", 2: "AXIAL"}.get(ax, "UNKNOWN")
            except Exception:
                planes = "UNKNOWN"

        fluid = str(_safe_get(ref, "DerivedSeriesDescription", "") or "") + " " + \
                str(_safe_get(ref, "SequenceName", "") or "") + " " + \
                str(_safe_get(ref, "SeriesDescription", "") or "")
        fs_blob = fluid.upper()
        fluid_sensitive = int(any(t in fs_blob for t in ("PDFS", "T2 FS", "T2FS", "STIR", "FLAIR", "FS", "FATSAT", "SPIR")))
        fat_suppression = int(any(t in fs_blob for t in ("FS", "FATSAT", "SPIR", "STIR", "FAT SAT", "FAT SUPP")))

        # Decode pixels
        arrays: List[np.ndarray] = []
        decode_errors = 0
        for ds in ds_list:
            try:
                arrays.append(dataset_to_float_array(ds))
            except RuntimeError:
                decode_errors += 1
        if decode_errors > len(ds_list) * 0.5 or not arrays:
            raise RuntimeError(
                f"{decode_errors}/{len(ds_list)} slices undecodable — install pylibjpeg-openjpeg/libjpeg wheels offline."
            )
        # Pad any skipped slices with zeros so depth stays consistent.
        while len(arrays) < len(ds_list):
            arrays.append(np.zeros_like(arrays[0]))

        vol_raw = np.stack(arrays, axis=0).astype(np.float32)  # (D,H,W)
        vol_u8 = percentile_normalize_stack(vol_raw, cfg.p_low, cfg.p_high)
        vol_resampled = resample_depth(vol_u8, cfg.num_slices)
        vol_final = resize_volume(vol_resampled, cfg.image_size)

        out_fp.parent.mkdir(parents=True, exist_ok=True)
        # np.save appends ".npy" when the path lacks that suffix, so build tmp paths
        # by string concatenation (out_fp.with_suffix() would mangle dotted UIDs).
        tmp_fp = Path(str(out_fp) + ".tmp")
        if cfg.compress:
            np.savez_compressed(str(tmp_fp), arr=vol_final)  # writes tmp_fp.npz
            os.replace(str(tmp_fp) + ".npz", out_fp)
        else:
            np.save(str(tmp_fp), vol_final)                   # writes tmp_fp.npy
            os.replace(str(tmp_fp) + ".npy", out_fp)

        record.update({
            "status": "ok",
            "AnatomicalPlane": planes.title(),
            "FluidSensitive": fluid_sensitive,
            "FatSuppression": fat_suppression,
            "Rows": int(_safe_get(ref, "Rows", vol_final.shape[1])),
            "Columns": int(_safe_get(ref, "Columns", vol_final.shape[2])),
            "NumSlicesOriginal": int(len(ds_list)),
            "SortKeyUsed": sort_key,
            "TransferSyntaxUID": str(_safe_get(ref, "TransferSyntaxUID", "unknown")),
            "SeriesDescription": str(_safe_get(ref, "SeriesDescription", "")),
            "Seconds": round(time.time() - t0, 3),
        })
        return record
    except Exception as exc:  # noqa: BLE001
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc(limit=3)
        logger.error("FAILED %s/%s: %s", study_uid, series_uid, exc)
        return record


# --------------------------------------------------------------------------------------
# Discovery & orchestration
# --------------------------------------------------------------------------------------
def discover_series(series_root: Path) -> List[Tuple[str, str, str]]:
    """Walk train_series/{StudyUID}/{SeriesUID}/ collecting (study, series, path) triples."""
    triples: List[Tuple[str, str, str]] = []
    root = Path(series_root)
    if not root.exists():
        raise FileNotFoundError(f"series root does not exist: {root}")
    for study_entry in os.scandir(str(root)):
        if not study_entry.is_dir():
            continue
        study_uid = study_entry.name
        for ser_entry in os.scandir(study_entry.path):
            if not ser_entry.is_dir():
                continue
            triples.append((study_uid, ser_entry.name, ser_entry.path))
    return triples


def run_preprocessing(
    series_root: str | Path,
    out_root: str | Path,
    manifest_path: str | Path,
    cfg: PreprocessConfig = PreprocessConfig(),
    limit: Optional[int] = None,
) -> "List[Dict[str, Any]]":
    """Main entrypoint: preprocess all series under `series_root` into `.npy` volumes + manifest."""
    series_root = Path(series_root)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    triples = discover_series(series_root)
    if limit:
        triples = triples[:limit]
    logger.info("Discovered %d series under %s", len(triples), series_root)

    cfg_dict = asdict(cfg)
    cfg_dict["_out_root"] = str(out_root)
    jobs = [(s, u, p, cfg_dict) for (s, u, p) in triples]

    records: List[Dict[str, Any]] = []
    workers = cfg.max_workers if cfg.max_workers > 0 else max(1, (os.cpu_count() or 4))
    logger.info("Spawning process pool with %d workers (decoders:%s)", workers, _DECODER_HINT or " native-only")

    if workers == 1:
        for j in jobs:
            records.append(process_one_series(j))
    else:
        from multiprocessing import Pool
        with Pool(processes=workers, maxtasksperchild=32) as pool:
            for i, rec in enumerate(pool.imap_unordered(process_one_series, jobs, chunksize=4)):
                records.append(rec)
                if (i + 1) % 500 == 0 or (i + 1) == len(jobs):
                    ok = sum(r["status"] in ("ok", "skipped") for r in records)
                    logger.info("Progress: %d/%d series processed (ok=%d)", i + 1, len(jobs), ok)

    # Persist manifest (csv always; parquet best-effort).
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    import csv
    fieldnames = ["StudyInstanceUID", "SeriesInstanceUID", "VolumePath", "status", "AnatomicalPlane",
                  "FluidSensitive", "FatSuppression", "Rows", "Columns", "NumSlicesOriginal",
                  "SortKeyUsed", "TransferSyntaxUID", "SeriesDescription", "Seconds", "error"]
    with open(manifest_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            writer.writerow(r)
    try:
        import pandas as pd
        df = pd.DataFrame(records)
        pq = manifest_path.with_suffix(".parquet")
        df.to_parquet(pq, index=False)
    except Exception:
        pass

    ok = sum(r["status"] in ("ok", "skipped") for r in records)
    failed = [r for r in records if r["status"] == "failed"]
    summary = {
        "total_series": len(records),
        "ok_or_skipped": ok,
        "failed": len(failed),
        "out_root": str(out_root),
        "manifest": str(manifest_path),
    }
    with open(out_root / "preprocess_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Preprocessing complete: %s", summary)
    if failed:
        logger.warning("First failure example: %s", failed[0].get("error", "n/a"))
    return records


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="RSNA Knee DICOM preprocessor (CPU, offline-safe)")
    parser.add_argument("--series-root", required=True, help="Root dir containing StudyUID/SeriesUID/*.dcm")
    parser.add_argument("--out-root", required=True, help="Where to write {study}_{series}.npy volumes")
    parser.add_argument("--manifest", required=True, help="Output manifest CSV path")
    parser.add_argument("--num-slices", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--p-low", type=float, default=0.5)
    parser.add_argument("--p-high", type=float, default=99.5)
    parser.add_argument("--compress", action="store_true", help="Save .npz compressed instead of .npy")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test cap on number of series")
    ns = parser.parse_args(argv)

    cfg = PreprocessConfig(
        num_slices=ns.num_slices,
        image_size=(ns.image_size, ns.image_size),
        p_low=ns.p_low,
        p_high=ns.p_high,
        compress=ns.compress,
        max_workers=ns.workers,
    )
    run_preprocessing(ns.series_root, ns.out_root, ns.manifest, cfg=cfg, limit=ns.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
