#!/usr/bin/env python3
"""
Utility: Synthetic RSNA-Knee-format test-data generator (CPU-only, offline-safe).

Creates a miniature dataset that mimics the real competition layout so every
module of this repository can be smoke-tested end-to-end without downloading
the ~1 TB official data:

    <out_root>/
        train.csv          StudyInstanceUID, Report, 12 targets (many NaN)
        test.csv           StudyInstanceUID only
        train_series/      {StudyUID}/{SeriesUID}/*.dcm   (uncompressed MR Image storage)
        test_series/       same layout for test studies
        sample_submission.csv

DICOM slices are generated with plane-dependent geometry (ImageOrientation-
Patient / ImagePositionPatient), SequenceName metadata (e.g. "SAG PD FS",
"COR T1", "AX T2 PDFS") and a synthetic abnormality signal baked into pixel
intensity so downstream benchmarking measures real learning, not noise.

Usage:
    python tools/make_synthetic_data.py --out-root /tmp/rsna_synth --n-train 60 --n-test 12
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import MRImageStorage, ExplicitVRLittleEndian

TARGET_COLS: List[str] = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
    "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]

# Plane -> (row cosine, col cosine) direction cosines of ImageOrientationPatient
PLANE_IOP: Dict[str, tuple] = {
    "SAGITTAL": ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    "CORONAL":  ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "AXIAL":    ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
}

# Which abnormalities are visible on which plane (drives signal injection).
PLANE_VISIBLE_TARGETS: Dict[str, List[str]] = {
    "SAGITTAL": ["ACL", "Medial Meniscus", "Lateral Meniscus", "Baker's", "PF OA"],
    "CORONAL":  ["MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "Contusion", "Fracture"],
    "AXIAL":    ["Effusion", "Synovitis", "PF OA", "Contusion", "Fracture"],
}

SEQ_BY_PLANE: Dict[str, List[str]] = {
    "SAGITTAL": ["SAG T1", "SAG PD FS", "SAG PDFS"],
    "CORONAL":  ["COR T1", "COR PD FS", "COR STIR"],
    "AXIAL":    ["AX T1", "AX T2 PDFS", "AX PD"],
}


def _uid(seed: int, prefix: str = "1.2.826.0.1.3680043.9.9.") -> str:
    """Deterministic fake DICOM UID from an integer seed."""
    return f"{prefix}{seed:012d}"


def make_slice_array(plane: str, n_slices: int, idx: int, rng: np.random.Generator,
                     abnormal_mask: Dict[str, int]) -> np.ndarray:
    """Render one 192x192 float slice of a synthetic knee MRI.

    Abnormalities modulate localized intensity patterns so a model that reads
    pixels can actually detect them (measurable benchmarking).
    """
    h = w = 192
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # Base anatomy: two rounded bones + joint space ellipse.
    bone1 = np.exp(-(((xx - w * 0.35) ** 2 + (yy - h * 0.62) ** 2) / (2 * 28.0 ** 2)))
    bone2 = np.exp(-(((xx - w * 0.65) ** 2 + (yy - h * 0.62) ** 2) / (2 * 28.0 ** 2)))
    cart = np.exp(-(((xx - w * 0.5) ** 2) / (2 * 55.0 ** 2) + ((yy - h * 0.47) ** 2) / (2 * 12.0 ** 2)))
    img = 0.15 + 0.75 * np.maximum(bone1, bone2) + 0.35 * cart
    # Plane-specific shape variation across depth.
    t = idx / max(1, n_slices - 1)
    img += 0.10 * np.sin(2 * np.pi * t) * cart
    # Noise floor.
    img += rng.normal(0.0, 0.03, size=(h, w)).astype(np.float32)

    def blob(cx: float, cy: float, sigma: float, amp: float) -> None:
        nonlocal img
        img = img + amp * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2)))

    vis_targets = PLANE_VISIBLE_TARGETS[plane]
    if abnormal_mask.get("ACL", 0) and "ACL" in vis_targets and 0.35 < t < 0.75:
        blob(w * 0.47, h * 0.35, 9.0, 0.55)          # bright fluid at tear site
    if abnormal_mask.get("MCL", 0) and "MCL" in vis_targets and 0.3 < t < 0.8:
        blob(w * 0.18, h * 0.55, 8.0, 0.45)
    if abnormal_mask.get("Medial Meniscus", 0) and "Medial Meniscus" in vis_targets:
        blob(w * 0.33, h * 0.50, 7.0, 0.40)
    if abnormal_mask.get("Lateral Meniscus", 0) and "Lateral Meniscus" in vis_targets:
        blob(w * 0.68, h * 0.50, 7.0, 0.40)
    if abnormal_mask.get("Medial OA", 0) and "Medial OA" in vis_targets:
        blob(w * 0.33, h * 0.55, 14.0, -0.25)         # joint-space narrowing
    if abnormal_mask.get("Lateral OA", 0) and "Lateral OA" in vis_targets:
        blob(w * 0.68, h * 0.55, 14.0, -0.25)
    if abnormal_mask.get("PF OA", 0) and "PF OA" in vis_targets:
        blob(w * 0.50, h * 0.28, 10.0, -0.30)
    if abnormal_mask.get("Effusion", 0) and "Effusion" in vis_targets and t > 0.25:
        blob(w * 0.50, h * 0.38, 22.0, 0.60)          # suprapatellar fluid
    if abnormal_mask.get("Synovitis", 0) and "Synovitis" in vis_targets:
        blob(w * 0.50, h * 0.42, 26.0, 0.25)
    if abnormal_mask.get("Baker's", 0) and "Baker's" in vis_targets and t > 0.5:
        blob(w * 0.72, h * 0.75, 12.0, 0.55)
    if abnormal_mask.get("Contusion", 0) and "Contusion" in vis_targets:
        blob(w * 0.55, h * 0.66, 11.0, 0.50)          # marrow edema
    if abnormal_mask.get("Fracture", 0) and "Fracture" in vis_targets:
        if 0.4 < t < 0.7:
            line = np.where(np.abs(xx - (w * 0.5 + 30 * np.sin(yy / 18.0))) < 2.0, 0.7, 0.0)
            img += line.astype(np.float32)
    return np.clip(img, 0.0, 1.0)


def write_series(series_root: Path, study_uid: str, seed: int,
                 abnormal_mask: Dict[str, int], n_planes: int = 3) -> List[str]:
    """Write up to 3 series (one per plane) of uncompressed DICOM slices; returns plane names."""
    planes = ["SAGITTAL", "CORONAL", "AXIAL"][:n_planes]
    rng_master = np.random.default_rng(seed)
    used_planes: List[str] = []
    for p_i, plane in enumerate(planes):
        # Occasionally drop a plane to exercise zero-pad fallbacks.
        if rng_master.random() < 0.08:
            continue
        used_planes.append(plane)
        seq_name = SEQ_BY_PLANE[plane][int(rng_master.integers(0, 3))]
        series_uid = f"{study_uid}.{p_i + 1}"
        sdir = series_root / study_uid / series_uid
        sdir.mkdir(parents=True, exist_ok=True)
        n_slices = int(rng_master.integers(18, 30))
        row_cos, col_cos = PLANE_IOP[plane]
        for i in range(n_slices):
            arr = make_slice_array(plane, n_slices, i, np.random.default_rng(seed * 1000 + i), abnormal_mask)
            px = (arr * 4095.0).astype(np.uint16)
            ds = Dataset()
            ds.file_meta = FileMetaDataset()
            ds.file_meta.MediaStorageSOPClassUID = MRImageStorage
            ds.file_meta.MediaStorageSOPInstanceUID = _uid(seed * 100000 + i * 10 + p_i)
            ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
            ds.SOPClassUID = MRImageStorage
            ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
            ds.StudyInstanceUID = study_uid
            ds.SeriesInstanceUID = series_uid
            ds.Modality = "MR"
            ds.Rows, ds.Columns = px.shape
            ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 16, 15
            ds.PixelRepresentation, ds.SamplesPerPixel = 0, 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.SequenceName = seq_name
            ds.SeriesDescription = f"MR {plane.lower()} {seq_name}"
            ds.ImageOrientationPatient = [*row_cos, *col_cos]
            if plane == "SAGITTAL":
                ds.ImagePositionPatient = [float(-16.0 + i * 1.2), 0.0, 0.0]
            elif plane == "CORONAL":
                ds.ImagePositionPatient = [0.0, float(-16.0 + i * 1.2), 0.0]
            else:
                ds.ImagePositionPatient = [0.0, 0.0, float(-16.0 + i * 1.2)]
            ds.SliceLocation = float(i)
            ds.InstanceNumber = i + 1
            ds.PatientName = "SYNTH^KNEE"
            ds.PatientID = f"SYNTH{seed}"
            ds.RescaleIntercept, ds.RescaleSlope = 0, 1
            ds.PixelData = px.tobytes()
            pydicom.dcmwrite(sdir / f"{i:03d}.dcm", ds, write_like_original=False)
    return used_planes


def generate_report(mask: Dict[str, int], rng: np.random.Generator) -> str:
    """Build a synthetic radiology report consistent with the label mask,
    including NegEx-style negations for absent findings."""
    positive_phrases: Dict[str, List[str]] = {
        "ACL": ["there is a full thickness ACL tear", "ACL disruption with graft failure"],
        "MCL": ["MCL sprain with edema along the ligament", "high grade MCL tear"],
        "Medial Meniscus": ["medial meniscus tear involving the posterior horn",
                            "complex medial meniscus tear", "medial meniscus root tear"],
        "Lateral Meniscus": ["lateral meniscus tear", "horizontal tear of the lateral meniscus"],
        "Medial OA": ["medial compartment narrowing with medial osteophytes",
                      "medial tibiofemoral osteoarthritis"],
        "Lateral OA": ["lateral compartment narrowing", "lateral tibiofemoral osteoarthritis"],
        "PF OA": ["patellofemoral osteoarthritis", "trochlear groove arthrosis with patellar cartilage loss"],
        "Effusion": ["moderate joint effusion", "large suprapatellar effusion", "joint fluid distension"],
        "Synovitis": ["synovial thickening consistent with synovitis", "synovial proliferation"],
        "Baker's": ["a baker's cyst is present",
                    "popliteal cyst at the semimembranosus-gastrocnemius bursitis region"],
        "Contusion": ["bone contusion pattern", "bone marrow edema in the lateral femoral condyle",
                      "subchondral edema", "bone bruise"],
        "Fracture": ["a tibial fracture is noted", "cortical breach with avulsion fragment"],
    }
    negative_phrases: Dict[str, List[str]] = {
        "ACL": ["the ACL is intact", "no evidence of ACL tear"],
        "MCL": ["the MCL is unremarkable", "no sign of MCL injury"],
        "Medial Meniscus": ["no evidence of medial meniscus tear"],
        "Lateral Meniscus": ["the lateral meniscus is intact", "without lateral meniscus tear"],
        "Medial OA": ["no medial compartment narrowing", "normal medial compartment"],
        "Lateral OA": ["free of lateral tibiofemoral osteoarthritis"],
        "PF OA": ["no patellofemoral osteoarthritis"],
        "Effusion": ["no joint effusion", "no evidence of effusion"],
        "Synovitis": ["no synovial thickening"],
        "Baker's": ["no baker's cyst", "no popliteal cyst"],
        "Contusion": ["no bone contusion", "no bone marrow edema"],
        "Fracture": ["no fracture", "no cortical breach or avulsion"],
    }
    lines: List[str] = ["MRI of the knee. "]
    for tgt in TARGET_COLS:
        if mask.get(tgt, 0) == 1:
            lines.append(rng.choice(positive_phrases[tgt]).capitalize() + ". ")
        elif rng.random() < 0.75:
            # Mostly negate absent findings; keep some silent to test parser coverage.
            lines.append(rng.choice(negative_phrases[tgt]).capitalize() + ". ")
    pos_names = [t for t in TARGET_COLS if mask.get(t, 0) == 1][:3]
    lines.append("IMPRESSION: " + "; ".join(pos_names or ["unremarkable knee exam"]) + ".")
    return "".join(lines)


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Synthetic RSNA-knee-format dataset generator")
    ap.add_argument("--out-root", default="/tmp/rsna_synth")
    ap.add_argument("--n-train", type=int, default=60)
    ap.add_argument("--n-test", type=int, default=12)
    ap.add_argument("--labeled-fraction", type=float, default=0.35,
                    help="fraction of train studies with ground-truth binary labels")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)

    out = Path(args.out_root)
    (out / "train_series").mkdir(parents=True, exist_ok=True)
    (out / "test_series").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    prevalence = {"ACL": .25, "MCL": .2, "Medial Meniscus": .3, "Lateral Meniscus": .25,
                  "Medial OA": .35, "Lateral OA": .3, "PF OA": .3, "Effusion": .45,
                  "Synovitis": .2, "Baker's": .15, "Contusion": .25, "Fracture": .1}

    rows: List[Dict[str, object]] = []
    for k in range(args.n_train):
        study_uid = _uid(1000 + k, "1.2.826.0.1.3680043.9.1.")
        mask = {t: int(rng.random() < prevalence[t]) for t in TARGET_COLS}
        write_series(out / "train_series", study_uid, seed=2000 + k, abnormal_mask=mask)
        report = generate_report(mask, rng)
        labeled = rng.random() < args.labeled_fraction
        rec: Dict[str, object] = {"StudyInstanceUID": study_uid, "Report": report}
        for t in TARGET_COLS:
            rec[t] = mask[t] if labeled else np.nan
        rows.append(rec)

    train_df = pd.DataFrame(rows)
    train_df.to_csv(out / "train.csv", index=False)

    test_rows = [{"StudyInstanceUID": _uid(5000 + k, "1.2.826.0.1.3680043.9.2.")} for k in range(args.n_test)]
    for k, r in enumerate(test_rows):
        mask = {t: int(rng.random() < prevalence[t]) for t in TARGET_COLS}
        write_series(out / "test_series", str(r["StudyInstanceUID"]), seed=9000 + k, abnormal_mask=mask)
    test_df = pd.DataFrame(test_rows)
    test_df.to_csv(out / "test.csv", index=False)

    sub = test_df.copy()
    for t in TARGET_COLS:
        sub[t] = 0.5
    sub.to_csv(out / "sample_submission.csv", index=False)

    print(f"Wrote synthetic dataset to {out}")
    print(f"  train.csv: {len(train_df)} studies ({int(train_df[TARGET_COLS].notna().all(axis=1).sum())} fully-labeled)")
    print(f"  test.csv : {len(test_df)} studies")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
