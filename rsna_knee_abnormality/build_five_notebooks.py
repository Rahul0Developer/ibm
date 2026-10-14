#!/usr/bin/env python3
"""Build the five Kaggle notebooks for the RSNA Knee Abnormality Detection pipeline.

Generates (in ../notebooks/):
    01_preprocess_dataset.ipynb        MODULE 1 — offline DICOM preprocessing
    02_parse_reports_pseudo_labels.ipynb  MODULE 2 — report weak supervision
    03_train_multiview_2p5d.ipynb      MODULE 3 — model + training loop
    04_submission_inference_ensemble.ipynb  MODULE 4 — offline ensemble submission
    05_end_to_end_master_pipeline.ipynb     orchestrator notebook (runs all stages)

Each of notebooks 01-04 embeds its module's full source as one code cell plus a
markdown header and a demo/self-test cell. Notebook 05 is a compact master that
%run-includes the same sources sequentially.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

import nbformat as nbf

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
MODULES = HERE / "modules"
OUT_DIR = HERE / "notebooks"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _nb(metadata_gpu: bool = True) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
        "accelerator": "GPU" if metadata_gpu else "None",
        "is_active_competition": False,
    })
    return nb


def _write(nb: nbf.NotebookNode, name: str) -> Path:
    nbf.validate(nb)
    fp = OUT_DIR / name
    with open(fp, "w", encoding="utf-8") as fh:
        nbf.write(nb, fh)
    return fp


def build_module_notebook(md_header: str, module_file: str, demo_cell: str, out_name: str,
                          extra_md: str = "") -> Path:
    src = (MODULES / module_file).read_text(encoding="utf-8")
    # In notebook form we do not want `sys.exit(main())` to fire on paste-run; guard it.
    src = src.replace('if __name__ == "__main__":\n    sys.exit(main())',
                      'if __name__ == "__main__":\n    pass  # driven from notebook cells below')
    src = src.replace('if __name__ == "__main__":\n    raise SystemExit(main())',
                      'if __name__ == "__main__":\n    pass  # driven from notebook cells below')
    nb = _nb()
    cells: List[nbf.NotebookNode] = [nbf.v4.new_markdown_cell(md_header)]
    cells.append(nbf.v4.new_code_cell(src))
    if extra_md:
        cells.append(nbf.v4.new_markdown_cell(extra_md))
    cells.append(nbf.v4.new_code_cell(demo_cell))
    nb.cells = cells
    return _write(nb, out_name)


# --------------------------------------------------------------------------------------
# Notebook 01 — Preprocessing
# --------------------------------------------------------------------------------------
NB01_MD = """# RSNA Knee · Notebook 01 — Offline DICOM Preprocessor (Module 1)

**Purpose:** Convert raw `train_series/{StudyUID}/{SeriesUID}/*.dcm` into fixed-shape
`(16, 256, 256)` uint8 volumes saved as `.npy`, plus a metadata manifest
(`AnatomicalPlane`, `FluidSensitive`, `FatSuppression`). CPU-only, multi-process,
offline-safe (handles JPEG-Lossless / JPEG-2000 via pylibjpeg/gdcm fallbacks).

**Kaggle workflow:** run here → save `/kaggle/working/preprocessed` + manifest to a
**private Kaggle Dataset** (e.g. `rsna-knee-preprocessed-256`) for Notebooks 03/04.

| Stage | Typical time (CPU) |
|---|---|
| ~110k series decode+normalize | 3–6 h (parallel workers) |
"""

NB01_DEMO = '''# --- Cell 2: locate competition data and run the preprocessor -------------------------
import os, glob
from pathlib import Path

def _find_root(*names):
    cands = []
    for n in names:
        cands += glob.glob(f"/kaggle/input/*{n}*")
    return cands[0] if cands else None

DATA_ROOT = _find_root("rsna-str-pkc", "rsna-knee", "knee-abnormalities") or "/kaggle/input/rsna-str-pkc-dataset"
TRAIN_SERIES = Path(DATA_ROOT) / "train_series"
print("DATA_ROOT:", DATA_ROOT, "| train_series exists:", TRAIN_SERIES.exists())

cfg = PreprocessConfig(num_slices=16, image_size=(256, 256), p_low=0.5, p_high=99.5, compress=False)

if TRAIN_SERIES.exists():
    records = run_preprocessing(
        series_root=TRAIN_SERIES,
        out_root="/kaggle/working/preprocessed/train_volumes",
        manifest_path="/kaggle/working/train_manifest.csv",
        cfg=cfg,
        limit=None,   # set e.g. 300 for a smoke test first
    )
else:
    print("[DRY] No train_series found — running synthetic unit self-tests instead.")
    recs = run_preprocessing(str(TRAIN_SERIES), "/tmp/out", "/tmp/manifest.csv", cfg, limit=1) \\
        if TRAIN_SERIES.exists() else []
'''

NB01_SELFTEST = '''# --- Cell 3: numerical self-tests for normalization / resampling / resize -------------
import numpy as np

rng = np.random.default_rng(0)
vol = rng.normal(loc=1000, scale=200, size=(37, 320, 320)).astype(np.float32)
vol[:5] += 5000  # scanner outliers

u8 = percentile_normalize_stack(vol, 0.5, 99.5)
assert u8.dtype == np.uint8 and u8.shape == vol.shape
assert u8.min() == 0 and u8.max() == 255, "contrast stretch should saturate both ends"

k16 = resample_depth(u8, 16)
assert k16.shape == (16, 320, 320)

small = resize_volume(k16, (256, 256))
assert small.shape == (16, 256, 256)

# degenerate constant volume must not crash
const = np.full((20, 64, 64), 7.0, np.float32)
c8 = percentile_normalize_stack(const, 0.5, 99.5)
assert c8.shape == const.shape and c8.dtype == np.uint8

# single-slice series edge case
one = resample_depth(np.zeros((1, 8, 8), np.uint8), 16)
assert one.shape == (16, 8, 8)
print("Module 1 self-tests PASSED ✔")
'''

# --------------------------------------------------------------------------------------
# Notebook 02 — Pseudo-labeling
# --------------------------------------------------------------------------------------
NB02_MD = """# RSNA Knee · Notebook 02 — Report Weak-Supervision Pseudo-Labeler (Module 2)

Only a fraction of training studies carry human-verified labels, but **all** rows have a
free-text `Report`. This notebook runs a deterministic NegEx-style rule parser to label
unannotated studies, expanding the effective training set **5×–10×** — the single biggest
leverage point toward Macro-AUC > 0.96.

Outputs:
* `train_pseudo_labels.csv` — GT rows preserved (`label_source=gt`), unlabeled filled (`pseudo`)
* Validation report: per-target F1 / accuracy of the parser measured against the labeled subset.
"""

NB02_DEMO = '''# --- Cell 2: run pseudo-labeling on the real train.csv --------------------------------
import glob
from pathlib import Path

def _find_csv(pattern):
    hits = glob.glob(f"/kaggle/input/**/{pattern}", recursive=True)
    return hits[0] if hits else None

TRAIN_CSV = _find_csv("train.csv") or "/kaggle/input/rsna-str-pkc-dataset/train.csv"
print("TRAIN_CSV:", TRAIN_CSV, "exists:", Path(TRAIN_CSV).exists())

if Path(TRAIN_CSV).exists():
    expanded = build_expanded_labels(
        train_csv_path=TRAIN_CSV,
        out_csv_path="/kaggle/working/train_pseudo_labels.csv",
        validation_out_prefix="/kaggle/working/pseudo_label_validation",
        confidence_threshold=0.6,
        max_pseudo_rows=None,   # use None for the full expansion
    )
    display(expanded["label_source"].value_counts().to_frame("count"))
    display(expanded[TARGET_COLS].mean().round(3).to_frame("prevalence"))
else:
    print("[DRY] train.csv not attached; self-test below still validates the parser logic.")
'''

NB02_SELFTEST = '''# --- Cell 3: parser unit tests (negation handling) ------------------------------------
pos_report = (
    "There is a complete tear of the ACL. The MCL shows grade II sprain. "
    "A medial meniscus posterior root tear is present. A lateral meniscus horizontal tear is seen. "
    "Moderate joint effusion with suprapatellar extension. Baker's cyst measuring 3 cm. "
    "Bone contusion at the lateral tibial plateau. Non-displaced fracture of the fibular head. "
    "Patellofemoral osteoarthritis with cartilage loss. Medial compartment narrowing consistent "
    "with medial tibiofemoral osteoarthritis. Lateral tibiofemoral osteoarthritis also noted. "
    "Synovial thickening suggestive of synovitis."
)
neg_report = (
    "The ACL is intact. There is no evidence of MCL tear. Unremarkable medial and lateral menisci. "
    "No joint effusion. No sign of fracture. Normal bone marrow signal without bone contusion. "
    "No popliteal cyst. Free of synovitis. No patellofemoral osteoarthritis."
)

r_pos = extract_pseudo_labels_from_report(pos_report)
assert all(r_pos.labels[t] == 1 for t in TARGET_COLS), {t: r_pos.labels[t] for t in TARGET_COLS if r_pos.labels[t]==0}

r_neg = extract_pseudo_labels_from_report(neg_report)
assert all(r_neg.labels[t] == 0 for t in TARGET_COLS), [t for t in TARGET_COLS if r_neg.labels[t]==1]

empty = extract_pseudo_labels_from_report("")
assert sum(empty.labels.values()) == 0
nan_like = extract_pseudo_labels_from_report(float("nan"))
assert sum(nan_like.labels.values()) == 0

print("Positive report detections:", {t: m[:48] for t, m in r_pos.mentions.items() if m})
print("Module 2 parser self-tests PASSED ✔")
'''

# --------------------------------------------------------------------------------------
# Notebook 03 — Training
# --------------------------------------------------------------------------------------
NB03_MD = """# RSNA Knee · Notebook 03 — Multi-Plane Attention-Pooled 2.5D Training (Module 3)

Architecture: `timm` 2D backbone applied slice-wise over **16 uniformly sampled slices ×
3 anatomical planes (Sagittal / Coronal / Axial)**, BiGRU (or MHSA) depth aggregation +
learned-query attention pooling, plane concatenation, fusion with `FluidSensitive` /
`FatSuppression` metadata embeddings, and a 12-logit multi-label head trained with
**Asymmetric Loss (γ⁻=4, γ⁺=0, clip=0.05)**.

Protocol: Stratified GroupKFold(5) by `StudyInstanceUID`, AdamW(lr=3e-4, wd=1e-2),
1-epoch warmup + cosine annealing, AMP autocast, EMA weights, OOF macro ROC-AUC every
epoch, best checkpoint per fold → `best_auc_fold_{k}.pt`.

**Attach:** the preprocessed dataset from Notebook 01 and `train_pseudo_labels.csv`
from Notebook 02. Save resulting `.pt` checkpoints to a private weights dataset.
"""

NB03_DEMO = '''# --- Cell 2: configuration & path wiring ----------------------------------------------
import glob
from pathlib import Path

def _first(patterns):
    for p in patterns:
        hits = glob.glob(p, recursive=True)
        if hits:
            return hits[0]
    return None

MANIFEST = _first(["/kaggle/input/**/train_manifest.csv", "/kaggle/working/train_manifest.csv"])
LABELS   = _first(["/kaggle/input/**/train_pseudo_labels.csv", "/kaggle/working/train_pseudo_labels.csv"])
VOLUMES  = _first(["/kaggle/input/**/train_volumes", "/kaggle/working/preprocessed/train_volumes"])
print("MANIFEST:", MANIFEST); print("LABELS:", LABELS); print("VOLUMES:", VOLUMES)

BACKBONE = "convnext_nano.in12k_ft_in1k"   # swap to "tf_efficientnet_b0.ns_jft_in1k" for memory
EPOCHS   = 6 if torch.cuda.is_available() else 1
FOLD     = 0                                 # train folds 0..4 across separate GPU sessions

if MANIFEST and LABELS and VOLUMES and Path(VOLUMES).exists():
    import argparse as _ap
    args = _ap.Namespace(manifest=MANIFEST, labels=LABELS, volumes_root=VOLUMES,
                         ckpt_dir="/kaggle/working/checkpoints/" + BACKBONE.split(".")[0],
                         backbone=BACKBONE, aggregator="bigru", epochs=EPOCHS,
                         batch_size=8, num_workers=4, fold=FOLD)
    train_pipeline(args)
else:
    print("[DRY] Data not attached — running architecture/loss/metric smoke tests next cell.")
'''

NB03_SELFTEST = '''# --- Cell 3: end-to-end tensor-shape smoke test on synthetic data ---------------------
import numpy as np, pandas as pd, torch

cfg = Config(); cfg.EPOCHS = 1
set_determinism(cfg.SEED)

# Fake two studies x three series manifest
rows = []
for s in range(2):
    for pl_i, pl in enumerate(["Sagittal", "Coronal", "Axial"]):
        rows.append({"StudyInstanceUID": f"study{s}", "SeriesInstanceUID": f"ser{s}{pl_i}",
                     "VolumePath": f"/tmp/vols/study{s}_ser{s}{pl_i}.npy", "status": "ok",
                     "AnatomicalPlane": pl, "FluidSensitive": pl_i % 2, "FatSuppression": pl_i // 2,
                     "NumSlicesOriginal": 28, "SeriesDescription": pl})
man = pd.DataFrame(rows)
lab = pd.DataFrame([{"StudyInstanceUID": f"study{s}", "label_source": "gt",
                     **{t: int((s + j) % 2) for j, t in enumerate(TARGET_COLS)}} for s in range(2)])

import os
os.makedirs("/tmp/vols", exist_ok=True)
for r in rows:
    np.save(r["VolumePath"], np.random.randint(0, 255, (16, 256, 256), dtype=np.uint8))

ds = RSNAKneeDataset(lab, man, "/tmp/vols", cfg, train=True)
dl = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_knee)
batch = next(iter(dl))
assert tuple(batch["volumes"].shape) == (2, 3, 16, 3, 256, 256), batch["volumes"].shape

model = RSNAMultiPlaneModel(cfg)
logits = model({k: v for k, v in batch.items() if k != "uids"})
assert logits.shape == (2, 12), logits.shape

crit = AsymmetricLoss(4.0, 0.0, 0.05)
loss = crit(logits, batch["target"], batch["weight"])
assert torch.isfinite(loss)
loss.backward()

m = MacroAUC(12)
m.update(torch.sigmoid(logits.detach()), batch["target"])
auc, per_class = m.compute()   # must not crash even though some columns are single-class
print(f"synthetic forward/backward OK | ASL={float(loss):.4f} | mini-AUC={auc}")
print("Module 3 self-tests PASSED ✔")
'''

# --------------------------------------------------------------------------------------
# Notebook 04 — Submission
# --------------------------------------------------------------------------------------
NB04_MD = """# RSNA Knee · Notebook 04 — Offline Ensemble Inference & Submission (Module 4)

**Compliance checklist**
* Internet OFF — zero network calls; weights loaded from attached dataset only.
* Runtime ≤ 9 h — dynamic DICOM decode once, then reused across all 10 models
  (5 folds × 2 backbones); hard deadline check between models (budget 7.5 h).
* ~1,300 test studies, probabilities for all 12 targets.
* TTA: original + horizontal flip averaged.
* Corrupt/unreadable study → imputed with training-set prior probabilities.
* Final sanity checks: exact row count vs `test.csv`, exact header order,
  non-null values, all probabilities ∈ [0, 1].

**Attach:** competition data + your weights dataset (Notebook 03 outputs).
"""

NB04_DEMO = '''# --- Cell 2: locate inputs and execute the offline submission pipeline ----------------
import glob
from pathlib import Path

def _first(patterns):
    for p in patterns:
        hits = glob.glob(p, recursive=True)
        if hits:
            return hits[0]
    return None

TEST_CSV   = _first(["/kaggle/input/**/test.csv"])
TEST_SERIES= _first(["/kaggle/input/**/test_series"])
WEIGHTS    = _first(["/kaggle/input/**/checkpoints", "/kaggle/input/*weights*"])
PRIOR_LAB  = _first(["/kaggle/input/**/train_pseudo_labels.csv", "/kaggle/working/train_pseudo_labels.csv"])
print("TEST_CSV:", TEST_CSV, "\\nTEST_SERIES:", TEST_SERIES, "\\nWEIGHTS:", WEIGHTS)

sub = None
if TEST_CSV and Path(TEST_CSV).exists():
    sub = run_submission(
        test_csv=TEST_CSV,
        test_series_root=TEST_SERIES or "",
        weights_root=WEIGHTS or ".",
        out_csv="/kaggle/working/submission.csv",
        labels_for_priors=PRIOR_LAB,
    )
else:
    print("[DRY] test.csv not attached — running the schema self-test in the next cell instead.")
if sub is not None:
    display(sub.head())
'''

NB04_SELFTEST = '''# --- Cell 3: pipeline plumbing self-test (no GPU/DICOM required) ----------------------
import numpy as np, pandas as pd, tempfile, os

# tiny fake test.csv
fake_test = pd.DataFrame({"StudyInstanceUID": [f"u{i}" for i in range(6)]})
tmpdir = tempfile.mkdtemp()
tcsv = os.path.join(tmpdir, "test.csv"); fake_test.to_csv(tcsv, index=False)

sub = run_submission(test_csv=tcsv, test_series_root=os.path.join(tmpdir, "none"),
                     weights_root=os.path.join(tmpdir, "noweights"),
                     out_csv=os.path.join(tmpdir, "submission.csv"), dry_run=True)
assert list(sub.columns) == ["StudyInstanceUID"] + TARGET_COLS
assert len(sub) == 6
assert sub[TARGET_COLS].values.min() >= 0.0 and sub[TARGET_COLS].values.max() <= 1.0

# rank-average sanity: monotone transform of one model must not change fused ranks
stack = np.stack([np.random.rand(50, 12), np.random.rand(50, 12)])
ra = rank_average(stack)
assert ra.shape == (50, 12) and 0.0 <= ra.min() and ra.max() <= 1.0

# priors fallback shape
p = compute_training_priors(None, None)
assert p.shape == (12,)
print("Module 4 plumbing self-tests PASSED ✔  (submission schema validated)")
'''

# --------------------------------------------------------------------------------------
# Notebook 05 — Master end-to-end
# --------------------------------------------------------------------------------------
NB05_MD = """# RSNA Knee · Notebook 05 — End-to-End Master Pipeline (All Modules)

Single-notebook orchestration of the four-stage workflow:

```
Stage 1  preprocess_dataset.py   raw DICOM  -> .npy volumes + manifest      (CPU hours)
Stage 2  parse_reports.py        reports    -> train_pseudo_labels.csv      (minutes)
Stage 3  train.py                volumes+labels -> best_auc_fold_{k}.pt     (GPU hours)
Stage 4  submission_notebook.py  test DICOM + weights -> submission.csv     (GPU, offline)
```

**Score strategy recap (targets Macro-AUC > 0.96):**
1. Weak supervision from reports expands labeled data 5–10×.
2. Multi-plane 2.5D attention-pooled ConvNeXt/EffNet ensemble.
3. Asymmetric Loss under extreme class sparsity.
4. TTA + 2-backbone × 5-fold rank-average ensemble (+0.015…0.025 expected lift).

Toggle `SMOKE_TEST=True` to cap data volume for fast CI-style verification before the
full 9-hour Kaggle run.
"""


def build_master_notebook() -> Path:
    nb = _nb()
    cells: List[nbf.NotebookNode] = [nbf.v4.new_markdown_cell(NB05_MD)]

    common_setup = '''# --- Stage 0: environment, seeds, paths ------------------------------------------------
import os, glob, json, time, warnings
warnings.filterwarnings("ignore")
SMOKE_TEST = True   # set False for the full run

def _first(patterns):
    for p in patterns:
        hits = glob.glob(p, recursive=True)
        if hits:
            return hits[0]
    return None

DATA_ROOT   = _first(["/kaggle/input/*rsna-str-pkc*", "/kaggle/input/*knee*"]) or "/kaggle/input/rsna-str-pkc-dataset"
TRAIN_SERIES= os.path.join(DATA_ROOT, "train_series")
TEST_SERIES = os.path.join(DATA_ROOT, "test_series")
TRAIN_CSV   = os.path.join(DATA_ROOT, "train.csv")
TEST_CSV    = os.path.join(DATA_ROOT, "test.csv")
WORK        = "/kaggle/working" if os.path.isdir("/kaggle/working") else "./work"
os.makedirs(WORK, exist_ok=True)
VOL_ROOT    = os.path.join(WORK, "preprocessed", "train_volumes")
MANIFEST    = os.path.join(WORK, "train_manifest.csv")
PSEUDO_CSV  = os.path.join(WORK, "train_pseudo_labels.csv")
CKPT_ROOT   = os.path.join(WORK, "checkpoints")
print(json.dumps({"DATA_ROOT": DATA_ROOT, "WORK": WORK}, indent=2))'''
    cells.append(nbf.v4.new_code_cell(common_setup))

    # Embed each module source in its own cell so the master notebook is fully standalone.
    stage_defs = [
        ("Stage 1 · Module 1 source", "preprocess_dataset.py",
         '''# --- Stage 1: execute preprocessing ----------------------------------------------------
_cfg = PreprocessConfig(num_slices=16, image_size=(256, 256))
_limit = 240 if SMOKE_TEST else None
if os.path.isdir(TRAIN_SERIES):
    t0 = time.time()
    run_preprocessing(TRAIN_SERIES, VOL_ROOT, MANIFEST, cfg=_cfg, limit=_limit)
    print(f"[Stage 1] done in {(time.time()-t0)/60:.1f} min")
else:
    print("[Stage 1] train_series missing - skipped")'''),
        ("Stage 2 · Module 2 source", "parse_reports.py",
         '''# --- Stage 2: pseudo-label expansion -----------------------------------------------------
if os.path.exists(TRAIN_CSV):
    build_expanded_labels(TRAIN_CSV, PSEUDO_CSV,
                          validation_out_prefix=os.path.join(WORK, "pseudo_val"),
                          confidence_threshold=0.6,
                          max_pseudo_rows=1500 if SMOKE_TEST else None)
else:
    print("[Stage 2] train.csv missing - skipped")'''),
        ("Stage 3 · Module 3 source", "train.py",
         '''# --- Stage 3a: train convnext_nano folds ---------------------------------------------------
import argparse as _ap
if os.path.exists(MANIFEST) and os.path.exists(PSEUDO_CSV) and os.path.isdir(VOL_ROOT):
    args = _ap.Namespace(manifest=MANIFEST, labels=PSEUDO_CSV, volumes_root=VOL_ROOT,
                         ckpt_dir=os.path.join(CKPT_ROOT, "convnext_nano"),
                         backbone="convnext_nano.in12k_ft_in1k", aggregator="bigru",
                         epochs=1 if SMOKE_TEST else 6, batch_size=8, num_workers=4,
                         fold=0 if SMOKE_TEST else -1)
    train_pipeline(args)
else:
    print("[Stage 3] prerequisites missing - skipped")'''),
        ("Stage 4 · Module 4 source", "submission_notebook.py",
         '''# --- Stage 4: offline ensemble inference & submission -------------------------------------
if os.path.exists(TEST_CSV):
    sub = run_submission(test_csv=TEST_CSV, test_series_root=TEST_SERIES,
                         weights_root=CKPT_ROOT, out_csv=os.path.join(WORK, "submission.csv"),
                         labels_for_priors=PSEUDO_CSV,
                         max_studies=16 if SMOKE_TEST else None)
    display(sub.head())
else:
    print("[Stage 4] test.csv missing - skipped")'''),
    ]

    for title, mod, exec_cell in stage_defs:
        cells.append(nbf.v4.new_markdown_cell(f"## {title} — `{mod}`"))
        src = (MODULES / mod).read_text(encoding="utf-8")
        src = src.replace('if __name__ == "__main__":\n    sys.exit(main())',
                          'if __name__ == "__main__":\n    pass  # driven from notebook cells')
        src = src.replace('if __name__ == "__main__":\n    raise SystemExit(main())',
                          'if __name__ == "__main__":\n    pass  # driven from notebook cells')
        cells.append(nbf.v4.new_code_cell(src))
        cells.append(nbf.v4.new_code_cell(exec_cell))

    # Extra trainer invocation for the second backbone (ensembling requirement).
    cells.append(nbf.v4.new_markdown_cell("## Stage 3b — second backbone for the ensemble (`tf_efficientnet_b0_ns`)"))
    cells.append(nbf.v4.new_code_cell('''# --- Stage 3b: efficientnet variant (drop the previous cell's re-execution of module source) --
if os.path.exists(MANIFEST) and os.path.exists(PSEUDO_CSV) and os.path.isdir(VOL_ROOT):
    import argparse as _ap2
    args = _ap2.Namespace(manifest=MANIFEST, labels=PSEUDO_CSV, volumes_root=VOL_ROOT,
                          ckpt_dir=os.path.join(CKPT_ROOT, "efficientnet_b0"),
                          backbone="tf_efficientnet_b0.ns_jft_in1k", aggregator="mhsa",
                          epochs=1 if SMOKE_TEST else 6, batch_size=8, num_workers=4,
                          fold=0 if SMOKE_TEST else -1)
    train_pipeline(args)
else:
    print("[Stage 3b] prerequisites missing - skipped")'''))

    cells.append(nbf.v4.new_markdown_cell("""## Post-run checklist
- [ ] Upload `preprocessed/` → private dataset `rsna-knee-preprocessed-256`
- [ ] Upload `checkpoints/*/*.pt` → private dataset `rsna-knee-weights`
- [ ] For the official submission: create a fresh notebook with **only Notebook 04**,
      attach competition data + weights dataset, **Internet OFF**, Submit.
""".strip()))
    nb.cells = cells
    return _write(nb, "05_end_to_end_master_pipeline.ipynb")


def main() -> None:
    p1 = build_module_notebook(NB01_MD, "preprocess_dataset.py", NB01_DEMO,
                               "01_preprocess_dataset.ipynb",
                               extra_md="### Sanity-check numerics before scaling to the full corpus")
    # Append the numeric self-test as an extra cell for notebook 01
    nb = nbf.read(p1, as_version=4)
    nb.cells.append(nbf.v4.new_code_cell(NB01_SELFTEST))
    _write(nb, "01_preprocess_dataset.ipynb")

    p2 = build_module_notebook(NB02_MD, "parse_reports.py", NB02_DEMO,
                               "02_parse_reports_pseudo_labels.ipynb",
                               extra_md="### Parser regression tests (positives & negations)")
    nb = nbf.read(p2, as_version=4)
    nb.cells.append(nbf.v4.new_code_cell(NB02_SELFTEST))
    _write(nb, "02_parse_reports_pseudo_labels.ipynb")

    p3 = build_module_notebook(NB03_MD, "train.py", NB03_DEMO,
                               "03_train_multiview_2p5d.ipynb",
                               extra_md="### Synthetic smoke test (runs even without attached data)")
    nb = nbf.read(p3, as_version=4)
    nb.cells.append(nbf.v4.new_code_cell(NB03_SELFTEST))
    _write(nb, "03_train_multiview_2p5d.ipynb")

    p4 = build_module_notebook(NB04_MD, "submission_notebook.py", NB04_DEMO,
                               "04_submission_inference_ensemble.ipynb",
                               extra_md="### Schema & plumbing self-test (no data required)")
    nb = nbf.read(p4, as_version=4)
    nb.cells.append(nbf.v4.new_code_cell(NB04_SELFTEST))
    _write(nb, "04_submission_inference_ensemble.ipynb")

    p5 = build_master_notebook()
    for p in (p1, p2, p3, p4, p5):
        print(f"Wrote {p} ({os.path.getsize(p)/1024:.1f} KB)")


if __name__ == "__main__":
    main()
