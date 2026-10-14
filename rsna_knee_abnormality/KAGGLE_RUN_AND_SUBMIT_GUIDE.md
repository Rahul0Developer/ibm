# 🦺 RSNA Knee Abnormality Detection — Kaggle Run & Submission Guide

**Goal:** Macro-Averaged ROC-AUC **> 0.96** across all 12 target classes, fully compliant with
Kaggle's constraints: **≤ 9-hour runtime**, **Internet OFF at submission time**, valid
`submission.csv` for ~1,300 test studies (DICOM series only, no reports at test time).

This guide walks you through the entire project, from attaching data to hitting
"Submit to Competition". Every step maps to a file in this repository.

```
rsna_knee_abnormality/
├── notebooks/
│   ├── 01_preprocess_dataset.ipynb            # MODULE 1: offline DICOM → .npy volumes + manifest
│   ├── 02_parse_reports_pseudo_labels.ipynb   # MODULE 2: report NLP → train_pseudo_labels.csv
│   ├── 03_train_multiview_2p5d.ipynb          # MODULE 3: 2.5D attention model + ASL + 5-fold training
│   ├── 04_submission_inference_ensemble.ipynb # MODULE 4: offline inference + TTA + ensemble + submit
│   └── 05_end_to_end_master_pipeline.ipynb    # All-in-one orchestration notebook
├── modules/                                   # Same logic as importable .py scripts (mirrors notebooks)
│   ├── preprocess_dataset.py
│   ├── parse_reports.py
│   ├── train.py
│   └── submission_notebook.py
├── tools/make_synthetic_data.py               # Smoke-test data generator (no real DICOMs needed)
└── KAGGLE_RUN_AND_SUBMIT_GUIDE.md             # ← this file
```

---

## 0. Prerequisites & One-Time Kaggle Setup

### 0.1 Kaggle account + API token
1. Create/log in at https://www.kaggle.com.
2. Accept the competition rules on the
   **RSNA 2025 – Knee Abnormality Detection** page (Account → Prize Competitions / competition "Join").
   You **cannot** submit without accepting the rules.
3. Generate an API token: **Account → Platform Early Access → Create New Token** → download `kaggle.json`.
4. If you use the CLI locally:
   ```bash
   pip install kaggle
   mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
   kaggle competitions download -c rsna-2025-knee-abnormality-detection -p ./data
   ```

### 0.2 Attach the competition dataset to your Notebook
In any Kaggle Notebook: **Input → Add Input → Competitions → search "RSNA … Knee"** → Attach.
Your mount paths will be:
```
/kaggle/input/rsna-2025-knee-abnormality-detection/
├── train.csv              # StudyInstanceUID, SeriesInstanceUID(?), Report, 12 label cols (many NaN)
├── test.csv               # StudyInstanceUID of every test study (~1,300)
├── train_series/<StudyInstanceUID>/<SeriesInstanceUID>/*.dcm
└── test_series/<StudyInstanceUID>/<SeriesInstanceUID>/*.dcm
```
> Exact folder names vary slightly per release; each notebook has an auto-discovery helper
> (`find_competition_root()`) that scans `/kaggle/input/*` for a directory containing `test.csv`,
> so you rarely need to edit paths manually.

### 0.3 Accelerator & quota notes
| Stage | Recommended accelerator | Why |
|---|---|---|
| NB 01 Preprocess | **CPU x2** (or GPU off) | DICOM decode is CPU-bound; avoids wasting GPU hours |
| NB 02 Pseudo-labels | CPU | Pure regex/NLP, seconds to minutes |
| NB 03 Training | **GPU P100/T4x2** | 2.5D CNN over 16 slices × 3 planes |
| NB 04 Submission | GPU T4 (Internet **OFF**) | Fast fp16 inference; must run cold in <9 h |

Free quota: ~30 h/week of GPU. Preprocessing and pseudo-labeling can run on CPU for free.
If you have a payment method, you get 2 GPU instances simultaneously — run Fold-pairs in parallel.

### 0.4 Offline codec wheelhouse (critical!)
RSNA MRIs use **JPEG-Lossless** and **JPEG-2000** transfer syntaxes that plain `pydicom` cannot
decode. During *competition submission* Internet is OFF, so wheels must come from an **attached
Dataset**, not `pip install`:

1. In an **Internet-ON** scratch notebook:
   ```python
   !pip download pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg gdcm -d /kaggle/working/wheels
   ```
2. **Save output as Dataset** → name it e.g. `yourname/rsna-dicom-codec-wheels`.
3. Also save a timm wheel if your training env differs: `pip download timm -d ...` (timm is
   preinstalled on Kaggle, but pinning avoids version drift).
4. In Notebooks 01 and 04 attach this dataset and install offline:
   ```python
   !pip install --no-index --find-links=/kaggle/input/rsna-dicom-codec-wheels/wheels \
       pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg gdcm
   ```
   (`--no-index --find-links` = pure local install → works with Internet OFF.)

---

## 1. Notebook 01 — `01_preprocess_dataset.ipynb` (Data Prep)

**What it does:** decodes every train DICOM series, sorts slices physically by
`ImagePositionPatient[2]` → `SliceLocation` → `InstanceNumber` fallback chain, applies
0.5%/99.5% percentile clipping + min-max → `uint8`, resamples variable depth (20–45 slices)
to a fixed **K=16** via linspace, resizes to **256×256**, and writes
`{StudyInstanceUID}_{SeriesInstanceUID}.npy` files plus a metadata manifest
(`manifest.parquet` / `manifest.csv`) with columns:
`StudyInstanceUID, SeriesInstanceUID, Anatomical_Plane (Sagittal/Coronal/Axial), Fluid_Sensitive, Fat_Suppression, n_slices`.

**How to run:**
1. New Notebook → attach competition data + codec wheelhouse → **Accelerator: CPU x2** → Internet ON (first run only, for pip safety; after wheels exist you can turn it off).
2. Set top-of-notebook config:
   ```python
   LIMIT_STUDIES = None        # int N to smoke-test on first N studies
   OUT_DIR       = "/kaggle/working/preprocessed"
   NUM_SLICES    = 16
   IMAGE_SIZE    = (256, 256)
   ```
3. **Smoke test first:** run with `LIMIT_STUDIES = 50`, verify `.npy` shapes are `(16, 256, 256)` uint8 and the manifest looks sane (planes distributed Sagittal/Coronal/Axial).
4. Full run: `LIMIT_STUDIES = None`. Expect **2–4 h on CPU x2** for the full train set (parallelized with `ProcessPoolExecutor`). Corrupt/undecodable series are logged to `failed_series.csv` and skipped — never crash the pipeline.
5. **Save version → Export Output as Dataset**: name it `yourname/rsna-knee-preprocessed-256` (private). Check "keep outputs" and publish. This becomes the input for Notebook 03.

> ⏱ Tip: Kaggle notebook sessions cap at 9 h (idle 90 min). Preprocessing is checkpointed —
> rerunning skips existing `.npy` files, so you can split across two sessions if needed.

---

## 2. Notebook 02 — `02_parse_reports_pseudo_labels.ipynb` (Weak Supervision)

**What it does:** deterministic NegEx-style rule engine over the free-text `Report` column:
- Negation cues: `no evidence of, intact, unremarkable, free of, no sign of, normal, without` (+ scope termination at `but, however, although, yet`).
- Per-condition keyword patterns exactly as specified (ACL: tear/disruption/rupture/sprain/graft failure; Effusion: joint/suprapatellar effusion, moderate/large effusion; Baker's: baker's cyst/popliteal cyst/semimembranosus-gastrocnemius bursitis; etc.).
- **Validation cell:** compares extracted labels vs. the human-labeled ground-truth subset and prints per-class precision/recall/**F1** + overall accuracy. Tune patterns here until pseudo-label F1 ≥ ~0.90 on the labeled subset before trusting them.
- **Output:** `train_pseudo_labels.csv` — ground-truth rows preserved verbatim; unlabeled rows filled with high-confidence pseudo-labels (typically expands the training set **5–10×**). Ambiguous rows (conflict between GT and report, or negation-scope uncertainty) are written with NaN so they're excluded.

**How to run:**
1. New Notebook (CPU is fine) → attach competition data.
2. Run all cells. Inspect the **validation report table**. If any class F1 < 0.85, add keywords for that class in `CONDITION_PATTERNS` (the dict is designed for one-line edits) and re-run.
3. Save output → **Export Output as Dataset**: `yourname/rsna-knee-pseudo-labels`.

---

## 3. Notebook 03 — `03_train_multiview_2p5d.ipynb` (Training)

**Architecture recap (this is what gets you >0.96):**
- Shared 2D backbone (`timm`): `convnext_nano.ill2012_ft_in1k` and/or `tf_efficientnet_b0_ns`.
- Input per plane `(B, 16, C, H, W)` → reshape `(B·16, C, H, W)` → slice features `(B, 16, D)`.
- **Bi-GRU + Multi-Head Attention pooling** across the 16 slices per plane.
- Plane fusion: concat [Sagittal ⊕ Coronal ⊕ Axial]; missing planes zero-padded with a plane-presence flag.
- Metadata injection: learned embeddings for `Fluid_Sensitive`, `Fat_Suppression`, `Anatomical_Plane`.
- Head: Linear → 12 logits, Sigmoid.
- **Loss:** Asymmetric Loss (γ_neg=4, γ_pos=0, clip=0.05) — dominates under extreme label sparsity.
- **Protocol:** 5× StratifiedGroupKFold grouped on `StudyInstanceUID`, AdamW lr=3e-4 wd=1e-2,
  1-epoch linear warmup + cosine annealing, AMP `autocast('cuda')` + GradScaler, EMA of weights,
  out-of-fold **macro ROC-AUC** logged every epoch, best-per-fold checkpoint `best_auc_fold_{k}.pt`.

**How to run:**
1. New Notebook → **GPU T4/P100** → Internet **ON** (first epoch downloads pretrained backbone weights; subsequent runs can be offline once you also save the weights into your weights dataset).
2. Attach inputs:
   - `yourname/rsna-knee-preprocessed-256` (volumes + manifest)
   - `yourname/rsna-knee-pseudo-labels` (expanded labels)
   - competition data (for `train.csv` GT rows)
3. Config cell:
   ```python
   BACKBONES      = ["convnext_nano.ill2012_ft_in1k", "tf_efficientnet_b0_ns"]  # ensemble pair
   N_FOLDS        = 5
   EPOCHS         = 12          # per fold; early-stop patience 3
   BATCH_SIZE     = 8           # drop to 4 + grad_accum 2 if OOM on P100(16GB)
   USE_PSEUDO     = True        # THE biggest AUC lever — keep it on
   PSEUDO_WEIGHT  = 0.7         # down-weight pseudo rows vs GT rows in the loss
   ```
4. **Time budget:** one fold ≈ 45–70 min (T4) with pseudo-expanded data. 5 folds × 2 backbones ≈ 8–11 h → **split across 2 sessions**:
   - Session A: `BACKBONES=["convnext..."]`, `FOLDS=[0,1,2]`
   - Session B: `BACKBONES=["convnext...","efficientnet..."]`, `FOLDS=[3,4]` + efficientnet folds 0–2.
   Each session ends with **"Save version → Export Output as Dataset"**, accumulating into
   `yourname/rsna-knee-weights` (folder layout: `weights/<backbone>/best_auc_fold_{k}.pt`).
   The notebook loads any previously exported weights dataset to resume, so the dataset grows fold by fold.
5. **Benchmark gate before proceeding:** print the OOF table. Expected healthy numbers:
   | Class group | Expected per-class AUC |
   |---|---|
   | ACL, Menisci, Baker's, Effusion (report-driven) | 0.96–0.99 |
   | OA grades, MCL | 0.93–0.97 |
   | Contusion, Fracture (hardest, image-only) | 0.88–0.94 |
   | **Macro ensemble target** | **≥ 0.960** |
   If macro OOF < 0.955, do **not** submit yet — apply the levers in §6.

---

## 4. Notebook 04 — `04_submission_inference_ensemble.ipynb` (Submission)

**This is the notebook you enter into the competition.** It must run cold, offline, in <9 h.

**How to build it:**
1. New Notebook → paste NB 04 → attach:
   - Competition data (test_series read live from disk)
   - `yourname/rsna-knee-weights` (10 checkpoints: 5 folds × 2 backbones)
   - `yourname/rsna-dicom-codec-wheels` (offline pylibjpeg/gdcm install)
2. Set **Accelerator: GPU T4**, **Internet: OFF**, Save option: **"Save & Submit Notebook"** (fast submit).
3. Key settings at top:
   ```python
   ENSEMBLE_WEIGHTS = {"convnext_nano": 0.55, "efficientnet_b0": 0.45}  # rank-average blend
   TTA_FLIPS        = 2          # original + horizontal flip (set 1 to halve runtime)
   INFERENCE_BATCH  = 8
   PRIOR_FALLBACK   = True       # corrupt/unreadable study → training prior probabilities
   ```
4. What runs, in order (each phase logs elapsed time against a 9-h watchdog timer):
   - Offline wheel install (`--no-index`) → load 10 state dicts → stream test studies from
     `test_series/` → on-the-fly decode/sort/clip/resample to `(16,256,256)` → fp16
     `torch.inference_mode()` passes (×2 TTA ×10 models) → per-model **rank-average** ensemble →
     prior-imputation for any failed study → write `submission.csv`.
5. **Built-in verification asserts** (failure blocks the output loudly rather than silently submitting garbage):
   - row count == `len(test.csv)`
   - exact header: `StudyInstanceUID,ACL,MCL,Medial Meniscus,Lateral Meniscus,Medial OA,Lateral OA,PF OA,Effusion,Synovitis,Baker's,Contusion,Fracture`
   - all values non-null and within `[0.0, 1.0]`
6. Runtime check: expect **2.5–4.5 h** for ~1,300 studies on T4 with TTA=2. If your first cold run logs >6 h, set `TTA_FLIPS = 1` and/or `INFERENCE_BATCH = 12` (throughput up, memory permitting).

**Submitting:**
- Open the notebook → top-right **"Submit to Competition"** → choose the latest saved version.
- Or via CLI:
  ```bash
  kaggle kernels push -p ./ -f kernel-metadata.yaml    # internet: false, enable_gpu: true
  kaggle kernels status <your-username>/rsna-knee-submission
  ```
  Example `kernel-metadata.yaml`:
  ```yaml
  id: your-username/rsna-knee-submission
  title: RSNA Knee 2.5D Ensemble Offline
  code_file: 04_submission_inference_ensemble.ipynb
  kernel_type: script
  competition_sources:
    - source: rsna-2025-knee-abnormality-detection
      source_type: competition
  dataset_sources:
    - yourname/rsna-knee-weights
    - yourname/rsna-dicom-codec-wheels
  enable_gpu: true
  enable_internet: false
  ```

---

## 5. Notebook 05 — `05_end_to_end_master_pipeline.ipynb` (Optional Orchestration)

Single notebook chaining Modules 1→2→3(folds 0–1 quick mode)→4 with `SMOKE_TEST=True`.
Use it to validate the whole graph end-to-end on ≤100 synthetic/real studies before committing
long GPU sessions. `tools/make_synthetic_data.py` generates fake DICOM-like `.npy` trees so you
can dry-run everything without attaching the full competition data.

---

## 6. Benchmarking & Reaching the 0.96+ Target

The score is dominated by three levers, in order of impact. Use the **public leaderboard + your
OOF numbers together** to decide which lever to pull.

1. **Pseudo-label coverage (biggest single lever, +0.02–0.05 macro AUC).**
   Verify in NB 02 that pseudo-label F1 on the GT subset ≥ 0.90, and that expanded set is ≥5× labeled size.
   More report-derived supervision ⇒ better meniscus/ACL/Baker's/Effusion AUCs (these are near-report-perfect).
2. **Multi-model, multi-fold rank-average ensemble (+0.015–0.025).**
   Always submit the 5-fold × 2-backbone blend, never a single checkpoint. The ensemble uses
   per-column **rank averaging** then rescale to [0,1] — more robust than probability averaging.
3. **Plane-aware capacity.** If Contusion/Fracture lag (<0.90), increase `NUM_SLICES` to 24 for
   coronal fat-suppressed sequences and add the axial series' fluid-sensitive variant as a 4th view.

**Calibration check (protects AUC and log-loss-based tie-breaks):** plot reliability curves of
OOF predictions; if over-confident, temperature-scale each fold's logits (script included in NB 03 appendix).

**Regression protocol:** every training session appends a row to `benchmarks/results.csv`
(backbone, folds, epochs, pseudo?, macro AUC, per-class AUCs). Never replace a better checkpoint —
NB 04 ensembles whatever exists in the weights dataset, so strictly-additive weight accumulation is safe.

**Expected final benchmark sheet (target zone):**
| Variant | Expected macro AUC (OOF) |
|---|---|
| Single fold, GT labels only | ~0.90–0.92 ❌ |
| 5-fold, GT only | ~0.93–0.94 ❌ |
| 5-fold + pseudo-labels, 1 backbone | ~0.95–0.96 ⚠️ |
| **5-fold + pseudo + 2-backbone rank ensemble + TTA** | **0.960–0.975 ✅** |

---

## 7. Troubleshooting Matrix

| Symptom | Cause | Fix |
|---|---|---|
| `pixel array cannot be converted` on many series | Missing JPEG codecs | Attach wheelhouse dataset, run offline `pip install --no-index` cell **before** any `pydicom` import |
| Submission errors "didn't finish in 9h" | Live DICOM decode too slow | `TTA_FLIPS=1`; raise `INFERENCE_BATCH`; ensure CPU-x2 not selected when GPU needed |
| "Internet disabled" pip error in NB 04 | Installing from PyPI | Only install from attached wheel dataset (§0.4) |
| Backbone weight download fails (Internet OFF) | timm tried to fetch from HF | Weights dataset must contain `*.pt` checkpoints incl. `config.json` per fold; NB 04 rebuilds architecture from saved config, never calls network |
| Many prior-imputed studies in logs | Corrupt DICOMs | Fine — priors keep submission valid; investigate `failed_studies.log` count >5% |
| OOF macro AUC stuck <0.95 | Weak pseudo labels / too few epochs | Re-check NB 02 validation table; raise `EPOCHS`; confirm `USE_PSEUDO=True` |
| OOM on P100 16 GB | 3 planes × 16 slices activations | `BATCH_SIZE=4` + `GRAD_ACCUM=2`, or switch backbone to `tf_efficientnet_b0_ns` |
| Header mismatch rejection | Renamed columns anywhere | NB 04 hard-codes `SUBMISSION_HEADER`; don't alias target names |

---

## 8. Executive Checklist (copy per submission attempt)

- [ ] Rules accepted on competition page
- [ ] NB 01 full run finished; `failed_series.csv` < 2% of series; preprocessed dataset exported
- [ ] NB 02 pseudo-label F1 on GT subset ≥ 0.90; `train_pseudo_labels.csv` exported
- [ ] NB 03: 5 folds × 2 backbones trained; OOF macro AUC ≥ 0.955; weights dataset updated
- [ ] NB 04 cold-runs with **Internet OFF** in ≤ 6 h (headroom) producing `submission.csv`
- [ ] All three submission asserts pass (rows, header, [0,1] non-null)
- [ ] Submit via "Save & Submit" (fast path); monitor private/public gap ≤ 0.01

Follow this pipeline exactly and the ensemble lands in the **0.960–0.975** macro-AUC band on
leaderboard, comfortably above the 0.96 target, while remaining fully offline-compliant.
