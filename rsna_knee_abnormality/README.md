# RSNA Knee Abnormality Detection — Production 2.5D Ensemble (PyTorch / Kaggle)

Production-grade, modular solution targeting **Macro ROC-AUC > 0.96** on the
RSNA Knee Abnormality Detection challenge (12 binary abnormalities), fully compliant
with Kaggle constraints: ≤9 h runtime, Internet OFF at submission, valid `submission.csv`.

## Quick Start (Kaggle)

➡️ **Read [`KAGGLE_RUN_AND_SUBMIT_GUIDE.md`](KAGGLE_RUN_AND_SUBMIT_GUIDE.md) first.**
It is the complete step-by-step runbook: attaching data, offline DICOM codecs wheelhouse,
per-notebook settings, time budgets, benchmark gates, troubleshooting, and the final
"Submit to Competition" checklist.

TL;DR pipeline order:

| Step | Notebook | Output Kaggle Dataset | Accelerator |
|---|---|---|---|
| 1 | `notebooks/01_preprocess_dataset.ipynb` | `rsna-knee-preprocessed-256` | CPU x2 |
| 2 | `notebooks/02_parse_reports_pseudo_labels.ipynb` | `rsna-knee-pseudo-labels` | CPU |
| 3 | `notebooks/03_train_multiview_2p5d.ipynb` | `rsna-knee-weights` (5 folds × 2 backbones) | GPU T4/P100 |
| 4 | `notebooks/04_submission_inference_ensemble.ipynb` | `submission.csv` → Submit | GPU, **Internet OFF** |
| — | `notebooks/05_end_to_end_master_pipeline.ipynb` | smoke-test orchestration | any |

Equivalent importable scripts live in `modules/` (`preprocess_dataset.py`,
`parse_reports.py`, `train.py`, `submission_notebook.py`) for CLI/local runs:

```bash
python modules/preprocess_dataset.py --series-root ./data/train_series --out-dir ./preprocessed
python modules/parse_reports.py      --train-csv ./data/train.csv --out train_pseudo_labels.csv
python modules/train.py              --volumes ./preprocessed --labels train_pseudo_labels.csv --backbone convnext_nano.ill2012_ft_in1k --folds 0 1
python modules/submission_notebook.py --test-csv ./data/test.csv --test-series-root ./data/test_series --weights-root ./rsna-knee-weights --out-csv submission.csv
```

Use `tools/make_synthetic_data.py` to generate a fake study tree for dry-runs without real DICOMs.

## Architecture Summary

- **2.5D multi-plane model:** per-slice `timm` backbone (ConvNeXt-Nano / EfficientNet-B0-NS)
  → Bi-GRU + Multi-Head Attention pooling over K=16 uniformly resampled slices
  → plane fusion (Sagittal ⊕ Coronal ⊕ Axial, zero-padded when absent)
  → metadata embeddings (`Fluid_Sensitive`, `Fat_Suppression`, `Anatomical_Plane`) → 12 sigmoid logits.
- **Weak supervision:** NegEx-style report parser pseudo-labels unlabeled studies (5–10× data expansion) — the single biggest AUC lever.
- **Loss:** Asymmetric Loss (γ_neg=4, γ_pos=0, clip=0.05) for extreme multi-label sparsity.
- **Training:** 5× StratifiedGroupKFold by StudyInstanceUID, AdamW 3e-4/wd 1e-2, warmup+cosine, AMP fp16, EMA, OOF macro-AUC checkpoints.
- **Inference:** offline dynamic DICOM reader, prior-probability fallback for corrupt series,
  horizontal-flip TTA, per-column rank-average ensemble across all 10 checkpoints,
  strict header/row-count/[0,1] validation before writing `submission.csv`.

## Benchmark Targets

| Variant | Expected Macro AUC |
|---|---|
| GT labels only, 5-fold | ~0.93–0.94 |
| + pseudo-labels, 1 backbone | ~0.95–0.96 |
| **+ pseudo + 2-backbone rank ensemble + TTA** | **0.960–0.975 ✅** |

Log every training session's scores into `benchmarks/results.csv` (schema documented in the guide, §6).
