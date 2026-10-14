# ============================================================
# SECTION 10 - MAIN EXECUTION DRIVER (TRAIN OR SUBMIT)
# ============================================================
# Kaggle-safe control flow:
#   * if train.csv exists  -> mine pseudo-labels, run dual-backbone 5-fold CV,
#                             persist best fold checkpoints, then infer on test set.
#   * else (submission)    -> load offline checkpoints only and write submission.csv.
from __future__ import annotations


def main() -> None:
    t0 = time.time()
    print(f"[main] start | device={CFG.DEVICE} submission_mode={SUBMISSION_MODE}")

    if not SUBMISSION_MODE and TRAIN_ENRICHED is not None and len(TRAIN_ENRICHED) > 0:
        series_dirs = LAYOUT.train_dirs or ([LAYOUT.root] if LAYOUT.has_real_data else [])
        backbones = [CFG.BACKBONE]
        # Second architecture only when the clock allows it (dual-arch ensemble boost).
        per_arch_budget = CFG.time_left_for_training() / max(1, CFG.N_FOLDS) / 2.0
        if per_arch_budget > 45 * 60:            # >= ~45 min/fold available -> add alt arch
            backbones.append(CFG.BACKBONE_ALT)
        print(f"[main] training backbones={backbones} epochs={CFG.EPOCHS} "
              f"folds={CFG.N_FOLDS} budget/arch~{per_arch_budget/60:.1f}min")
        try:
            results = train_cross_backbone_cv(TRAIN_ENRICHED, series_dirs, backbones, CFG)
            with open(Path(CFG.ARTIFACT_DIR) / "cv_results.json", "w") as fh:
                json.dump({"folds": results["folds"]}, fh, indent=2, default=str)
            for entry in results["folds"]:
                print(f"[main] {entry['backbone']} mean AUC={entry['mean_auc']:.5f} "
                      f"folds={['%.4f' % a for a in entry['aucs']]}")
        except Exception as exc:                 # never let training errors kill inference
            print(f"[main] training stage failed: {exc!r} -> proceeding to inference "
                  f"with whatever checkpoints exist")

    # ------------------------------------------------------------------ submit
    sub = run_inference_pipeline(TEST_DF, CFG)
    display(sub.head()) if "display" in globals() else print(sub.head().to_string())
    print(f"[main] DONE in {(time.time()-t0)/60:.1f} min | submission.csv ready "
          f"({len(sub)} rows x {len(CFG.TARGET_COLS)} targets)")


main()
