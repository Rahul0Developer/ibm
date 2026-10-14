# ============================================================
# SECTION 9 - DATA LOADING, PSEUDO-LABEL EXPANSION & CV DRIVER
# ============================================================
# Loads train/test tables (real Kaggle mount OR synthetic demo), mines reports for
# weak supervision, runs the multi-backbone Stratified Group K-Fold training and
# finally produces submission.csv. Time-boxed to respect the 9h Kaggle wall.
from __future__ import annotations


def load_tables(cfg: Config = CFG) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Read competition CSVs; synthesise a small demo corpus if nothing is mounted."""
    train_df: Optional[pd.DataFrame] = None
    test_df: Optional[pd.DataFrame] = None
    if LAYOUT.train_csv is not None:
        train_df = pd.read_csv(LAYOUT.train_csv)
        print(f"[data] train.csv rows={len(train_df)} cols={list(train_df.columns)[:16]}")
    if LAYOUT.test_csv is not None:
        test_df = pd.read_csv(LAYOUT.test_csv)
        print(f"[data] test.csv rows={len(test_df)}")
    # Attach reports if shipped separately (parquet/json/csv side-car).
    if train_df is not None and LAYOUT.reports_parquet is not None and "Report" not in train_df.columns:
        try:
            rep = pd.read_parquet(LAYOUT.reports_parquet) if str(LAYOUT.reports_parquet).endswith(".parquet") \
                else pd.read_csv(LAYOUT.reports_parquet)
            key = cfg.GROUP_COL if cfg.GROUP_COL in rep.columns else rep.columns[0]
            train_df = train_df.merge(rep, how="left", left_on=cfg.GROUP_COL, right_on=key)
            print(f"[data] merged {len(rep)} reports")
        except Exception as exc:
            print(f"[data] report merge skipped: {exc}")

    if train_df is None or test_df is None:
        print("[data] no competition mount detected -> SYNTHETIC DEMO MODE "
              "(full pipeline executes on CPU so every cell is verifiably runnable)")
        s_train, s_test = make_synthetic_records(cfg, n_train=24, n_test=8)
        train_df = s_train if train_df is None else train_df
        test_df = s_test if test_df is None else test_df
        cfg.FAST_DEV_RUN = True
        cfg.EPOCHS = 1
        cfg.IMAGE_SIZE = (96, 96)               # keep the smoke run fast on CPU
        cfg.NUM_SLICES = 4
    return train_df, test_df


TRAIN_DF, TEST_DF = load_tables(CFG)

# ------------------------------------------------------------------ expansion
if SUBMISSION_MODE:
    print("[mode] SUBMISSION inference mode -> skipping label mining/training")
    TRAIN_ENRICHED = None
else:
    TRAIN_ENRICHED = build_labelled_frame(TRAIN_DF, CFG)
    n_gt = int((TRAIN_ENRICHED["label_source"] == "gt").sum())
    n_rep = int((TRAIN_ENRICHED["label_source"] == "report").sum())
    n_zero = int((TRAIN_ENRICHED["label_weight"] <= 0).sum())
    pos_rate = TRAIN_ENRICHED[[f"{t}_pseudo" for t in TARGET_COLS]].mean().to_dict()
    print(f"[weak-sup] human-GT studies={n_gt} | report-pseudo studies={n_rep} "
          f"| unusable={n_zero} | expansion factor={((n_gt + n_rep) / max(1, n_gt)):.1f}x")
    print("[weak-sup] pseudo positive rates:",
          json.dumps({k: round(float(v), 3) for k, v in pos_rate.items()}))

    # Keep only studies that contribute gradient signal.
    TRAIN_ENRICHED = TRAIN_ENRICHED[TRAIN_ENRICHED["label_weight"] > 0].reset_index(drop=True)
    if CFG.FAST_DEV_RUN:
        TRAIN_ENRICHED = TRAIN_ENRICHED.head(24).reset_index(drop=True)

print(f"[data] final train rows={0 if TRAIN_ENRICHED is None else len(TRAIN_ENRICHED)}, "
      f"test rows={len(TEST_DF)}")
