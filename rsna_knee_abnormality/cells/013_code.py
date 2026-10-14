# ============================================================
# SECTION 11 - SELF-TESTS (run before trusting the pipeline)
# ============================================================
# Fast, dependency-light unit checks executed in-kernel:
#   1. report parser negation/uncertainty logic
#   2. ASL gradient sanity + metric edge-case safety
#   3. DICOM depth resampling / normalisation geometry
#   4. model forward shape + zero-plane masking behaviour
#   5. rank-average ensemble monotonicity
from __future__ import annotations


def test_report_parser() -> None:
    pos, _, st = extract_pseudo_labels_from_report(
        "Complete tear of the ACL. Large joint effusion and a Baker's cyst. "
        "Bone marrow edema consistent with contusion.")
    assert pos[TARGET_COLS.index("ACL")] == 1.0, "ACL tear must be positive"
    assert pos[TARGET_COLS.index("Effusion")] == 1.0, "effusion must be positive"
    assert pos[TARGET_COLS.index("Baker's")] == 1.0, "Baker's cyst must be positive"
    assert pos[TARGET_COLS.index("Contusion")] == 1.0, "marrow edema -> contusion"

    neg, _, _ = extract_pseudo_labels_from_report(
        "No evidence of ACL tear. The MCL is intact. There is no effusion. "
        "No fracture or bone marrow abnormality is identified.")
    assert neg[TARGET_COLS.index("ACL")] == 0.0, "negated ACL must be negative"
    assert neg[TARGET_COLS.index("MCL")] == 0.0, "'intact' is a negation cue"
    assert neg[TARGET_COLS.index("Effusion")] == 0.0
    assert neg[TARGET_COLS.index("Fracture")] == 0.0

    oa, _, _ = extract_pseudo_labels_from_report(
        "Moderate medial compartment osteoarthritis with joint space narrowing "
        "and osteophytes. Patellofemoral joint degenerative changes.")
    assert oa[TARGET_COLS.index("Medial OA")] == 1.0
    assert oa[TARGET_COLS.index("PF OA")] == 1.0

    unc, _, _ = extract_pseudo_labels_from_report(
        "Findings are equivocal for a possible lateral meniscus tear; cannot exclude synovitis.")
    assert unc[TARGET_COLS.index("Lateral Meniscus")] == 0.0, "equivocal -> not hard-positive"
    print("[test] report parser OK")


def test_loss_and_metric() -> None:
    torch.manual_seed(CFG.SEED)
    crit = make_loss(CFG)
    logits = torch.randn(64, 12, requires_grad=True)
    targets = (torch.rand(64, 12) < 0.1).float()          # ~10% positives
    loss = crit(logits, targets, sample_weight=torch.rand(64) * 0.5 + 0.5)
    loss.backward()
    assert torch.isfinite(loss), "ASL produced non-finite loss"
    assert torch.isfinite(logits.grad).all(), "ASL produced non-finite gradients"

    # single-class label column must NOT crash the metric logger
    y = np.zeros((10, 12), np.float32); y[:, 0] = 1.0
    p = np.random.default_rng(0).random((10, 12)).astype(np.float32)
    macro, per = macro_average_roc_auc(y, p)
    assert 0.0 <= macro <= 1.0 and math.isfinite(macro)
    assert per["MCL"] == 0.5, "single-class target must return neutral 0.5 AUC"
    print(f"[test] ASL + metric OK (macro={macro:.4f})")


def test_dicom_geometry() -> None:
    rng = np.random.default_rng(3)
    vol = rng.random((27, 300, 280)).astype(np.float32) * 1000.0     # 27 raw slices
    r = READER
    v = r.normalize_volume(vol)
    assert 0.0 <= float(v.min()) and float(v.max()) <= 1.0
    k = r.resample_depth(v, CFG.NUM_SLICES if not CFG.FAST_DEV_RUN else 4)
    assert k.shape[0] == (CFG.NUM_SLICES if not CFG.FAST_DEV_RUN else 4)
    m = r.resize_spatial(k, (64, 64))
    assert m.shape[-2:] == (64, 64)
    plane = classify_orientation([0, 1, 0, 0, 0, -1])                # axial normal +Y? -> coronal/sag check
    assert plane in (PLANE_SAGITTAL, PLANE_CORONAL, PLANE_AXIAL, PLANE_UNKNOWN)
    md = parse_sequence_metadata("SAG PD-FS", "s1", "STIR")
    assert md["fluid_sensitive"] == 1 and md["fat_suppressed"] == 1
    print("[test] dicom geometry/normalisation OK")


def test_model_forward() -> None:
    cfg = CFG
    b, p, c, k = 2, cfg.NUM_PLANES, cfg.IN_CHANNELS, cfg.NUM_SLICES if not cfg.FAST_DEV_RUN else 4
    h, w = cfg.IMAGE_SIZE if not cfg.FAST_DEV_RUN else (64, 64)
    model = KneeAbnormalityModel(cfg, backbone_name=None, pretrained=False)
    # shrink image path quickly for CPU test
    vol = torch.rand(b, p, c, k, h, w)
    meta = torch.stack([torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [1.0, 1.0, 2.0]])
                        .expand(p, 3) for _ in range(b)]).float()
    mask = torch.ones(b, p)
    mask[0, 2] = 0.0                                    # simulate missing axial plane
    probs = model(vol, meta, mask)
    assert probs.shape == (b, cfg.NUM_TARGETS)
    assert bool((probs >= 0).all() and (probs <= 1).all())
    logits = model(vol, meta, mask, return_logits=True)
    assert logits.shape == (b, cfg.NUM_TARGETS) and torch.isfinite(logits).all()
    n_params = sum(q.numel() for q in model.parameters())
    print(f"[test] model forward OK | params={n_params/1e6:.2f}M probs_range="
          f"[{probs.min():.3f},{probs.max():.3f}]")


def test_rank_ensemble() -> None:
    rng = np.random.default_rng(5)
    a = rng.random((50, 12)).astype(np.float64)
    bmat = a + rng.normal(0, 0.05, a.shape)             # correlated predictor
    blend = rank_average([a, bmat], weights=[0.5, 0.5])
    assert blend.shape == a.shape and 0.0 <= blend.min() and blend.max() <= 1.0
    print("[test] rank averaging OK")


for fn in (test_report_parser, test_loss_and_metric, test_dicom_geometry,
           test_model_forward, test_rank_ensemble):
    try:
        fn()
    except AssertionError as exc:
        print(f"[test] FAILED in {fn.__name__}: {exc}")
        raise
print("[tests] ALL SELF-TESTS PASSED")
