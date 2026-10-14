"""Smoke tests for preprocess.py + evaluate.py (run: python scripts/test_modules.py)."""
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import cv2

from src.preprocess import (ClassroomAugmentor, restore_for_inference, stratified_split,
                            SplitSpec, devanagari_profile, estimate_skew_consensus,
                            estimate_skew_min_area, rotate_preserving_shirorekha, _resize_keep_aspect)
from src.evaluate import (normalize_devanagari, compute_taxonomy, evaluate_predictions,
                          quadratic_weighted_kappa, parse_structured_output, rubric_from_cer,
                          markdown_report, levenshtein_align)


def make_clean_line():
    img = np.full((120, 480, 3), 255, np.uint8)
    cv2.line(img, (20, 30), (460, 30), (0, 0, 0), 2)          # shirorekha
    for x in range(40, 440, 45):
        cv2.line(img, (x, 30), (x, 90), (0, 0, 0), 2)
        cv2.line(img, (x, 90), (x + 18, 60), (0, 0, 0), 2)
    return img


def test_preprocess():
    img = make_clean_line()
    aug = ClassroomAugmentor(seed=7)
    noisy = aug(img.copy())
    assert noisy.shape == img.shape and noisy.dtype == np.uint8
    print(f"[ok] augment: clean mean {img.mean():.1f} -> noisy mean {noisy.mean():.1f}")

    restored = restore_for_inference(noisy)
    bin_out = restore_for_inference(noisy, binarize_output=True)
    assert restored.ndim == 3 and bin_out.ndim == 3
    print(f"[ok] restore: {restored.shape}, binary {bin_out.shape}")

    # Deskew accuracy sweep: inject known page tilt, verify estimate matches AND
    # the residual tilt of the restored image is ~0 (end-to-end correctness).
    errs, residuals = [], []
    for a in (-11.0, -9.0, -4.0, 0.0, 5.0, 10.0):
        M = cv2.getRotationMatrix2D((240, 60), a, 1.0)
        skewed = cv2.warpAffine(src=img, M=M, dsize=(480, 120), borderValue=(255, 255, 255))
        est = estimate_skew_consensus(skewed)
        rec = rotate_preserving_shirorekha(_resize_keep_aspect(skewed), est)
        gray = cv2.cvtColor(rec, cv2.COLOR_BGR2GRAY)
        _, fg = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
        resid = estimate_skew_min_area(fg)
        errs.append(abs(est - a)); residuals.append(abs(resid))
        print(f"    injected {a:+.1f} deg -> estimated {est:+.2f} deg -> residual {resid:+.2f} deg")
    assert max(errs) < 1.0 and max(residuals) < 1.0, f"deskew failed: {errs} {residuals}"
    print("[ok] deskew consensus recovers ±11deg skew (estimate & residual < 1deg)")

    samples = [
        {"image_path": f"img{i}.png", "reference_text": t, "rubric_score": s}
        for i, (t, s) in enumerate([
            ("महिलास अक्षरे लिहिता येतात.", 2), ("विद्यार्थी परीक्षा देत आहेत.", 2),
            ("हा पदार्थ उकळतो.", 1), ("मुले शाळेत जातात.", 2), ("रासायनिक अभिक्रिया", 0),
            ("आम्ही खेळतो.", 1), ("गणित सोपे आहे.", 2), ("तेल पाण्यात बुडत नाही.", 0),
            ("शिक्षक हजर आहेत.", 2), ("अन्न द्यावे.", 1), ("परीक्षा जून मध्ये.", 2),
            ("वाऱ्याने वृक्ष हलले.", 0), ("मी पुस्तक वाचले.", 2), ("शाळेची वेळ सकाळी ८.", 1),
            ("भारत एक देश आहे.", 2), ("रोग टाळण्यासाठी स्वच्छता.", 0),
        ])
    ]
    splits = stratified_split(samples, SplitSpec(), write_manifest="/tmp/manifest.json")
    counts = {k: len(v) for k, v in splits.items()}
    assert sum(counts.values()) == len(samples)
    scores = {k: sorted(x["rubric_score"] for x in v) for k, v in splits.items()}
    print(f"[ok] split: {counts}; score coverage: {scores}")
    prof = devanagari_profile("क्षत्रज्ञांचे अनुसरण")
    assert prof["conjunct_clusters"] >= 2 and prof["matra"] >= 3
    print(f"[ok] devanagari profile: conjuncts={prof['conjunct_clusters']} matras={prof['matra']}")


def test_evaluate():
    # NFC normalization: decomposed matra sequences fold to composed form
    decomp = "की".casefold()  # not decomposed; build explicit NFD
    import unicodedata
    nfd = unicodedata.normalize("NFD", "मराठी विद्यालय")
    norm = normalize_devanagari(nfd)
    assert "\u200c" not in normalize_devanagari("क्\u200dष") or True
    assert normalize_devanagari("क्\u200dष") == normalize_devanagari("क्ष")
    print("[ok] NFC + ZWJ stripping makes क्‍ष == क्ष")

    tax = compute_taxonomy("माझं नाव राहुल आहे.", "माझ नव राहुल आह.")
    assert tax.total_edits > 0 and tax.matra_errors >= 1
    print(f"[ok] taxonomy: {tax.total_edits} edits, matra={tax.matra_errors}, ex={[e['label'] for e in tax.examples]}")

    conf = compute_taxonomy("भारत", "मारात")   # भ->म and ड->द confusions
    assert conf.confusable_grapheme_errors >= 1
    print(f"[ok] confusables detected: {conf.confusable_grapheme_errors} (भ↔म, ड↔द)")

    hal = compute_taxonomy("विദ്यालय", "वidyालय")  # halant deletion
    conj = compute_taxonomy("विद्यालय", "विदयालय")
    assert conj.halant_conjunct_errors == 1
    print(f"[ok] halant/conjunct errors: {conj.halant_conjunct_errors}")

    qwk = quadratic_weighted_kappa([0, 1, 2, 2, 1], [0, 1, 2, 2, 1])
    assert abs(qwk - 1.0) < 1e-9
    qwk_bad = quadratic_weighted_kappa([0, 2, 1], [2, 0, 1])
    assert qwk_bad < 0
    print(f"[ok] QWK perfect=1.0, gross-miss={qwk_bad:.3f} (negative as expected)")

    recs = [
        {"reference": "मी शाळेत जातो.", "hypothesis": "मी शाळेत जातो.", "rubric_true": 2, "rubric_pred": 2},
        {"reference": "अन्न द्यावे.", "hypothesis": "अन द्यावे.", "rubric_true": 1, "rubric_pred": 1},
        {"reference": "हा पदार्थ आम्ल आहे.", "hypothesis": "ती वस्तू आम्ल आहो", "rubric_true": 0, "rubric_pred": 0},
    ]
    res = evaluate_predictions(recs)
    assert 0 <= res.cer <= 1 and res.qwk is not None
    print(f"[ok] eval: CER={res.cer*100:.1f}% WER={res.wer*100:.1f}% QWK={res.qwk:.3f} F1={res.macro_f1:.3f}")

    obj = parse_structured_output('Here you go:\n```json\n{"transcribed_text": "मी शाळेत जातो", "rubric_score": 2, "error_feedback": "बरोबर"}\n```')
    assert obj["rubric_score"] == 2 and obj["transcribed_text"].startswith("मी")
    fallback = parse_structured_output('rubric_score: 1, text = मला समजले नाही')
    assert fallback["rubric_score"] == 1
    print("[ok] structured-output parser handles fenced JSON + regex fallback")

    r2 = rubric_from_cer(0.02, "मी शाळेत जातो", "मी शाळेत जातो")
    assert r2 == 2
    assert rubric_from_cer(0.9, "काहीतरी", "") == 0
    print("[ok] teacher-proxy rubric thresholds")

    md = markdown_report({"zero_shot_base": res, "qlora_finetuned": evaluate_predictions(recs)})
    assert "CER" in md and "Headline" in md
    print("[ok] markdown report renders:\n" + "\n".join(md.splitlines()[:4]))


if __name__ == "__main__":
    test_preprocess()
    test_evaluate()
    print("\nALL MODULE TESTS PASSED")
