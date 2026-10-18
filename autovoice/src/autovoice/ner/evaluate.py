"""Evaluate NER on the GOLD test set only (never on weak labels).

Produces:
    reports/ner_metrics.json  - seqeval entity-level P/R/F1 per type + micro/macro,
                                plus the identical scores for the rule/dictionary
                                BASELINE so the model's gain is measured honestly.
    reports/ner_errors.md     - top false-positive / false-negative spans.

If only a *provisional* gold file exists (silver sampled via
``gold.py --allow-provisional``), every report is flagged
``"gold_provenance": "provisional-silver"`` so no claim of human verification
is made.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from autovoice.config import REPORTS_DIR
from autovoice.logging_conf import get_logger
from autovoice.ner.gold import GOLD_TEST_PATH, GOLD_TEST_PROVISIONAL_PATH
from autovoice.ner.schema import bio_labels
from autovoice.ner.train import build_dataset_records
from autovoice.ner.weak_labeler import tag_text

log = get_logger(__name__)


def _seqeval_scores(true_tags: list[list[str]], pred_tags: list[list[str]]) -> dict:
    from seqeval.metrics import classification_report

    rep = classification_report(true_tags, pred_tags, output_dict=True, zero_division=0)
    out = {"per_entity": {k: v for k, v in rep.items() if k in ("COMPONENT", "SYMPTOM")},
           "micro_f1": rep.get("micro avg", {}).get("f1-score", 0.0),
           "macro_f1": rep.get("macro avg", {}).get("f1-score", 0.0),
           "weighted_f1": rep.get("weighted avg", {}).get("f1-score", 0.0)}
    return out


def baseline_predict(records: list[dict]) -> list[list[str]]:
    """Pure rule/dictionary baseline: re-tag each text with the weak labeler."""
    from autovoice.ner.schema import spans_to_bio

    preds = []
    for rec in records:
        text = " ".join(rec["words"])
        spans = tag_text(text)
        # map char offsets back onto whitespace words
        preds.append(spans_to_bio(rec["words"], spans, bio_labels()))
    return preds


def model_predict(model_dir: str, records: list[dict], max_seq_len: int = 128) -> list[list[str]]:
    """Run the fine-tuned HF token classifier; returns per-word BIO tags."""
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    from autovoice.ner.infer import decode_word_tags

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    model.eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    preds = []
    with torch.no_grad():
        for rec in records:
            enc = tokenizer(rec["words"], truncation=True, is_split_into_words=True,
                            return_tensors="pt", max_length=max_seq_len)
            logits = model(**enc).logits[0]
            tag_ids = logits.argmax(dim=-1).tolist()
            word_ids = enc.word_ids()
            tags = decode_word_tags(tag_ids, word_ids, id2label, n_words=len(rec["words"]))
            preds.append(tags)
    return preds


def error_analysis(true_tags: list[list[str]], pred_tags: list[list[str]],
                   records: list[dict], top_n: int = 15) -> str:
    """Markdown confusion analysis: most common FP/FN surface forms."""
    from autovoice.ner.schema import bio_to_spans

    fp, fn = Counter(), Counter()
    for rec, tr, pr in zip(records, true_tags, pred_tags, strict=True):
        def span_set(tags):
            s = set()
            for ws, we, etype, _n in bio_to_spans(rec["words"], tags):
                s.add((" ".join(rec["words"][ws:we]), etype))
            return s
        ts, ps = span_set(tr), span_set(pr)
        for surf, etype in ps - ts:
            fp[(surf, etype)] += 1
        for surf, etype in ts - ps:
            fn[(surf, etype)] += 1
    lines = ["# NER error analysis (gold test)", "",
             "## Top false positives (model tagged, gold did not)", ""]
    lines += [f"- `{surf}` ({etype}) x{c}" for (surf, etype), c in fp.most_common(top_n)] or ["- none"]
    lines += ["", "## Top false negatives (gold had, model missed)", ""]
    lines += [f"- `{surf}` ({etype}) x{c}" for (surf, etype), c in fn.most_common(top_n)] or ["- none"]
    return "\n".join(lines) + "\n"


def evaluate(model_dir: str | None = None, test_path: Path | None = None,
             out_json: Path | None = None, out_md: Path | None = None) -> dict:
    """Full evaluation; returns the metrics dict also written to disk."""
    provisional = False
    if test_path is None:
        if GOLD_TEST_PATH.exists():
            test_path = GOLD_TEST_PATH
        elif GOLD_TEST_PROVISIONAL_PATH.exists():
            test_path, provisional = GOLD_TEST_PROVISIONAL_PATH, True
        else:
            raise SystemExit(
                "No gold test file. Run `python -m autovoice.ner.gold export/import/split` "
                "(human loop) or `... split --allow-provisional` for a provisional run."
            )
    records = build_dataset_records(test_path)
    true_tags = [r["tags"] for r in records]
    log.info("ner.evaluate_start", n=len(records), path=str(test_path), provisional=provisional)

    base_pred = baseline_predict(records)
    baseline_metrics = _seqeval_scores(true_tags, base_pred)

    result: dict = {
        "gold_test_file": str(test_path),
        "gold_test_size": len(records),
        "gold_provenance": "provisional-silver (NOT human-verified)" if provisional else "human-corrected gold",
        "evaluation_policy": "metrics computed on gold ONLY; silver used for training only",
        "baseline_rules": baseline_metrics,
    }
    if model_dir:
        try:
            pred = model_predict(model_dir, records)
            result["model"] = _seqeval_scores(true_tags, pred)
            result["delta_micro_f1_vs_baseline"] = round(result["model"]["micro_f1"] - baseline_metrics["micro_f1"], 4)
            (out_md or REPORTS_DIR / "ner_errors.md").parent.mkdir(parents=True, exist_ok=True)
            (out_md or REPORTS_DIR / "ner_errors.md").write_text(
                error_analysis(true_tags, pred, records), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001 - report, don't crash pipelines
            log.error("ner.model_eval_failed", error=str(exc))
            result["model"] = {"error": f"not evaluated: {exc}"}
    else:
        result["model"] = {"error": "no --model-dir provided; baseline-only run"}

    out_json = out_json or REPORTS_DIR / "ner_metrics.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    log.info("ner.metrics_written", path=str(out_json))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--test-file", type=Path, default=None)
    args = parser.parse_args()
    res = evaluate(args.model_dir, args.test_file)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
