"""Gold-set tooling: Label Studio export/import + deterministic split.

**Weak-label disclosure:** silver labels (``data/labeled/silver_ner.jsonl``)
are rule-derived and are used ONLY for training.  Evaluation happens strictly
on the human-corrected gold set (``data/labeled/gold_test.jsonl``).

Workflow
--------
1. ``python -m autovoice.ner.gold export``  -> writes a Label Studio task file
   with pre-annotations so a human only has to *correct* spans.
2. Human corrects in Label Studio (config JSON printed by the command, also in
   ``docs/model_card.md``), exports annotations as JSON.
3. ``python -m autovoice.ner.gold import --annotations <file.json>`` converts
   the export into ``gold_raw.jsonl`` (char-offset BIO spans).
4. ``python -m autovoice.ner.gold split`` carves the corrected set into a
   held-out test split (>=50 examples committed) and a dev split; the rest
   stays silver-only for training.

To keep the project runnable end-to-end before manual correction happens,
``split`` falls back to a *seeded sample of silver data marked clearly as
provisional* **only** when ``--allow-provisional`` is passed; metrics computed
that way are labeled "provisional" in every report.  The README/model card
must state the true gold-set size from ``reports/ner_metrics.json``.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from autovoice.config import DATA_LABELED_DIR
from autovoice.logging_conf import get_logger

log = get_logger(__name__)

SILVER_PATH = DATA_LABELED_DIR / "silver_ner.jsonl"
GOLD_RAW_PATH = DATA_LABELED_DIR / "gold_raw.jsonl"
GOLD_TEST_PATH = DATA_LABELED_DIR / "gold_test.jsonl"
GOLD_TEST_PROVISIONAL_PATH = DATA_LABELED_DIR / "gold_test_provisional.jsonl"
GOLD_DEV_PATH = DATA_LABELED_DIR / "gold_dev.jsonl"
LS_TASKS_PATH = DATA_LABELED_DIR / "label_studio_tasks.json"

#: Label Studio NER labeling interface (HTML) for step 2 above.
LABEL_STUDIO_CONFIG = """<View>
  <Labels name="label" toName="text">
    <Label value="COMPONENT" background="#2e86de"/>
    <Label value="SYMPTOM" background="#ee5253"/>
  </Labels>
  <Text name="text" value="$text"/>
</View>"""

MIN_GOLD_TEST = 50


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def export_label_studio_tasks(silver_path: Path = SILVER_PATH, out_path: Path = LS_TASKS_PATH,
                              limit: int = 300, seed: int = 42) -> int:
    """Sample silver records and write Label Studio tasks with pre-annotations."""
    records = _read_jsonl(silver_path)
    rng = random.Random(seed)
    # prefer records that actually contain spans so corrections are meaningful
    with_spans = [r for r in records if r["spans"]]
    without = [r for r in records if not r["spans"]]
    rng.shuffle(with_spans)
    rng.shuffle(without)
    sample = (with_spans + without)[:limit]
    tasks = []
    for i, rec in enumerate(sample):
        tasks.append({
            "data": {"text": rec["text"]},
            "predictions": [{
                "result": [
                    {"value": {"start": s["start"], "end": s["end"],
                               "text": rec["text"][s["start"]:s["end"]],
                               "labels": [s["label"].removeprefix("B-")]},
                     "from_name": "label", "to_name": "text", "type": "labels"}
                    for s in rec["spans"]
                ],
                "model_version": "weak_labeler-rules-v1",
            }],
            "meta": {"task_id": i},
        })
    out_path.write_text(json.dumps(tasks, indent=1), encoding="utf-8")
    log.info("gold.label_studio_exported", n=len(tasks), path=str(out_path))
    return len(tasks)


def import_annotations(annotations_path: Path, out_path: Path = GOLD_RAW_PATH) -> int:
    """Convert a Label Studio JSON export into gold BIO-span JSONL.

    Accepts either a JSON array of tasks or JSONL; human answers live under
    ``annotations[0].result``. Falls back to predictions when an annotation is
    missing (never silently - logged as unreviewed).
    """
    raw = annotations_path.read_text(encoding="utf-8")
    tasks = json.loads(raw) if raw.lstrip().startswith("[") else [json.loads(l) for l in raw.splitlines() if l.strip()]
    records, unreviewed = [], 0
    for task in tasks:
        text = task.get("data", {}).get("text") or task.get("text")
        if not text:
            continue
        result = None
        if task.get("annotations"):
            result = task["annotations"][0].get("result", [])
        elif task.get("predictions"):
            result = task["predictions"][0].get("result", [])
            unreviewed += 1
        spans = []
        for item in result or []:
            val = item.get("value", {})
            labels = val.get("labels", [])
            if not labels or "start" not in val or "end" not in val:
                continue
            spans.append({"start": val["start"], "end": val["end"], "label": f"B-{labels[0]}"})
        spans.sort(key=lambda s: s["start"])
        records.append({"text": text, "spans": spans,
                        "reviewed": bool(task.get("annotations"))})
    _write_jsonl(out_path, records)
    log.info("gold.annotations_imported", n=len(records), unreviewed=unreviewed, path=str(out_path))
    return len(records)


def split_gold(test_size: int = MIN_GOLD_TEST, seed: int = 42,
               allow_provisional: bool = False) -> dict[str, int]:
    """Deterministically split the corrected gold set into dev/test files."""
    src = GOLD_RAW_PATH if GOLD_RAW_PATH.exists() else None
    provisional = False
    if src is None:
        if not allow_provisional:
            raise SystemExit(
                "No gold_raw.jsonl found. Complete the Label Studio correction loop first "
                "(see module docstring), or pass --allow-provisional to sample silver data "
                "explicitly marked provisional."
            )
        src, provisional = SILVER_PATH, True
    records = _read_jsonl(src)
    rng = random.Random(seed)
    rng.shuffle(records)
    if len(records) < test_size:
        test_size = max(10, len(records) // 3)
        log.warning("gold.small_set_shrinking_test", n_records=len(records), test_size=test_size)
    test, dev, rest = records[:test_size], records[test_size:test_size * 2], records[test_size * 2:]
    tag = "_provisional" if provisional else ""
    _write_jsonl(GOLD_TEST_PATH.with_name(f"{GOLD_TEST_PATH.stem}{tag}.jsonl"), test)
    _write_jsonl(GOLD_DEV_PATH.with_name(f"{GOLD_DEV_PATH.stem}{tag}.jsonl"), dev)
    train_path = DATA_LABELED_DIR / f"gold_train{tag}.jsonl"
    _write_jsonl(train_path, rest)
    counts = {"test": len(test), "dev": len(dev), "train": len(rest), "provisional": int(provisional)}
    log.info("gold.split", **counts)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_exp = sub.add_parser("export", help="write Label Studio tasks from silver data")
    p_exp.add_argument("--limit", type=int, default=300)
    p_imp = sub.add_parser("import", help="convert Label Studio export to gold JSONL")
    p_imp.add_argument("--annotations", type=Path, required=True)
    p_split = sub.add_parser("split", help="deterministic dev/test split of gold set")
    p_split.add_argument("--test-size", type=int, default=MIN_GOLD_TEST)
    p_split.add_argument("--allow-provisional", action="store_true")
    sub.add_parser("config", help="print the Label Studio labeling interface XML")
    args = parser.parse_args()
    if args.cmd == "export":
        export_label_studio_tasks(limit=args.limit)
        print(f"Wrote {LS_TASKS_PATH}. Import into Label Studio with this config:\n{LABEL_STUDIO_CONFIG}")
    elif args.cmd == "import":
        import_annotations(args.annotations)
    elif args.cmd == "split":
        print(split_gold(args.test_size, allow_provisional=args.allow_provisional))
    else:
        print(LABEL_STUDIO_CONFIG)


if __name__ == "__main__":
    main()
