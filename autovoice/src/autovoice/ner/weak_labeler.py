"""Weak (silver) labeling of complaints by dictionary/regex matching.

**Disclosure:** these labels are heuristic, not human-verified. They are used
ONLY for training; dev/test evaluation uses the gold set exclusively
(see data/labeled/ and docs/model_card.md).

Lexicons:
    COMPONENT - derived from the NHTSA `component` field values present in the
                cleaned data plus a hand-curated list of common part names.
    SYMPTOM   - curated failure-symptom phrases ("stalled", "grinding noise",
                "warning light on", "failed to start", "leaking" ...).

Matching is plain regex with word boundaries (no spaCy dependency - we only
claim tools we actually use).  Overlapping matches resolve longest-first so
"brake caliper" wins over "caliper".
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from autovoice.config import DATA_PROCESSED_DIR
from autovoice.logging_conf import get_logger

log = get_logger(__name__)

#: Hand-curated symptom phrases (lowercase; matched case-insensitively).
SYMPTOM_TERMS: list[str] = [
    "stall(ed|s|ing)?",
    "won'?t start",
    "wouldn'?t start",
    "fails? to start",
    "failed to start",
    "no start",
    "hesitat(ed|ion|es|ing)?",
    "surging",
    "misfir(e|es|ing|ed)",
    "shut(ting)? off",
    "died while driving",
    "engine (cut|stopped|shut)",
    "overheat(s|ing|ed)?",
    "leak(ing|s|ed)?",
    "grind(ing|s|ed)? noise",
    "grinding",
    "vibrat(e|ion|es|ing|ed)",
    "shakin(g)?",
    "pulsat(ing|ion|es|ed)",
    "sudden(ly)? (acceleration|braking|loss)",
    "loss of (power|control|brakes|steering)",
    "warning light",
    "(dashboard|dash) light",
    "check engine light",
    "smell(ing)? (of )?(burnt|burning|smoke|fuel|gas)",
    "smoke( coming)?",
    "fire",
    "catch(ing)? fire",
    "sparks?",
    "short(ing)? out",
    "battery (dying|dead|not charging)",
    "not charging",
    "won'?t charge",
    "lock(ed|ing)? (up|out)",
    "drifting",
    "pull(ing)? (to|left|right)",
    "swerv(e|ing|ed)",
    "collision",
    "crash(es|ed|ing)?",
    "accident",
    "injur(y|ied|ies)",
    "recall(ed)? software",
    "unintended (acceleration|movement)",
    "failure",
    "defect(ive|s)?",
    "malfunction(s|ing|ed)?",
]

#: Hand-curated component/part terms (beyond those harvested from the data).
COMPONENT_TERMS: list[str] = [
    "brakes?",
    "brake (caliper|pad|rotor|line|hose|master cylinder|booster)",
    "abs (module|pump|system)",
    "engine",
    "transmission",
    "cv?t",
    "turbo",
    "turbocharger",
    "alternator",
    "starter",
    "starter motor",
    "battery",
    "12[- ]volt battery",
    "fuse(s| box)?",
    "wiring (harness|loom)?",
    "sensor",
    "fuel (pump|injector|tank|line|rail)",
    "radiator",
    "coolant",
    "water pump",
    "thermostat",
    "timing (chain|belt)",
    "serpentine belt",
    "steering (column|rack|wheel|pump)",
    "suspension",
    "shocks?",
    "shock absorber",
    "strut(s)?",
    "wheel bearing",
    "wheel bearings",
    "tires?",
    "axle",
    "driveshaft",
    "airbag",
    "seat belt",
    "door latch",
    "door lock",
    "door handle",
    "window regulator",
    "window switch",
    "mirror",
    "wipers?",
    "wiper motor",
    "headlight",
    "taillight",
    "turn signal",
    "infotainment",
    "touchscreen",
    "software",
    "ecm",
    "pcm",
    "bcm",
    "camera",
    "radar sensor",
    "cruise control",
    "seat",
    "exhaust",
    "catalytic converter",
    "gasket",
    "oil (leak|pump|sending unit)",
    "hood latch",
    "parking brake",
]


def _compile(terms: list[str], entity: str) -> list[re.Pattern[str]]:
    return [re.compile(rf"\b(?:{t})\b", re.I) for t in terms]


_SYMPTOM_PATTERNS = _compile(SYMPTOM_TERMS, "SYMPTOM")
_COMPONENT_PATTERNS = _compile(COMPONENT_TERMS, "COMPONENT")

_PATTERN_INDEX = {
    "SYMPTOM": _SYMPTOM_PATTERNS,
    "COMPONENT": _COMPONENT_PATTERNS,
}


def component_lexicon_from_data(parquet_path: Path | None = None) -> list[str]:
    """Harvest component vocabulary from the NHTSA `component` column."""
    import pandas as pd

    path = parquet_path or DATA_PROCESSED_DIR / "complaints.parquet"
    df = pd.read_parquet(path, columns=["component"])
    parts: set[str] = set()
    for value in df["component"].dropna().unique():
        for chunk in re.split(r"[:;]", str(value)):
            chunk = chunk.strip().lower()
            if len(chunk) > 3 and not chunk.isdigit():
                parts.add(re.escape(chunk))
    log.info("weak_label.lexicon_harvested", n=len(parts))
    return sorted(parts)


def tag_text(text: str, extra_component_terms: list[str] | None = None) -> list[tuple[int, int, str]]:
    """Return non-overlapping (start, end_excl, BIO_TAG) char spans for ``text``.

    Longest match wins; overlapping candidates are dropped.
    """
    candidates: list[tuple[int, int, str]] = []
    groups = dict(_PATTERN_INDEX)
    if extra_component_terms:
        groups["COMPONENT"] = _compile(extra_component_terms, "COMPONENT") + _COMPONENT_PATTERNS
    for etype, patterns in groups.items():
        for pat in patterns:
            for m in pat.finditer(text):
                candidates.append((m.start(), m.end(), f"B-{etype}"))
    # longest-first greedy selection
    candidates.sort(key=lambda c: (-(c[1] - c[0]), c[0]))
    chosen: list[tuple[int, int, str]] = []
    taken: list[tuple[int, int]] = []
    for s, e, tag in candidates:
        if any(not (e <= ts or s >= te) for ts, te in taken):
            continue
        chosen.append((s, e, tag))
        taken.append((s, e))
    chosen.sort()
    return chosen


def weak_label_records(
    texts: list[str], extra_component_terms: list[str] | None = None
) -> list[dict]:
    """Produce silver-labeled JSONL-ready dicts: {text, spans:[{start,end,label}]}."""
    out = []
    for t in texts:
        spans = tag_text(t, extra_component_terms)
        out.append(
            {
                "text": t,
                "spans": [{"start": s, "end": e, "label": lab} for s, e, lab in spans],
            }
        )
    return out


def write_silver_jsonl(out_path: Path, parquet_path: Path | None = None) -> int:
    """Weak-label all processed narratives and write silver JSONL. Returns count."""
    import pandas as pd

    path = parquet_path or DATA_PROCESSED_DIR / "complaints.parquet"
    df = pd.read_parquet(path, columns=["narrative"])
    lex = component_lexicon_from_data(path)
    records = weak_label_records(df["narrative"].tolist(), extra_component_terms=lex)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    labeled = sum(1 for r in records if r["spans"])
    log.info("weak_label.written", file=str(out_path), rows=len(records), with_spans=labeled)
    return len(records)


if __name__ == "__main__":
    from autovoice.config import DATA_LABELED_DIR

    n = write_silver_jsonl(DATA_LABELED_DIR / "silver_ner.jsonl")
    print(f"wrote {n} silver records")
