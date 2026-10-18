"""Clean NHTSA complaint records into a compact analysis-ready parquet file.

Steps (each deterministic, logged with row counts before/after):
    1. keep only the columns AutoVoice needs;
    2. normalize encoding + whitespace in the narrative;
    3. drop rows with empty/boilerplate-only narratives;
    4. parse dates and numeric fields;
    5. dedupe on (make, model, year, narrative);
    6. filter to recent years and cap total rows (free-tier budget control).

Output: ``data/processed/complaints.parquet`` (+ a small quality report JSON).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

from autovoice.config import DATA_PROCESSED_DIR, DATA_RAW_DIR, REPORTS_DIR
from autovoice.data.download import RAW_FILENAME, load_raw
from autovoice.logging_conf import get_logger

log = get_logger(__name__)

KEEP_COLUMNS = [
    "cmplid",
    "make",
    "model",
    "model_year",
    "comp_desc",
    "complaint_desc",
    "crash",
    "fire",
    "injured",
    "deaths",
    "date_added",
]

#: Common boilerplate / noise stripped from narratives (light-touch only).
_BOILERPLATE_PATTERNS = [
    re.compile(r"\bthis (is|was) not a (recall|dealer) issue\b", re.I),
    re.compile(r"\bcustomer states? that\b", re.I),
    re.compile(r"\(have a nice day\)", re.I),
    re.compile(r"\bsee attached\b", re.I),
]

_MIN_NARRATIVE_CHARS = 40  # shorter texts carry almost no signal


def normalize_text(text: str) -> str:
    """Normalize whitespace/encoding artifacts in one narrative.

    Args:
        text: Raw narrative string.

    Returns:
        Cleaned string: unicode-normalized quotes, collapsed whitespace.
    """
    if not isinstance(text, str):
        return ""
    replacements = {
        "\u2019": "'",
        "\u2018": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\xa0": " ",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    for pat in _BOILERPLATE_PATTERNS:
        text = pat.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _to_bool_01(series: pd.Series) -> pd.Series:
    """Map NHTSA indicator values ('1'/'Y'/'True'...) to 0/1 ints."""
    truthy = {"1", "y", "yes", "true", "t"}
    return series.astype(str).str.strip().str.lower().isin(truthy).astype(int)


def clean(df: pd.DataFrame, years_back: int = 5, max_rows: int = 5000) -> pd.DataFrame:
    """Apply all cleaning steps to a raw complaints frame.

    Args:
        df: Raw dataframe from :mod:`autovoice.data.download`.
        years_back: Keep complaints added within the last N calendar years.
        max_rows: Hard cap on output rows (keeps BigQuery/Docker footprint free).

    Returns:
        Cleaned dataframe ready for parquet + BigQuery batch load.
    """
    out = df[[c for c in KEEP_COLUMNS if c in df.columns]].copy()
    n0 = len(out)

    out["narrative"] = out.pop("complaint_desc").map(normalize_text)
    out["component"] = out.pop("comp_desc").map(normalize_text)

    # indicators -> ints (kept ONLY as label sources; never fed to models)
    for col in ("crash", "fire", "injured", "deaths"):
        out[col] = _to_bool_01(out[col])

    out["date_added"] = pd.to_datetime(out["date_added"], errors="coerce")
    out["model_year"] = pd.to_numeric(out["model_year"], errors="coerce").astype("Int64")

    before = len(out)
    out = out[out["narrative"].str.len() >= _MIN_NARRATIVE_CHARS]
    log.info("clean.drop_short_narratives", removed=before - len(out))

    before = len(out)
    cutoff = pd.Timestamp.now().normalize() - pd.DateOffset(years=years_back)
    recent = out[out["date_added"] >= cutoff]
    # If the mirror's dates are older than the window, keep newest rows so the
    # pipeline still produces data instead of an empty set (logged honestly).
    if recent.empty:
        log.warning("clean.no_rows_in_window_using_newest", cutoff=str(cutoff.date()))
        recent = out.sort_values("date_added")
    else:
        out = recent
        log.info("clean.year_filter", removed=before - len(out), cutoff=str(cutoff.date()))
    out = recent

    before = len(out)
    out = out.drop_duplicates(subset=["make", "model", "model_year", "narrative"])
    log.info("clean.dedupe", removed=before - len(out))

    out = out.sort_values(["date_added", "cmplid"]).head(max_rows).reset_index(drop=True)
    log.info("clean.done", rows_in=n0, rows_out=len(out), cap=max_rows)
    return out


def write_quality_report(raw_rows: int, cleaned: pd.DataFrame, path: Path) -> None:
    """Persist honest row-count/quality stats used by the data card & EDA."""
    stats = {
        "raw_rows": int(raw_rows),
        "cleaned_rows": int(len(cleaned)),
        "dropped_rows": int(raw_rows - len(cleaned)),
        "empty_narrative_min_chars": _MIN_NARRATIVE_CHARS,
        "indicator_positive_counts": {
            c: int(cleaned[c].sum()) for c in ("crash", "fire", "injured", "deaths")
        },
        "distinct_makes": int(cleaned["make"].nunique()),
        "distinct_components": int(cleaned["component"].nunique()),
        "mean_narrative_chars": float(cleaned["narrative"].str.len().mean()),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2))
    log.info("clean.report_written", path=str(path))


def run(dest_dir: Path | None = None) -> Path:
    """Full local pipeline: cached raw CSV -> processed parquet. Idempotent."""
    dest_dir = dest_dir or DATA_PROCESSED_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    raw = load_raw(DATA_RAW_DIR / RAW_FILENAME)
    cleaned = clean(raw)
    target = dest_dir / "complaints.parquet"
    cleaned.to_parquet(target, index=False)
    write_quality_report(len(raw), cleaned, REPORTS_DIR / "data_quality.json")
    log.info("clean.written", path=str(target), rows=len(cleaned))
    return target


def main() -> None:
    """CLI entry point used by the Makefile ``data`` target."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None, help="Override output parquet path.")
    args = parser.parse_args()
    path = run(Path(args.out).parent if args.out else None)
    print(f"OK: wrote {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
