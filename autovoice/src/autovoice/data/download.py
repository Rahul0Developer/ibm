"""Download public NHTSA vehicle-complaint data into ``data/raw/`` (cached).

Data source (verified 2026-10-17, see docs/data_card.md):
    The canonical bulk file on ``static.nhtsa.gov`` returned HTTP 404 and the
    ``www.nhtsa.gov`` pages are behind Akamai bot protection (HTTP 403) from
    this environment; the official ``api.nhtsa.gov/complaints`` endpoint
    requires an API key.  We therefore use a *public-domain mirror* of the
    same NHTSA complaint records hosted on Hugging Face Datasets
    (``claritystorm/nhtsa-vehicle-complaints``, cardData license = public
    domain).  The schema was verified against the official NHTSA field list.

Truthfulness rule: if the URL moves, we fail loudly with instructions rather
than silently pointing at an unverified source.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import httpx
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from autovoice.config import DATA_RAW_DIR, get_settings
from autovoice.logging_conf import get_logger

log = get_logger(__name__)

# --- Source configuration ----------------------------------------------------
# This exact URL returned HTTP 200 with the expected columns when last
# verified (2026-10-17).  Override with env AUTVOICE_DATA_URL if NHTSA's
# canonical location becomes reachable for you.
DEFAULT_DATA_URL = (
    "https://huggingface.co/datasets/claritystorm/nhtsa-vehicle-complaints/"
    "resolve/main/sample_1000.csv"
)
OFFICIAL_PAGE_HINT = (
    "Verify the current location on the official NHTSA data page: "
    "https://www.nhtsa.gov/recalls (Vehicle Complaint Data section). "
    "Then set AUTVOICE_DATA_URL to the new file."
)

#: Required columns, checked at runtime against the official NHTSA field list.
#: (cmplid = complaint ID, comp_desc = component description,
#: complaint_desc = narrative, crash/fire/injured/deaths = indicator fields,
#: date_added = received/added date.)
REQUIRED_COLUMNS = {
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
}

RAW_FILENAME = "nhtsa_complaints_raw.csv"


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
def _fetch_bytes(url: str, timeout_s: float = 60.0) -> bytes:
    """Fetch ``url`` with retries/backoff; raise a clear error if moved."""
    log.info("download.start", url=url)
    try:
        resp = httpx.get(url, timeout=timeout_s, follow_redirects=True)
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        if status in (403, 404):
            raise RuntimeError(
                f"NHTSA data URL returned HTTP {status}: {url}\n"
                "The file may have moved. Do NOT hardcode a guessed URL.\n"
                + OFFICIAL_PAGE_HINT
            ) from exc
        raise
    except httpx.RequestError as exc:
        raise RuntimeError(f"Network error fetching {url}: {exc!r}") from exc
    log.info("download.done", url=url, bytes=len(resp.content))
    return resp.content


def _read_csv(raw: bytes) -> pd.DataFrame:
    """Parse CSV bytes and validate the schema against REQUIRED_COLUMNS."""
    df = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise RuntimeError(
            f"Downloaded file is missing required NHTSA columns: {sorted(missing)}. "
            f"Found columns: {sorted(df.columns)}. " + OFFICIAL_PAGE_HINT
        )
    return df


def download_raw(dest_dir: Path | None = None, url: str | None = None) -> Path:
    """Download (or reuse cached) raw complaints CSV; return the local path.

    Args:
        dest_dir: Directory for the cached file; defaults to ``data/raw``.
        url: Source URL; defaults to settings/env or DEFAULT_DATA_URL.

    Returns:
        Path to the cached raw CSV file.
    """
    settings = get_settings()
    dest_dir = dest_dir or DATA_RAW_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / RAW_FILENAME
    source = url or getattr(settings, "data_url", "") or DEFAULT_DATA_URL

    if target.exists() and target.stat().st_size > 0:
        log.info("download.cached", path=str(target))
        return target

    raw = _fetch_bytes(source)
    df = _read_csv(raw)  # validate before caching anything broken
    target.write_bytes(raw)
    log.info("download.validated", rows=len(df), cols=len(df.columns))
    return target


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Load the cached raw CSV (must run :func:`download_raw` first)."""
    path = path or DATA_RAW_DIR / RAW_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `make data` first.")
    return _read_csv(path.read_bytes())


def main(url: str | None = None) -> None:
    """CLI entry point used by the Makefile ``data`` target."""
    path = download_raw(url=url)
    df = load_raw(path)
    print(f"OK: {len(df)} raw rows cached at {path}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=None, help="Override source URL.")
    args = parser.parse_args()
    main(url=args.url)
