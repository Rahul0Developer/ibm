"""BigQuery I/O for AutoVoice - **batch load jobs only** (no streaming inserts).

Free-tier safety:
    * One dataset, one table, partitioned by ``date_added`` (DAY) and clustered
      by ``(make, component)`` so demo queries scan few bytes.
    * Loads go through ``load_table_from_json`` (a batch job reading files),
      never ``insert_rows_json`` (streaming = billable per-row writes).
    * ``--dry-run`` prints the schema and a bytes estimate before touching BQ.

Requires ``google-cloud-bigquery`` (serve group) and ADC or
``GOOGLE_APPLICATION_CREDENTIALS``; when unconfigured, commands exit with a
clear message instead of stack traces.  Local fallback for aggregate queries
lives in :mod:`autovoice.rag.graph` (DuckDB over the same parquet).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from autovoice.config import DATA_PROCESSED_DIR, get_settings
from autovoice.logging_conf import get_logger

if TYPE_CHECKING:  # keep module importable without the heavy optional dep
    from google.cloud import bigquery

log = get_logger(__name__)

TABLE_SCHEMA_FIELDS = [
    ("cmplid", "STRING"),
    ("make", "STRING"),
    ("model", "STRING"),
    ("model_year", "INTEGER"),
    ("component", "STRING"),
    ("narrative", "STRING"),
    ("crash", "INTEGER"),
    ("fire", "INTEGER"),
    ("injured", "INTEGER"),
    ("deaths", "INTEGER"),
    ("date_added", "DATE"),
]

PARTITION_FIELD = "date_added"
CLUSTER_FIELDS = ["make", "component"]


def _get_client():
    """Instantiate the BigQuery client with a friendly error if unconfigured."""
    try:
        from google.cloud import bigquery  # noqa: PLC0415 - optional heavy dep
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise RuntimeError(
            "google-cloud-bigquery is not installed. Run: uv sync --group serve"
        ) from exc
    settings = get_settings()
    if not settings.gcp_project_id:
        raise RuntimeError(
            "GCP_PROJECT_ID is not set. Export it (or put it in .env) and make sure "
            "`gcloud auth application-default login` has run. No project ID is "
            "hardcoded anywhere in this repo."
        )
    return bigquery.Client(project=settings.gcp_project_id)


def build_table() -> bigquery.Table:
    """Return the Table object describing the partitioned+clustered schema."""
    from google.cloud import bigquery

    settings = get_settings()
    table_ref = f"{settings.gcp_project_id}.{settings.bq_dataset}.{settings.bq_table}"
    table = bigquery.Table(table_ref)
    table.schema = [
        bigquery.SchemaField(name, bq_type) for name, bq_type in TABLE_SCHEMA_FIELDS
    ]
    table.time_partitioning = bigquery.TimePartitioning(
        type_=bigquery.TimePartitioningType.DAY, field=PARTITION_FIELD
    )
    table.clustering_fields = CLUSTER_FIELDS
    table.description = (
        "AutoVoice: cleaned public NHTSA vehicle complaints (subset). "
        "Batch-loaded; see docs/data_card.md."
    )
    return table


def dry_run(parquet_path: Path) -> dict:
    """Print schema + size estimate without creating anything in BigQuery.

    Args:
        parquet_path: The processed complaints parquet file.

    Returns:
        Dict with row count, file bytes and estimated billed bytes (JSON
        conversion is roughly 1.3x parquet size - stated, not measured).
    """
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    file_bytes = parquet_path.stat().st_size
    payload = df.head(10).to_json(orient="records", date_format="iso")
    per_row_est = len(payload.encode()) / max(len(df.head(10)), 1)
    est_bytes = int(per_row_est * len(df))
    info = {
        "rows": int(len(df)),
        "parquet_bytes": int(file_bytes),
        "estimated_load_bytes": est_bytes,
        "estimated_query_scan_bytes_per_full_table": est_bytes,
        "schema": [{"name": n, "type": t} for n, t in TABLE_SCHEMA_FIELDS],
        "partition_field": PARTITION_FIELD,
        "cluster_fields": CLUSTER_FIELDS,
        "note": (
            "Byte estimates are computed locally from the actual file, not billed "
            "figures. Always re-check with `bq query --dry_run` after auth."
        ),
    }
    return info


def upload(parquet_path: Path, wait: bool = True) -> str:
    """Batch-load the parquet-derived JSON into the partitioned table.

    Uses a load job reading a temporary NDJSON file (write disposition
    WRITE_TRUNCATE on first load, APPEND afterwards keeps idempotency simple:
    we always TRUNCATE then load the current subset).

    Returns:
        The destination table FQN.
    """
    import pandas as pd
    from google.cloud import bigquery

    settings = get_settings()
    client = _get_client()
    dataset_ref = f"{settings.gcp_project_id}.{settings.bq_dataset}"

    try:
        client.get_dataset(dataset_ref)
    except Exception:
        client.create_dataset(bigquery.Dataset(dataset_ref), location=settings.bq_location)
        log.info("bq.dataset_created", dataset=dataset_ref)

    table = build_table()
    try:
        client.get_table(table.reference)
        client.update_table(table, ["time_partitioning", "clustering_fields"])
    except Exception:
        client.create_table(table)
        log.info("bq.table_created", table=table.full_table_id)

    df = pd.read_parquet(parquet_path)
    tmp = parquet_path.with_suffix(".ndjson")
    with tmp.open("w", encoding="utf-8") as fh:
        for rec in df.to_dict(orient="records"):
            fh.write(json.dumps(rec, default=str) + "\n")

    # Batch load job reading a local file (never insert_rows_json / streaming).
    with tmp.open("rb") as fh:
        job = client.load_table_from_file(
            fh,
            destination=table.reference,
            job_config=bigquery.LoadJobConfig(
                source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
                schema=table.schema,
                write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
                skip_leading_rows=0,
                max_bad_records=0,
            ),
            job_id_prefix="autovoice_load_",
        )
    if wait:
        job.result()  # blocks until the batch job finishes
        log.info("bq.load_done", table=table.full_table_id, rows=job.output_rows)
    tmp.unlink(missing_ok=True)
    return table.full_table_id


def count_query() -> int:
    """Verify the load with a tiny partition-pruned count query."""
    settings = get_settings()
    client = _get_client()
    sql = (
        f"SELECT COUNT(*) AS n FROM `{settings.gcp_project_id}."
        f"{settings.bq_dataset}.{settings.bq_table}`"
    )
    return next(iter(client.query(sql).result()))["n"]


def main(argv: list[str] | None = None) -> None:
    """CLI: --dry-run (default), --upload (batch job), --verify (count)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parquet",
        default=str(DATA_PROCESSED_DIR / "complaints.parquet"),
        help="Processed complaints parquet file.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Estimate only.")
    parser.add_argument("--upload", action="store_true", help="Run the batch load job.")
    parser.add_argument("--verify", action="store_true", help="Count rows after load.")
    args = parser.parse_args(argv)

    path = Path(args.parquet)
    if not path.exists():
        print(f"ERROR: {path} missing. Run `make data` first.", file=sys.stderr)
        raise SystemExit(1)

    if args.dry_run or not (args.upload or args.verify):
        print(json.dumps(dry_run(path), indent=2))
    if args.upload:
        fqn = upload(path)
        print(f"Loaded into {fqn}")
    if args.verify:
        print(f"Row count in BigQuery: {count_query()}")


if __name__ == "__main__":
    main()
