"""Central application configuration via pydantic-settings.

Every setting can be provided by environment variable or a local ``.env`` file
(see ``.env.example``).  Nothing secret is ever committed; secrets come from the
environment at runtime only.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository layout constants -------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = REPO_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = REPO_ROOT / "data" / "processed"
DATA_LABELED_DIR = REPO_ROOT / "data" / "labeled"
REPORTS_DIR = REPO_ROOT / "reports"
MODELS_DIR = REPO_ROOT / "models"


class Settings(BaseSettings):
    """Runtime settings for AutoVoice (env vars are case-insensitive)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- general ---
    app_name: str = "autovoice"
    environment: str = Field(default="dev", pattern=r"^(dev|test|prod)$")
    log_level: str = Field(default="INFO")
    log_raw_text: bool = Field(
        default=False,
        description="If true, request text may appear in logs. Keep false in prod.",
    )

    # --- API service ---
    api_host: str = "0.0.0.0"  # noqa: S104 - Cloud Run requires binding all interfaces
    api_port: int = 8080
    cors_origins_raw: str = Field(
        default="http://localhost:8501",
        validation_alias="CORS_ORIGINS",
        description="Comma-separated list, e.g. CORS_ORIGINS='https://a.b,https://c.d'",
    )
    rate_limit_per_minute: int = 60
    max_text_length: int = 5000

    # --- models ---
    ner_model_path: str = "models/ner"
    clf_model_path: str = "models/clf"
    model_version: str = "0.1.0"
    hf_token_env: str = "HF_TOKEN"  # name only; never the value

    # --- RAG ---
    llm_provider: str = Field(default="none", pattern=r"^(none|gemini|local)$")
    gemini_api_key_env: str = "GEMINI_API_KEY"  # name of the env var holding the key
    gemini_model: str = "gemini-1.5-flash"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chroma_dir: str = "data/processed/chroma"
    rag_top_k: int = 5
    rag_min_similarity: float = 0.35

    # --- BigQuery (all optional; local DuckDB/pandas fallback when unset) ---
    gcp_project_id: str | None = None
    bq_dataset: str = "autovoice"
    bq_table: str = "complaints"
    bq_location: str = "US"

    @property
    def cors_origins(self) -> list[str]:
        """CORS origins parsed from the comma-separated env value."""
        return [o.strip() for o in self.cors_origins_raw.split(",") if o.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance (reads .env / environment once)."""
    return Settings()
