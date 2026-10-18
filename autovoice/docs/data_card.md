# Data Card — AutoVoice (NHTSA Vehicle Complaints)

_Last updated: 2026-10-18. All counts below come from `reports/data_quality.json`,
which is produced by `src/autovoice/data/clean.py` — not typed by hand._

## 1. Source

- **Dataset:** Public NHTSA vehicle-complaint records (Office of Defects
  Investigation consumer complaints).
- **Access path used:** public-domain mirror on Hugging Face Datasets,
  `claritystorm/nhtsa-vehicle-complaints` (file `sample_1000.csv`), because at
  verification time (2026-10-17):
  - the legacy `static.nhtsa.gov` bulk TXT paths returned **HTTP 404**,
  - `www.nhtsa.gov` data pages are behind Akamai bot protection (**HTTP 403**)
    from this environment, and
  - `api.nhtsa.gov/complaints` requires a personal API key.
  The download module fails loudly with instructions if the URL moves; the
  canonical location must be re-verified on NHTSA's official site before use.
- **License / data-use note:** NHTSA complaint data is a U.S. government
  publication (public domain); the HF mirror declares license = "Public
  Domain" in its cardData. **No personal data** is intentionally used —
  narratives are sanitized by NHTSA before publication; we additionally never
  log raw text in production (`LOG_RAW_TEXT=false` default).

## 2. Fields kept (after cleaning)

| Column | Meaning | Used for |
|---|---|---|
| `cmplid` | complaint ID | dedupe/key |
| `make`, `model`, `model_year` | vehicle identity | metadata filters, clustering |
| `component` | NHTSA component description | RAG metadata, NER lexicon, clustering |
| `narrative` | complaint text | **the only model input** |
| `crash`, `fire`, `injured`, `deaths` | indicator flags (0/1) | **rule-derived severity labels only — never model input** |
| `date_added` | date added to database | partition field, recency filter |

## 3. Cleaning steps (`make data`)

1. Schema validation against the official required column set (fails loudly).
2. Unicode normalization (smart quotes → ASCII), whitespace collapse.
3. Light boilerplate removal (regex list in `clean.py`).
4. Drop narratives shorter than 40 characters.
5. Parse dates/years; map indicators to 0/1 ints.
6. Dedupe on `(make, model, model_year, narrative)`.
7. Filter to last 5 years (fallback: newest rows, logged as a warning) and cap
   at 5,000 rows to stay inside free-tier budgets.

## 4. Current subset statistics (from `reports/data_quality.json`)

- Raw rows downloaded: **1,000**; cleaned rows kept: **190** (810 dropped,
  mostly short narratives).
- Distinct makes: **30**; distinct components: **33**.
- Mean narrative length: **≈574 characters**.
- Positive indicators in the subset: crash **7**, fire **3**, injured **2**,
  deaths **0** — heavily imbalanced; see severity-label disclosure in the
  model card.

## 5. Known limitations

- The mirror is a **sample** (1,000 rows), not the full multi-million-row
  database; demo metrics reflect that subset only.
- Indicator fields are sparsely populated by complainants; absence of a flag
  does not mean the event did not happen.
- Narratives are unverified consumer allegations, **not confirmed defects**.
- Dates in mirrors can lag reality; year filtering logs what it actually did.
- English-only, US-market vehicles.

## 6. BigQuery handling

One dataset/table, **partitioned by `date_added` (DAY)** and **clustered by
`(make, component)`**; loaded exclusively via **batch load jobs**
(`load_table_from_file`, NDJSON) — no streaming inserts. `--dry-run` prints
schema + byte estimates computed locally from the actual file (estimates, not
billed figures).
