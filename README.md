# CyberResearchAI
Big Data-based Cybersecurity Research Assistant with RAG and Verified Citations

## Status

| Phase | Scope | Status |
|---|---|---|
| 0 | Repository audit, environment, configuration | Done |
| 1 | Academic data ingestion (arXiv, OpenAlex, Semantic Scholar) | Done |

## Setup (Windows PowerShell)

```powershell
# 1. Create the virtual environment (Python 3.11) - skip if .venv already exists
py -3.11 -m venv .venv

# 2. Activate it. If scripts are blocked, run once:
#    Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies (runtime + test tools)
python -m pip install -r requirements-dev.txt

# 4. Create your local secrets file and fill in values (never commit .env)
Copy-Item .env.example .env

# 5. Verify the environment and configuration
python main.py check

# 6. Run the tests
python -m pytest
```

## Data ingestion (Phase 1)

```powershell
python main.py init-db                                    # create tables (idempotent)
python main.py daily                                      # incremental: recent papers
python main.py daily --sources arxiv openalex --max-records 50
python main.py backfill --start 2023-01-01 --end 2024-12-31   # historical range
python main.py status                                     # counts, checkpoints, last logs
```

### How it works

```
arXiv (Atom XML) ─┐
OpenAlex (JSON)  ─┼─> HttpClient ──> normalize ──> PaperRecord ──> dedup ──> PostgreSQL
Semantic Scholar ─┘   timeout, retry,   DOI, arXiv ID,   common      ID / title+year
                      backoff, rate     title, dates     schema      matching
                      limit
```

| Concern | Implementation |
|---|---|
| Separate API clients | `src/cyberresearch/ingestion/clients/{arxiv,openalex,semantic_scholar}.py`, one shared `fetch_page()` interface |
| Common schema | `PaperRecord` in `ingestion/schema.py`; original payload kept in `raw` |
| Timeouts, retries, backoff | `ingestion/http.py`: retries timeouts, connection errors, 429 and 5xx with `base * 2^attempt` + jitter, honours `Retry-After`; other 4xx fail fast |
| Rate limits | Minimum interval per source (`config/settings.yaml`): arXiv 3 s, OpenAlex 0.2 s, Semantic Scholar 1.1 s with key / 3 s without |
| Deduplication | `ingestion/dedup.py`, priority: arXiv ID → DOI → Semantic Scholar ID → OpenAlex ID → normalized title + year (rejected if DOIs/arXiv IDs conflict). Missing fields are filled from later sources, never overwritten |
| Missing metadata | Every field except `source`/`source_id` is optional; items with neither a title nor an identifier are skipped and counted |
| No needless reprocessing | `paper_sources.content_hash`: an identical record is only marked as seen (`unchanged`) |
| Checkpoint & resume | `ingestion_checkpoints` stores the API cursor after every page, in the same transaction as that page's records |
| Daily vs backfill | `daily`: window = last completed end − 1 overlap day → today. `backfill`: range split into 12-month windows (newest first); completed windows are skipped |
| Logs | `ingestion_logs` table (one row per source/topic/window) + JSON-lines file `logs/ingestion.log` with API keys redacted |

### Database tables

| Table | One row per | Key columns |
|---|---|---|
| `papers` | unique paper | `doi`, `arxiv_id`, `s2_paper_id`, `openalex_id` (unique), `normalized_title`+`year`, `topics[]` |
| `paper_sources` | (source, source_id) | `paper_id` FK, `raw` JSONB, `content_hash`, first/last seen |
| `ingestion_logs` | run × source × topic × window | status, fetched / inserted / merged / updated / unchanged / failed |
| `ingestion_checkpoints` | mode × source × topic × window | `cursor`, `status` |

PDFs are **not** downloaded in this phase.

### Source notes (verified 2026-11)

- **OpenAlex** requires an API key for cursor paging beyond the first page
  (HTTP 401 otherwise). Without `OPENALEX_API_KEY` the client uses page-number
  paging, limited by OpenAlex to the first 10,000 results per query.
- **Semantic Scholar** works without a key, but the shared pool is often
  throttled (HTTP 429, retried automatically). A free key is recommended.
- **arXiv** asks for at most one request every 3 seconds; large backfills are
  therefore slow by design.

## Tests

```powershell
python -m pytest              # offline tests (fake HTTP + temporary PostgreSQL)
python -m pytest -m live -v   # real API calls (network required)
```

Database tests use `TEST_DATABASE_URL` if set; otherwise a temporary embedded
PostgreSQL. Each test runs in its own throw-away schema.

## Project layout

```
config/settings.yaml          Non-secret settings: sources, rate limits, topics
src/cyberresearch/
  config.py                   Loads settings.yaml + environment variables (.env)
  logging_setup.py            JSON-lines logging with API-key redaction
  db.py                       DATABASE_URL or embedded dev PostgreSQL
  ingestion/
    http.py                   Timeouts, retries, backoff, rate limiting
    normalize.py              DOI / arXiv ID / title / date normalization
    schema.py                 PaperRecord (common schema)
    dedup.py                  Duplicate detection and merge rules
    storage.py                PostgreSQL persistence
    pipeline.py               Daily and backfill modes, checkpoints
    cli.py                    init-db, daily, backfill, status commands
    clients/                  arxiv.py, openalex.py, semantic_scholar.py
    sql/schema.sql            Table definitions
tests/                        Unit tests (+ tests/live for real APIs)
main.py                       Entry point
```

## Database

Set `DATABASE_URL` in `.env` to use your own PostgreSQL (with the pgvector
extension). If it is empty, development and tests use an embedded
PostgreSQL 16 + pgvector instance provided by the `pgserver` package and stored
in `data/pgdata/` (git-ignored).

## Branch strategy

- `main` - stable, presentation-ready releases
- `develop` - integration branch; every feature merges here first
- `feature/<name>` - one branch per phase (`feature/data-ingestion`,
  `feature/pdf-processing`, `feature/rag-chatbot`, `feature/web-interface`)
- `fix/<name>` - bug fixes, branched from `develop`
