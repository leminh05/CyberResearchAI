# CyberResearchAI
Big Data-based Cybersecurity Research Assistant with RAG and Verified Citations

## Status

| Phase | Scope | Status |
|---|---|---|
| 0 | Repository audit, environment, configuration | Done |
| 1 | Academic data ingestion (arXiv, OpenAlex, Semantic Scholar) | Not started |

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

## Project layout

```
config/settings.yaml      Non-secret settings: sources, rate limits, topics
src/cyberresearch/        Application package
  config.py               Loads settings.yaml + environment variables (.env)
  logging_setup.py        JSON-lines logging with API-key redaction
tests/                    Unit tests (pytest)
main.py                   Entry point: `python main.py check`
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
