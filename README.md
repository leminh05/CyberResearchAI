# CyberResearchAI
Big Data-based Cybersecurity Research Assistant with RAG and Verified Citations

## Status

| Phase | Scope | Status |
|---|---|---|
| 0 | Repository audit, environment, configuration | Done |
| 1 | Academic data ingestion (arXiv, OpenAlex, Semantic Scholar) | Done |
| 2 | PDF processing, chunking, embeddings, pgvector, PySpark + benchmark | Done |

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

## PDF processing and embeddings (Phase 2)

```powershell
python main.py process --limit 40          # access check -> download -> extract -> chunk -> embed -> store
python main.py process --no-download       # abstracts only, no PDF downloads
python main.py process --reprocess-pdfs    # re-chunk PDFs already on disk (after rule changes)
python main.py search "intrusion detection with graph neural networks" -k 5
```

```
papers ──> access policy ──> polite downloader ──> PyMuPDF ──> cleaning ──> sections ──> chunks ──> embeddings ──> pgvector
           (license check)   (robots.txt,          (blocks     (headers,    (numbered    (page per   (all-MiniLM    (HNSW,
                              crawl-delay,          per page)   footers,     headings)    word)       -L6-v2, 384d)   cosine)
                              %PDF check)                       hyphens)
```

### Legal access policy (`processing/access.py`)

| Case | Decision |
|---|---|
| Paper has an arXiv ID | Download from `arxiv.org/pdf/<id>` (arXiv's robots.txt allows `/pdf/` with a **15 s crawl delay**; `export.arxiv.org` disallows it) |
| OpenAlex / Semantic Scholar PDF with a Creative Commons or public-domain license | Download |
| License missing, `other-oa`, or restrictive | **Not downloaded**; the abstract from the API is used instead |
| No PDF URL | Abstract only |

The downloader also honours `robots.txt` (a site that disallows us or whose
robots.txt is unavailable is skipped), rate-limits each host, rejects files
larger than 30 MB, and checks the `%PDF-` signature (many "PDF" links are HTML
landing pages). PDFs are stored in `data/pdfs/` (git-ignored, never committed).

### Text processing (`processing/`)

| Step | Module | What it does |
|---|---|---|
| Extract | `extract.py` | PyMuPDF text blocks per page; rejects non-PDF, encrypted and scanned (no text layer) files |
| Clean | `cleaning.py` | Removes running headers/footers (blocks repeated on ≥ 50% of pages), page numbers, arXiv margin stamps; fixes ligatures and hyphenation |
| Sections | `sections.py` | Detects `1 Introduction`, `1\nIntroduction` (LaTeX layout), `IV. EXPERIMENTS`, `References`; a numbered heading is accepted only if its number follows the previous one, which rejects list items and algorithm steps |
| Chunk | `chunking.py` | ~220 words with 40 words overlap, never across sections, cut at sentence ends; every word carries its page, so `page_start`/`page_end` are exact; references are excluded |
| Embed | `embeddings.py` | `sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions, L2-normalized |
| Store | `store.py` | `pdf_documents` (status per paper) and `paper_chunks` (text, section, pages, URL, `vector(384)`, HNSW index) |

A database `CHECK` constraint makes it impossible to store a PDF chunk without
a page number. Abstract-only chunks are stored with `content_source =
'abstract'` and no page, so later phases can cite them as "abstract".

### PySpark (`bigdata/`)

| Job | Spark operations |
|---|---|
| Metadata normalization | `regexp_replace`, `lower`, `substring`, `cast`, `when` - built-in JVM functions, no Python UDF |
| Data filtering | `filter` with null checks and year range |
| Deduplication | priority key with `coalesce`; `Window.partitionBy(key).orderBy(completeness, source rank)` + `row_number() == 1` |
| Data quality checks | one `agg` pass for null rates, counts before/after each step, `groupBy(source).count()` |
| Batch processing | `groupBy(paper_id)` + `collect_list` of pages, then the same Python cleaning/section/chunking code in a UDF, `explode` to one row per chunk |

`bigdata/python_baseline.py` implements exactly the same rules in plain
Python; tests assert that both produce identical results. Spark runs in
**local mode** (`local[*]`: one JVM on this machine) - see the benchmark
section for what this does and does not show.

### Benchmark: plain Python vs PySpark

```powershell
python main.py benchmark --sizes 1000 10000 100000 300000 --repeats 3
```

Same input file, same rules, identical output for every size (checked
record by record). Full report: [docs/benchmark_results.md](docs/benchmark_results.md).
Measured 2026-11-02 on this laptop (12 logical CPUs, Windows 11, Java 8,
Spark 3.5.9 `local[*]`, driver memory 2 GB):

| Records | File | Python | Spark | Spark / Python | Python peak heap | Spark JVM heap used |
|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 1.7 MB | 0.08 s | 0.87 s | 10.4x slower | 4 MB | 437 MB |
| 10,000 | 17.7 MB | 0.82 s | 1.10 s | 1.3x slower | 40 MB | 212 MB |
| 100,000 | 177 MB | 8.99 s | 4.17 s | 2.2x faster | 397 MB | 505 MB |
| 300,000 | 532 MB | 26.74 s | 8.79 s | 3.0x faster | 1,195 MB | 849 MB |

Spark session startup: **4.2 s**, paid once per process and *not* included above.

What this shows:

- **Overhead dominates small data.** Planning a job, scheduling tasks and
  starting the JVM-side work costs ~0.8 s even for 1,000 records; with the
  4.2 s startup, plain Python wins easily below ~10,000 records. Our real
  corpus (381 records today) is far below that point.
- **Spark wins on larger files because it uses all 12 cores.** Python runs
  on one core; Spark splits the file into partitions processed in parallel.
  Part of the gain is simply parallelism - a `multiprocessing` version of
  the Python code would narrow the gap.
- **Memory grows differently.** The Python version holds every record in a
  list (heap grows linearly: ~4 KB per record, 1.2 GB at 300k). Spark streams
  partitions and spills to disk if needed; its JVM heap figure is
  approximate (it depends on garbage collection) and includes Spark's own
  overhead, which is why it is large for small inputs.

What this does **not** show:

- **No cluster performance.** Everything ran on one machine. Network shuffle,
  data locality and executor failures - the main costs and benefits of a
  real cluster - are not measured. These numbers must not be presented as
  distributed-scaling results.
- **Synthetic scale-up.** Only 381 records are real; larger sizes are copies
  with new identifiers and ~20% injected duplicates (`bigdata/dataset.py`).
- **Single-key deduplication.** Both implementations use one dedup key per
  record. Linking records through different identifiers (A~B by DOI, B~C by
  arXiv ID) needs connected components (e.g. GraphFrames), which is handled
  by the database stage instead.
- **Python UDF cost.** The batch-chunking job reuses Python code in a UDF,
  which serializes data between the JVM and Python workers; built-in Spark
  functions (used for metadata) avoid that cost.

## Tests

```powershell
python -m pytest              # offline tests (fake HTTP, temporary PostgreSQL, local Spark)
python -m pytest -m live -v   # real API calls (network required)
python -m pytest -m model -v  # real Sentence Transformers model (downloads ~90 MB once)
```

Test PDFs are generated by `tests/pdf_factory.py` at run time (running
headers, page footers, arXiv stamp, hyphenation, numbered sections), so no
binary or copyrighted PDFs are stored in the repository.

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
  processing/
    access.py                 Legal PDF access policy
    downloader.py             robots.txt, crawl-delay, PDF validation
    extract.py                PyMuPDF text blocks per page
    cleaning.py               Header/footer/noise removal
    sections.py               Section heading detection
    chunking.py               Page-preserving chunking
    embeddings.py             Sentence Transformers
    store.py                  pgvector storage and similarity search
    pipeline.py               Phase 2 pipeline (incremental)
    sql/processing_schema.sql pdf_documents, paper_chunks
  bigdata/
    spark_jobs.py             PySpark DataFrame jobs
    python_baseline.py        Same rules in plain Python
    dataset.py                Benchmark dataset (real + synthetic scale-up)
    benchmark.py              Python vs Spark timing and memory
docs/benchmark_results.md     Latest benchmark report (generated)
tests/                        Unit tests (+ tests/live for real APIs)
main.py                       Entry point
```

## Database

Set `DATABASE_URL` in `.env` to use your own PostgreSQL (with the pgvector
extension). If it is empty, development and tests use an embedded
PostgreSQL 16 + pgvector instance provided by the `pgserver` package and stored
in `data/pgdata/` (git-ignored).

## Claude on GitHub (`@claude`)

`.github/workflows/claude.yml` lets the repository owner or collaborators
mention `@claude` in an issue, PR comment or review; Claude replies and can
push commits as `claude[bot]` (branches prefixed `claude/`).

Setup, done once by the owner:

1. Install the Claude GitHub App for this repository: https://github.com/apps/claude
2. Add **one** repository secret (Settings > Secrets and variables > Actions):
   `CLAUDE_CODE_OAUTH_TOKEN` (Claude Pro/Max, created with `claude setup-token`)
   or `ANTHROPIC_API_KEY` (pay per use). Never commit these values.
3. Merge the workflow into `main`: GitHub runs comment-triggered workflows
   from the default branch only.

Runs use GitHub Actions minutes of this private repository.

## Branch strategy

- `main` - stable, presentation-ready releases
- `develop` - integration branch; every feature merges here first
- `feature/<name>` - one branch per phase (`feature/data-ingestion`,
  `feature/pdf-processing`, `feature/rag-chatbot`, `feature/web-interface`)
- `fix/<name>` - bug fixes, branched from `develop`
