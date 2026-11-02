# CyberResearchAI
Big Data-based Cybersecurity Research Assistant with RAG and Verified Citations

## Status

| Phase | Scope | Status |
|---|---|---|
| 0 | Repository audit, environment, configuration | Done |
| 1 | Academic data ingestion (arXiv, OpenAlex, Semantic Scholar) | Done |
| 2 | PDF processing, chunking, embeddings, pgvector, PySpark + benchmark | Done |
| 3 | RAG engine with verified citations + terminal chatbot | Done |

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

## Docker (PostgreSQL + application)

Needs Docker Desktop with Compose v2. To keep images off drive C, set
*Settings → Resources → Advanced → Disk image location* (for example `D:\Docker`)
before the first pull.

```powershell
Copy-Item .env.example .env       # then set POSTGRES_PASSWORD and the same value in DATABASE_URL
docker compose up -d              # PostgreSQL 16 + pgvector 0.8.0 on 127.0.0.1:5432 (volume "pgdata")
python main.py check              # the host app now uses DATABASE_URL from .env

# Optional: run the application itself in a container (profile "app")
docker compose --profile app build app
docker compose run --rm app check
docker compose run --rm app rag-eval --retrieval-only
docker compose run --rm app chat --provider ollama   # Ollama stays on the host (GPU)

docker compose down               # stop; data stays in the pgdata volume
```

| Item | Choice | Why |
|---|---|---|
| Image | `pgvector/pgvector:0.8.0-pg16` (pinned) | Same PostgreSQL 16 as before, newer pgvector than the embedded 0.6.2 |
| Port | `127.0.0.1:5432` only | The database is not reachable from the network |
| Host name in `DATABASE_URL` | `127.0.0.1`, **not** `localhost` | On Windows `localhost` resolves to IPv6 `::1` first; the connection then stalls (measured: 8 s with a timeout, a hang without) |
| App image | `python:3.11-slim` + Java 17 + CPU-only PyTorch, non-root user | PySpark runs with Java 17 inside the image regardless of the host's Java 8 |
| Ollama | Host, reached at `host.docker.internal:11434` | GPU access is simpler outside Docker on Windows |

**Moving existing data.** `scripts/migrate_embedded_to_docker.py` copies the
embedded development database (`data/pgdata`) into the Docker database with the
`pg_dump`/`pg_restore` that ship with `pgserver`, refuses to overwrite a
non-empty target, and compares row counts per table. Run on 2026-11-02: 377
papers, 381 source records, 7,691 chunks - identical counts on both sides.

**Tests against Docker:** set `TEST_DATABASE_URL` to the same URL as
`DATABASE_URL`; every test runs in its own temporary schema and drops it.

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

## RAG chatbot (Phase 3)

Two interchangeable answer generators, selected with `rag.provider` in
`config/settings.yaml` or `--provider` on the command line:

| Provider | Needs | Notes |
|---|---|---|
| `claude` (default) | `ANTHROPIC_API_KEY` in `.env` (billed per use) | `claude-opus-5-5`, structured output, refusal fallback |
| `ollama` | [Ollama](https://ollama.com) running locally + `ollama pull qwen2.5:7b` | Free, offline. On this laptop (GTX 1650 Ti 4 GB, 32 GB RAM) about 2-3 min per answer; less accurate |

Prompt, JSON schema and citation validation are the same for both, so a
weaker local model cannot slip unverified citations through - its answers are
downgraded instead. Ollama's context window is set explicitly
(`rag.ollama.num_ctx`), and a prompt that fills it is rejected, because Ollama
otherwise silently cuts long prompts. To keep models off drive C, set the user
environment variable `OLLAMA_MODELS` (for example `D:\Ollama\models`) and
restart Ollama before pulling.

```powershell
python main.py chat --provider ollama                  # free local model
python main.py chat                                    # interactive chatbot (Claude)
python main.py chat --year-min 2025 --topic intrusion_detection
python main.py ask "How are graph neural networks used for intrusion detection?"
python main.py ask "..." --json                        # validated answer + evidence as JSON
python main.py rag-eval --retrieval-only               # test questions, no Claude calls
python main.py rag-eval                                # test questions, full pipeline
```

Chat commands: `/citations` (references with the verified quotes), `/evidence`
(all retrieved chunks), `/filter year>=2024 | year<=Y | topic=<key> |
source=pdf|abstract | clear`, `/new` (forget the conversation), `/help`, `/exit`.

```
question ──> embed (same model as chunks) ──> pgvector search + SQL filters ──> per-paper cap ──> top-K evidence
                                                                                        │
             best similarity < min_similarity ──> "insufficient evidence" (Claude not called)
                                                                                        │
Claude (claude-opus-5-5, structured JSON) <── evidence E1..Ek + question + previous turns
   │
   └─> claims + citations {evidence_id, verbatim quote} ──> validation ──> answer + references from the database
```

| Concern | Implementation (`src/cyberresearch/rag/`) |
|---|---|
| Same embedding model | `retrieval.py` refuses to run if `paper_chunks.embedding_model` differs from the query embedder |
| Metadata filtering | year range, topic, `pdf`/`abstract`, excluded sections (default: `Front Matter`), paper ids - all in SQL |
| Top-K selection | 40 nearest candidates → at most 3 chunks per paper → best 8 |
| Reranking | **Not used** (see below) |
| Grounded prompt | `prompts.py`: evidence in `<evidence id="E3" ...>` blocks with year, section, page and similarity; no URLs or DOIs are shown to Claude; text inside excerpts is treated as data (prompt-injection guard) |
| Structured output | `output_config.format` JSON schema: `answer_status`, `summary`, `claims[{statement, kind: evidence\|interpretation, citations[{evidence_id, quote}]}]`, `ambiguity_note`, `limitations`, `follow_up_suggestions` |
| Citation validation | `citations.py`: the evidence id must be one retrieved for this question **and** the quote (≥ 5 words) must occur in that chunk's stored text. Unverified citations are dropped; an evidence claim with none left is removed and the answer is downgraded |
| Citation format | Built from the database, never from model text: title, authors, year, DOI → arXiv → landing-page URL, section and page (or "abstract"), evidence id → `paper_id`, `chunk_id`, chunk number |
| Arbitrary URLs | Any URL, DOI or arXiv id written by Claude is removed and reported |
| Insufficient evidence | Two layers: a similarity gate before Claude, and Claude's own `insufficient_evidence` / `partial` / `out_of_scope` status; evidence claims are never shown with those statuses |
| Evidence vs interpretation | Separate sections in the answer: "Findings from the papers" (cited) and "Interpretation" (assistant's reasoning) |
| Follow-ups | The previous question is added to the retrieval query; the last 3 turns are given to Claude as context, marked "not evidence" |
| Refusals | `fallbacks: "default"` (beta `server-side-fallback-2026-07-01`) re-runs a declined request on the model Anthropic recommends for that category; a final refusal is shown as "declined", never as an answer |

**Test questions** (`config/rag_test_questions.yaml`): 15 questions on network
security, intrusion detection and AI for cybersecurity, in four groups -
sufficient evidence, weak evidence, outside the corpus, ambiguous terms (one is
in Vietnamese). Latest run with the local model: **15/15 passed**, see
[docs/rag_eval_results.md](docs/rag_eval_results.md). The Claude API path is
covered by unit tests with a fake client; it has not been run live yet.

**Why no reranker.** A cross-encoder reranker would add a second model
download and per-question latency. On the test questions the bi-encoder
ranking with the per-paper cap already puts on-topic chunks first, and Claude
discards irrelevant excerpts itself (it must quote what it cites). Reranking
becomes worth measuring when the corpus grows or the Phase 4 evaluation shows
relevant chunks ranked below the top 8.

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
  rag/
    retrieval.py              Query embedding, filters, top-K, diversity
    prompts.py                System prompt, evidence blocks, JSON schema
    llm.py                    Claude API client (structured output, fallback)
    ollama_client.py          Local Ollama client (same interface)
    citations.py              Citation mapping and validation
    engine.py                 Retrieve -> gate -> generate -> validate
    chat.py                   Terminal chatbot
    evaluation.py             Test-question runner and pass rules
config/rag_test_questions.yaml  Test questions (sufficient/weak/outside/ambiguous)
docs/benchmark_results.md     Latest benchmark report (generated)
tests/                        Unit tests (+ tests/live for real APIs)
main.py                       Entry point
```

## Database

The default is PostgreSQL in Docker (see [Docker](#docker-postgresql--application)),
selected by `DATABASE_URL` in `.env`. If `DATABASE_URL` is empty, development
and tests fall back to an embedded PostgreSQL 16 + pgvector instance provided by
the `pgserver` package and stored in `data/pgdata/` (git-ignored). On Windows
the embedded server can get stuck in crash recovery because its log file lives
inside the data directory; Docker avoids this.

**Vector search accuracy.** pgvector's HNSW index is approximate and SQL
filters are applied after it, so retrieval sets `hnsw.ef_search = 200`
(`rag.hnsw_ef_search`). With the default of 40 one test question lost its best
chunk (similarity 0.379 instead of 0.414 from an exact search); with 200 the
results match exact search on this corpus.

## Branch strategy

- `main` - stable, presentation-ready releases
- `develop` - integration branch; every feature merges here first
- `feature/<name>` - one branch per phase (`feature/data-ingestion`,
  `feature/pdf-processing`, `feature/rag-chatbot`, `feature/web-interface`)
- `fix/<name>` - bug fixes, branched from `develop`
