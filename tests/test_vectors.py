"""Embedding dimensions, pgvector storage/retrieval and the processing pipeline."""

import math
from dataclasses import replace
from pathlib import Path

import psycopg
import pytest

from cyberresearch.config import EmbeddingSettings, ProcessingSettings
from cyberresearch.ingestion.schema import PaperRecord
from cyberresearch.processing.chunking import Chunk
from cyberresearch.processing.downloader import DownloadResult
from cyberresearch.processing.pipeline import ProcessingPipeline, abstract_chunks
from cyberresearch.processing.store import ChunkStore, parse_vector, to_vector_literal
from fakes import HashEmbedder
from pdf_factory import make_paper_pdf

DIM = 384


def chunk(i: int, text: str, page: int | None = 1) -> Chunk:
    return Chunk(i, text, "1 Introduction", page, page, len(text.split()))


@pytest.fixture
def chunk_store(store) -> ChunkStore:
    cs = ChunkStore(store.conn, DIM)
    cs.init_schema()
    return cs


def add_paper(store, source_id: str, title: str, **kw) -> int:
    store.upsert_record(PaperRecord(source="openalex", source_id=source_id, openalex_id=source_id, title=title, **kw))
    return store.conn.execute("SELECT paper_id FROM papers WHERE openalex_id = %s", (source_id,)).fetchone()["paper_id"]


# --------------------------------------------------------------- embedding

def test_vector_literal_roundtrip() -> None:
    v = [0.1, -0.25, 1e-8]
    assert parse_vector(to_vector_literal(v)) == pytest.approx(v)


@pytest.mark.model
def test_real_model_dimension_and_normalization() -> None:
    from cyberresearch.processing.embeddings import SentenceTransformerEmbedder

    embedder = SentenceTransformerEmbedder(EmbeddingSettings())
    vectors = embedder.embed(["intrusion detection with deep learning", "network IDS using neural networks",
                              "recipe for chocolate cake"])
    assert embedder.dimension == DIM and all(len(v) == DIM for v in vectors)
    assert all(math.isclose(sum(x * x for x in v), 1.0, rel_tol=1e-4) for v in vectors)
    dot = lambda a, b: sum(x * y for x, y in zip(a, b))  # noqa: E731
    assert dot(vectors[0], vectors[1]) > dot(vectors[0], vectors[2])  # semantics, not just words


@pytest.mark.model
def test_real_model_dimension_mismatch_is_reported() -> None:
    from cyberresearch.processing.embeddings import SentenceTransformerEmbedder

    with pytest.raises(ValueError, match="384-d"):
        SentenceTransformerEmbedder(EmbeddingSettings(dimension=768))


# ----------------------------------------------------------------- storage

@pytest.mark.db
def test_schema_creates_pgvector_tables(chunk_store: ChunkStore) -> None:
    chunk_store.init_schema()  # idempotent
    ext = chunk_store.conn.execute("SELECT extname FROM pg_extension WHERE extname = 'vector'").fetchone()
    assert ext is not None
    dim = chunk_store.conn.execute(
        "SELECT format_type(atttypid, atttypmod) AS t FROM pg_attribute "
        "WHERE attrelid = 'paper_chunks'::regclass AND attname = 'embedding'"
    ).fetchone()["t"]
    assert dim == "vector(384)"


@pytest.mark.db
def test_store_and_retrieve_nearest_chunks(store, chunk_store: ChunkStore) -> None:
    embedder = HashEmbedder()
    ids_paper = add_paper(store, "W1", "IDS paper", year=2024)
    other_paper = add_paper(store, "W2", "Cooking paper", year=2023)
    texts_a = ["intrusion detection on network traffic", "anomaly based intrusion alerts"]
    texts_b = ["baking bread with sourdough starter"]
    chunk_store.replace_chunks(ids_paper, [chunk(i, t, i + 1) for i, t in enumerate(texts_a)],
                               embedder.embed(texts_a), "pdf", "https://x/a.pdf", embedder.model_name)
    chunk_store.replace_chunks(other_paper, [chunk(0, texts_b[0], 3)], embedder.embed(texts_b),
                               "pdf", "https://x/b.pdf", embedder.model_name)

    results = chunk_store.search(embedder.embed(["network intrusion detection"])[0], k=3)
    assert results[0]["paper_id"] == ids_paper and results[0]["text"] == texts_a[0]
    assert results[0]["title"] == "IDS paper" and results[0]["page_start"] == 1
    assert results[0]["similarity"] > results[-1]["similarity"]
    assert results[-1]["paper_id"] == other_paper

    stored = chunk_store.get_embedding(results[0]["chunk_id"])
    assert len(stored) == DIM
    assert stored == pytest.approx(embedder.embed([texts_a[0]])[0], abs=1e-6)


@pytest.mark.db
def test_replace_chunks_is_idempotent(store, chunk_store: ChunkStore) -> None:
    paper_id = add_paper(store, "W1", "P")
    emb = HashEmbedder()
    for _ in range(2):
        chunk_store.replace_chunks(paper_id, [chunk(0, "a b c")], emb.embed(["a b c"]), "pdf", None, emb.model_name)
    assert chunk_store.conn.execute("SELECT count(*) AS n FROM paper_chunks").fetchone()["n"] == 1


@pytest.mark.db
def test_pdf_chunk_without_page_is_rejected(store, chunk_store: ChunkStore) -> None:
    paper_id = add_paper(store, "W1", "P")
    emb = HashEmbedder()
    with pytest.raises(psycopg.errors.CheckViolation):
        with chunk_store.transaction():
            chunk_store.replace_chunks(paper_id, [chunk(0, "x y", page=None)], emb.embed(["x y"]), "pdf", None, "m")


@pytest.mark.db
def test_wrong_dimension_is_rejected(store, chunk_store: ChunkStore) -> None:
    paper_id = add_paper(store, "W1", "P")
    with pytest.raises(psycopg.errors.DataException):
        with chunk_store.transaction():
            chunk_store.replace_chunks(paper_id, [chunk(0, "x")], [[0.1] * 10], "pdf", None, "m")


# ---------------------------------------------------------------- pipeline

class FakeDownloader:
    """Pretends to download by copying a generated PDF."""

    def __init__(self, pdf: Path):
        self.pdf = pdf
        self.calls = []

    def download(self, paper_id, candidate):
        self.calls.append((paper_id, candidate.url))
        return DownloadResult("downloaded", file_path=self.pdf, sha256="0" * 64, size_bytes=self.pdf.stat().st_size,
                              http_status=200)


@pytest.mark.db
def test_pipeline_end_to_end(store, chunk_store: ChunkStore, tmp_path: Path) -> None:
    pdf = make_paper_pdf(tmp_path / "paper.pdf")
    arxiv_paper = add_paper(store, "W1", "Deep Learning for Network Intrusion Detection", arxiv_id="2401.01234",
                            abstract="We study IDS.")
    closed_paper = add_paper(store, "W2", "Closed paper", abstract="Only the abstract is available here " * 3)
    no_text_paper = add_paper(store, "W3", "No abstract, no PDF")

    settings = replace(ProcessingSettings(), chunk_max_words=80, chunk_overlap_words=15)
    downloader = FakeDownloader(pdf)
    totals = ProcessingPipeline(chunk_store, HashEmbedder(), settings, downloader).run()
    assert totals["papers"] == 3 and totals["errors"] == 0
    assert downloader.calls == [(arxiv_paper, "https://arxiv.org/pdf/2401.01234")]  # only the allowed one

    rows = chunk_store.conn.execute("SELECT * FROM paper_chunks ORDER BY paper_id, chunk_index").fetchall()
    pdf_chunks = [r for r in rows if r["paper_id"] == arxiv_paper]
    assert pdf_chunks and all(r["content_source"] == "pdf" for r in pdf_chunks)
    assert all(1 <= r["page_start"] <= r["page_end"] <= 4 for r in pdf_chunks)  # real pages of the PDF
    assert all(r["source_url"] == "https://arxiv.org/pdf/2401.01234" for r in pdf_chunks)
    assert not any("References" in (r["section"] or "") for r in pdf_chunks)

    [abstract_chunk] = [r for r in rows if r["paper_id"] == closed_paper]
    assert abstract_chunk["content_source"] == "abstract" and abstract_chunk["page_start"] is None

    docs = {r["paper_id"]: r for r in chunk_store.conn.execute("SELECT * FROM pdf_documents")}
    assert docs[arxiv_paper]["pdf_status"] == "downloaded" and docs[arxiv_paper]["page_count"] == 4
    assert docs[arxiv_paper]["extraction_status"] == "extracted"
    assert docs[closed_paper]["access_reason"] == "no_pdf_url" and docs[closed_paper]["pdf_status"] == "not_allowed"
    assert docs[no_text_paper]["content_source"] == "none" and docs[no_text_paper]["chunk_count"] == 0

    # Incremental: a second run finds nothing left to process.
    again = ProcessingPipeline(chunk_store, HashEmbedder(), settings, downloader).run()
    assert again["papers"] == 0 and len(downloader.calls) == 1

    # Re-chunking already downloaded PDFs keeps the same result, no downloads.
    redo = ProcessingPipeline(chunk_store, HashEmbedder(), settings, downloader).reprocess_downloaded()
    assert redo["papers"] == 1 and redo["chunks"] == len(pdf_chunks) and len(downloader.calls) == 1
    doc = chunk_store.conn.execute("SELECT * FROM pdf_documents WHERE paper_id = %s", (arxiv_paper,)).fetchone()
    assert doc["pdf_status"] == "downloaded" and doc["access_reason"] == "arxiv"


@pytest.mark.db
def test_pipeline_without_download_uses_abstracts(store, chunk_store: ChunkStore) -> None:
    add_paper(store, "W1", "ArXiv paper", arxiv_id="2401.00001", abstract="An abstract about intrusion detection.")
    totals = ProcessingPipeline(chunk_store, HashEmbedder(), ProcessingSettings(), downloader=None).run()
    assert totals["pdf:skipped_download"] == 1 and totals["content:abstract"] == 1


def test_abstract_chunks_have_no_page() -> None:
    [c] = abstract_chunks("word " * 50, 220, 40)
    assert c.section == "Abstract" and c.page_start is None and c.page_end is None
    assert abstract_chunks(None, 220, 40) == []
