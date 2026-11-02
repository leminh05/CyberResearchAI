"""Phase 2 pipeline: access check -> download -> extract -> chunk -> embed -> store.

Papers without a legally downloadable PDF are not dropped: their abstract
(from the metadata APIs) becomes a single ``abstract`` chunk without a page
number, clearly marked as such.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from cyberresearch.config import ProcessingSettings
from cyberresearch.processing.access import decide_access
from cyberresearch.processing.chunking import Chunk, chunk_paragraphs
from cyberresearch.processing.cleaning import clean_pages
from cyberresearch.processing.downloader import PdfDownloader
from cyberresearch.processing.embeddings import Embedder
from cyberresearch.processing.extract import PdfExtractionError, extract_pages
from cyberresearch.processing.sections import Paragraph, section_name, segment
from cyberresearch.processing.store import ChunkStore

logger = logging.getLogger(__name__)


def embedding_input(title: str | None, chunk: Chunk) -> str:
    """Text sent to the embedding model: title and section give context."""
    return f"{title or ''}. {section_name(chunk.section)}. {chunk.text}"


def abstract_chunks(abstract: str | None, max_words: int, overlap_words: int) -> list[Chunk]:
    """Chunks for papers processed from their abstract only (no page numbers)."""
    if not abstract:
        return []
    # min_words=1: a short abstract is still real content (unlike PDF fragments).
    chunks = chunk_paragraphs([Paragraph(0, "Abstract", abstract)], max_words, overlap_words, min_words=1)
    for chunk in chunks:
        chunk.page_start = chunk.page_end = None  # an abstract has no PDF page
    return chunks


class ProcessingPipeline:
    """Processes papers that have not been processed yet (incremental)."""

    def __init__(
        self,
        store: ChunkStore,
        embedder: Embedder,
        settings: ProcessingSettings,
        downloader: PdfDownloader | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.settings = settings
        self.downloader = downloader  # None = do not download, use abstracts/known files only

    def pdf_chunks(self, status: dict[str, Any]) -> list[Chunk]:
        """Extract, clean, segment and chunk a downloaded PDF; updates ``status``."""
        try:
            pages = extract_pages(status["file_path"])
        except PdfExtractionError as exc:
            status.update(extraction_status="extraction_failed", error=str(exc))
            return []
        paragraphs = segment(clean_pages(pages))
        chunks = chunk_paragraphs(
            paragraphs,
            self.settings.chunk_max_words,
            self.settings.chunk_overlap_words,
            self.settings.skip_sections,
        )
        status.update(extraction_status="extracted", page_count=len(pages))
        return chunks

    def process_paper(self, paper: dict[str, Any]) -> dict[str, Any]:
        """Process one paper and store its status and chunks. Returns the status."""
        decision = decide_access(paper, paper.get("sources") or [], self.settings)
        candidate = decision.candidate
        status: dict[str, Any] = {
            "access_reason": decision.reason,
            "pdf_url": candidate.url if candidate else None,
            "pdf_origin": candidate.origin if candidate else None,
            "license": candidate.license if candidate else None,
        }

        chunks: list[Chunk] = []
        if not decision.allowed:
            status["pdf_status"] = "not_allowed"
        elif self.downloader is None:
            status["pdf_status"] = "skipped_download"
        else:
            result = self.downloader.download(paper["paper_id"], candidate)
            status.update(
                pdf_status=result.status,
                http_status=result.http_status,
                file_path=str(result.file_path) if result.file_path else None,
                sha256=result.sha256,
                size_bytes=result.size_bytes,
                error=result.error,
            )
            if result.status == "downloaded":
                chunks = self.pdf_chunks(status)

        return self._store(paper, status, chunks, attempts=paper.get("attempts", 0) + 1)

    def _store(self, paper: dict[str, Any], status: dict[str, Any], chunks: list[Chunk], attempts: int) -> dict[str, Any]:
        """Fall back to the abstract if needed, embed, and store chunks + status atomically."""
        content_source, source_url = "pdf", status.get("pdf_url")
        if not chunks:
            chunks = abstract_chunks(
                paper.get("abstract"), self.settings.chunk_max_words, self.settings.chunk_overlap_words
            )
            content_source, source_url = "abstract", paper.get("url")

        embeddings = self.embedder.embed([embedding_input(paper.get("title"), c) for c in chunks])
        status.update(content_source=content_source if chunks else "none", chunk_count=len(chunks))
        with self.store.transaction():
            self.store.replace_chunks(
                paper["paper_id"], chunks, embeddings, content_source, source_url, self.embedder.model_name
            )
            self.store.record_document(paper["paper_id"], attempts, **status)
        return status

    def reprocess_downloaded(self) -> Counter:
        """Re-extract and re-chunk PDFs already on disk (e.g. after rule changes).

        Nothing is downloaded again; access and download fields are kept.
        """
        totals: Counter = Counter()
        for row in self.store.downloaded_documents():
            status = {k: row[k] for k in ("access_reason", "pdf_status", "pdf_url", "pdf_origin", "license",
                                          "http_status", "file_path", "sha256", "size_bytes")}
            chunks = self.pdf_chunks(status)
            status = self._store(row, status, chunks, attempts=row["attempts"])
            totals["papers"] += 1
            totals[f"content:{status['content_source']}"] += 1
            totals["chunks"] += status["chunk_count"]
        return totals

    def run(self, limit: int | None = None, retry_failed: bool = False) -> Counter:
        """Process up to ``limit`` unprocessed papers; returns summary counts."""
        totals: Counter = Counter()
        for paper in self.store.papers_to_process(limit, retry_failed):
            try:
                status = self.process_paper(paper)
            except Exception:
                totals["errors"] += 1
                logger.exception("paper processing failed", extra={"paper_id": paper["paper_id"]})
                continue
            totals["papers"] += 1
            totals[f"pdf:{status['pdf_status']}"] += 1
            totals[f"content:{status['content_source']}"] += 1
            totals["chunks"] += status["chunk_count"]
            logger.info(
                "paper processed",
                extra={"paper_id": paper["paper_id"], "pdf_status": status["pdf_status"],
                       "chunks": status["chunk_count"], "content_source": status["content_source"]},
            )
        return totals
