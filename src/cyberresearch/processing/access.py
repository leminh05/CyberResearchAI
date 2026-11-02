"""Decide whether a paper's PDF may be downloaded.

Policy (conservative, documented in README):
  1. arXiv papers are allowed (arXiv grants a distribution license; its
     robots.txt allows /pdf/ on arxiv.org with a 15 s crawl delay, while
     export.arxiv.org disallows /pdf/ - verified 2026-11).
  2. Other PDFs are allowed only with an explicit open license from the
     configured allow-list (Creative Commons / public domain).
  3. Unknown or custom licenses are skipped unless explicitly enabled.
A URL alone is never treated as permission.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

from cyberresearch.config import ProcessingSettings

ARXIV_PDF_URL = "https://arxiv.org/pdf/{arxiv_id}"


@dataclass(frozen=True)
class PdfCandidate:
    """A possible PDF location for a paper and the license that came with it."""

    url: str
    origin: str  # arxiv | openalex | semantic_scholar
    license: str | None


@dataclass(frozen=True)
class AccessDecision:
    """Result of the access check; ``reason`` is stored as the PDF status."""

    allowed: bool
    reason: str
    candidate: PdfCandidate | None


def normalize_license(value: str | None) -> str | None:
    """Map license strings/URLs to short ids such as ``cc-by-nc``.

    Examples: ``"CC-BY"`` -> ``cc-by``;
    ``"https://creativecommons.org/licenses/by-nc/4.0/"`` -> ``cc-by-nc``.
    """
    if not value:
        return None
    text = value.strip().lower()
    match = re.search(r"creativecommons\.org/licenses/([a-z\-]+)", text)
    if match:
        return f"cc-{match.group(1)}"
    if "publicdomain/zero" in text or text in ("cc0", "cc-0"):
        return "cc0"
    if "publicdomain" in text or text in ("public-domain", "pd"):
        return "public-domain"
    return text.replace("_", "-").replace(" ", "-")


def pdf_candidates(paper: Mapping[str, Any], sources: list[Mapping[str, Any]]) -> list[PdfCandidate]:
    """Collect PDF locations from the paper and its original source payloads.

    The license is read from the same API object as the URL, so a license is
    never attached to a different copy of the paper.
    """
    candidates: list[PdfCandidate] = []
    if paper.get("arxiv_id"):
        candidates.append(PdfCandidate(ARXIV_PDF_URL.format(arxiv_id=paper["arxiv_id"]), "arxiv", "arxiv"))
    for source in sources:
        raw = source.get("raw") or {}
        if source.get("source") == "openalex":
            location = raw.get("best_oa_location") or {}
            url, license_ = location.get("pdf_url"), location.get("license")
        elif source.get("source") == "semantic_scholar":
            pdf = raw.get("openAccessPdf") or {}
            url, license_ = pdf.get("url"), pdf.get("license")
        else:
            continue
        if url and all(c.url != url for c in candidates):
            candidates.append(PdfCandidate(url, source["source"], license_))
    return candidates


def decide_access(
    paper: Mapping[str, Any], sources: list[Mapping[str, Any]], settings: ProcessingSettings
) -> AccessDecision:
    """Return the first candidate that satisfies the access policy."""
    candidates = pdf_candidates(paper, sources)
    if not candidates:
        return AccessDecision(False, "no_pdf_url", None)

    allowed = set(settings.allowed_licenses)
    saw_unknown = False
    for candidate in candidates:
        if candidate.origin == "arxiv":
            return AccessDecision(True, "arxiv", candidate)
        license_id = normalize_license(candidate.license)
        if license_id in allowed:
            return AccessDecision(True, f"license:{license_id}", candidate)
        if license_id in (None, "other-oa", "unspecified-oa", "publisher-specific-oa"):
            saw_unknown = True
            if settings.allow_unknown_license:
                return AccessDecision(True, "unknown_license_allowed", candidate)
    reason = "license_unknown" if saw_unknown else "license_not_allowed"
    return AccessDecision(False, reason, candidates[0])
