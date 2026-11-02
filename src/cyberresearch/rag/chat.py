"""Terminal chatbot and plain-text rendering of answers."""

from __future__ import annotations

import sys
from typing import Callable, TextIO

from cyberresearch.rag.engine import RAGEngine, RAGResponse
from cyberresearch.rag.retrieval import SearchFilters

STATUS_LABELS = {
    "answered": "ANSWERED - supported by retrieved evidence",
    "partial": "PARTIAL - only part of the question is supported",
    "insufficient_evidence": "INSUFFICIENT EVIDENCE - no supported answer",
    "out_of_scope": "OUT OF SCOPE - not a cybersecurity research question",
    "refused": "DECLINED by the model",
    "error": "ERROR",
}

HELP = """\
Commands:
  <question>            ask a cybersecurity research question (follow-ups keep context)
  /citations            full references and verified quotes of the last answer
  /evidence             all retrieved chunks for the last question
  /filter year>=2024    filters: year>=Y, year<=Y, topic=<key>, source=pdf|abstract, clear
  /filters              show active filters
  /new                  start a new conversation (forget previous questions)
  /help                 this help
  /exit                 quit (also: exit, quit, Ctrl+C)"""


def render_answer(response: RAGResponse) -> str:
    a = response.answer
    lines = [f"Status: {STATUS_LABELS.get(a.status, a.status)}"]
    meta = f"evidence: {len(response.evidence)} chunks, best similarity {response.best_similarity:.2f}"
    if response.llm and response.llm.model:
        meta += f", model {response.llm.model}"
    if response.gated:
        meta += ", Claude not called"
    lines.append(f"({meta})")
    if a.summary:
        lines += ["", a.summary]
    evidence_claims = [c for c in a.claims if c.kind == "evidence"]
    interpretations = [c for c in a.claims if c.kind == "interpretation"]
    if evidence_claims:
        lines += ["", "Findings from the papers:"]
        for i, claim in enumerate(evidence_claims, 1):
            marks = "".join(f"[{c.reference.number}]" for c in claim.accepted if c.reference)
            lines.append(f"  {i}. {claim.statement} {marks}")
    if interpretations:
        lines += ["", "Interpretation (assistant's reasoning, not a finding of the papers):"]
        for claim in interpretations:
            marks = "".join(f"[{c.reference.number}]" for c in claim.accepted if c.reference)
            lines.append(f"  - {claim.statement} {marks}".rstrip())
    if a.ambiguity_note:
        lines += ["", f"Ambiguous terms: {a.ambiguity_note}"]
    if a.limitations:
        lines += ["", f"Limitations: {a.limitations}"]
    if a.references:
        lines += ["", "References:"] + [f"  {r.format()}" for r in a.references]
    if a.warnings:
        lines += ["", "Validation:"] + [f"  ! {w}" for w in a.warnings]
    if a.follow_ups:
        lines += ["", "You could also ask:"] + [f"  - {s}" for s in a.follow_ups[:3]]
    return "\n".join(lines)


def render_citations(response: RAGResponse) -> str:
    a = response.answer
    if not a.references:
        return "The last answer has no verified citations."
    lines = []
    for ref in a.references:
        lines.append(ref.format())
        quotes = {c.quote for claim in a.claims for c in claim.accepted if c.reference is ref}
        lines += [f'     quote: "{q}"' for q in quotes]
    rejected = a.invalid_citations
    if rejected:
        lines += ["", "Rejected citations (not shown in the answer):"]
        lines += [f"  {c.evidence_id or '?'}: {c.reason} - \"{c.quote[:120]}\"" for c in rejected]
    return "\n".join(lines)


def render_evidence(response: RAGResponse) -> str:
    if not response.evidence:
        return "No evidence was retrieved."
    lines = [f"Retrieval query: {response.retrieval_query}", f"Filters: {response.filters.describe()}"]
    for e in response.evidence:
        lines.append(
            f"{e.evidence_id} [{e.similarity:.2f}] {e.title} ({e.year}) - {e.section or '-'}, {e.location}, "
            f"paper_id {e.paper_id}, chunk_id {e.chunk_id}"
        )
        lines.append(f"     {e.text[:200]}...")
    return "\n".join(lines)


def apply_filter(filters: SearchFilters, arg: str) -> str:
    """Update ``filters`` from a /filter argument; returns a confirmation message."""
    arg = arg.strip().replace(" ", "")
    if arg == "clear":
        filters.__init__()  # reset all fields
        return "Filters cleared."
    try:
        if arg.startswith("year>="):
            filters.year_min = int(arg[6:])
        elif arg.startswith("year<="):
            filters.year_max = int(arg[6:])
        elif arg.startswith("topic="):
            filters.topics = [t for t in arg[6:].split(",") if t]
        elif arg.startswith("source="):
            sources = [s for s in arg[7:].split(",") if s in ("pdf", "abstract")]
            if not sources:
                return "source must be pdf or abstract."
            filters.content_sources = sources
        else:
            return "Unknown filter. Use year>=Y, year<=Y, topic=<key>, source=pdf|abstract or clear."
    except ValueError:
        return "Year must be a number."
    return f"Active filters: {filters.describe()}"


def run_chat(
    engine: RAGEngine,
    filters: SearchFilters | None = None,
    read: Callable[[str], str] = input,
    out: TextIO = sys.stdout,
) -> None:
    """Read-eval-print loop. ``read``/``out`` are injectable for tests."""
    filters = filters or SearchFilters()
    last: RAGResponse | None = None
    print("CyberResearchAI - evidence-grounded cybersecurity research assistant", file=out)
    print("Type a question, /help for commands, /exit to quit.", file=out)
    while True:
        try:
            line = read("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.", file=out)
            return
        if not line:
            continue
        command, _, arg = line.partition(" ")
        command = command.lower()
        if command in ("/exit", "/quit", "exit", "quit"):
            print("Bye.", file=out)
            return
        if command == "/help":
            print(HELP, file=out)
        elif command == "/new":
            engine.reset()
            print("New conversation started.", file=out)
        elif command == "/filter":
            print(apply_filter(filters, arg), file=out)
        elif command == "/filters":
            print(f"Active filters: {filters.describe()}", file=out)
        elif command in ("/citations", "/evidence"):
            if last is None:
                print("Ask a question first.", file=out)
            else:
                print(render_citations(last) if command == "/citations" else render_evidence(last), file=out)
        elif command.startswith("/"):
            print("Unknown command. Type /help.", file=out)
        else:
            print("searching papers and generating an evidence-based answer...", file=out)
            last = engine.ask(line, filters)
            print("\n" + render_answer(last), file=out)
