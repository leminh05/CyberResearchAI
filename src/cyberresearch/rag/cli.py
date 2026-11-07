"""Command-line commands for Phase 3 (wired into ``main.py``)."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from cyberresearch.config import Settings
from cyberresearch.db import resolve_database_url
from cyberresearch.ingestion.storage import PostgresStore

RAG_COMMANDS = {"chat", "ask", "rag-eval"}


def add_rag_commands(subparsers: argparse._SubParsersAction) -> None:
    def add_filters(p: argparse.ArgumentParser) -> None:
        p.add_argument("--provider", choices=["claude", "ollama", "groq"], help="answer generator (default: settings.yaml)")
        p.add_argument("--year-min", type=int)
        p.add_argument("--year-max", type=int)
        p.add_argument("--topic", nargs="+", help="topic keys, e.g. intrusion_detection")
        p.add_argument("--source", nargs="+", choices=["pdf", "abstract"])

    chat = subparsers.add_parser("chat", help="interactive research chatbot")
    add_filters(chat)
    ask = subparsers.add_parser("ask", help="answer one question and exit")
    ask.add_argument("question")
    ask.add_argument("--json", action="store_true", help="print the validated answer as JSON")
    add_filters(ask)
    ev = subparsers.add_parser("rag-eval", help="run the test questions")
    ev.add_argument("--retrieval-only", action="store_true", help="no LLM calls; similarity and gate only")
    ev.add_argument("--provider", choices=["claude", "ollama", "groq"], help="answer generator (default: settings.yaml)")
    ev.add_argument("--ids", nargs="+", help="run only these question ids")


def _filters(args: argparse.Namespace):
    from cyberresearch.rag.retrieval import SearchFilters

    return SearchFilters(
        year_min=getattr(args, "year_min", None),
        year_max=getattr(args, "year_max", None),
        topics=getattr(args, "topic", None) or [],
        content_sources=getattr(args, "source", None) or [],
    )


def run_rag_command(args: argparse.Namespace, settings: Settings) -> int:
    from cyberresearch.processing.embeddings import SentenceTransformerEmbedder
    from cyberresearch.processing.store import ChunkStore
    from cyberresearch.rag.retrieval import EmbeddingModelMismatch, Retriever

    # Windows consoles default to a legacy code page; answers may be Vietnamese.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    store = PostgresStore.connect(resolve_database_url(settings))
    try:
        ChunkStore(store.conn, settings.embedding.dimension).init_schema()  # databases older than hybrid search
        retriever = Retriever(store.conn, SentenceTransformerEmbedder(settings.embedding), settings.rag)
        try:
            retriever.check_embedding_model()
        except EmbeddingModelMismatch as exc:
            print(f"Embedding model mismatch: {exc}. Re-run `python main.py process --reprocess-pdfs`.")
            return 2

        if args.command == "rag-eval" and args.retrieval_only:
            from cyberresearch.rag.evaluation import load_questions, retrieval_report

            for row in retrieval_report(retriever, load_questions(), settings.rag.min_similarity):
                print(f"{row['id']:<18}{row['category']:<12}{row['best_similarity']:>6.2f}  {row['gate']}")
                for title in row["top_titles"]:
                    print(f"{'':<20}{title}")
            return 0

        from cyberresearch.rag.engine import RAGEngine

        provider = getattr(args, "provider", None) or settings.rag.provider
        if provider == "groq":
            from cyberresearch.rag.groq_client import GroqGenerator, key_is_set

            if not key_is_set():
                print("GROQ_API_KEY is not set. Add it to .env (see .env.example), or use the free local model: --provider ollama")
                return 2
            generator = GroqGenerator(settings.rag)
            print(f"Using Groq model {settings.rag.groq_model}: questions and excerpts are sent to Groq.")
        elif provider == "ollama":
            from cyberresearch.rag.ollama_client import OllamaGenerator

            generator = OllamaGenerator(settings.rag)
            print(f"Using local Ollama model {settings.rag.ollama_model} (answers take a few minutes on CPU).")
        else:
            from cyberresearch.rag.llm import ClaudeClient

            if not settings.anthropic_api_key_present:
                print("ANTHROPIC_API_KEY is not set. Add it to .env (see .env.example), "
                      "or use the free local model: --provider ollama")
                return 2
            generator = ClaudeClient(settings.rag)
        engine = RAGEngine(retriever, generator, settings.rag, translator=generator)

        if args.command == "chat":
            from cyberresearch.rag.chat import run_chat

            run_chat(engine, _filters(args))
            return 0
        if args.command == "ask":
            from cyberresearch.rag.chat import render_answer

            response = engine.ask(args.question, _filters(args))
            if args.json:
                payload = asdict(response.answer)
                payload["evidence"] = [asdict(e) for e in response.evidence]
                print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
            else:
                print(render_answer(response))
            return 0 if response.answer.status not in ("error", "refused") else 1

        from cyberresearch.rag.evaluation import evaluate, format_rows, load_questions

        questions = [q for q in load_questions() if not args.ids or q["id"] in args.ids]
        rows = evaluate(
            engine,
            questions,
            on_row=lambda r: print(f"  done {r.id}: {r.status} ({'pass' if r.passed else 'FAIL'})", flush=True),
        )
        print(format_rows(rows))
        return 0 if all(r.passed for r in rows) else 1
    finally:
        store.close()
