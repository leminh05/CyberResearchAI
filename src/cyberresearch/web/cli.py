"""The ``serve`` command (wired into ``main.py``)."""

from __future__ import annotations

import argparse

from cyberresearch.config import PROJECT_ROOT, Settings
from cyberresearch.db import resolve_database_url
from cyberresearch.ingestion.storage import PostgresStore


def add_web_commands(subparsers: argparse._SubParsersAction) -> None:
    serve = subparsers.add_parser("serve", help="web API and chat website")
    serve.add_argument("--provider", choices=["claude", "ollama", "gemini"], help="answer generator (default: settings.yaml)")
    # Loopback by default: the API has no user accounts.
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)


def run_web_command(args: argparse.Namespace, settings: Settings) -> int:
    import uvicorn

    from cyberresearch.processing.embeddings import SentenceTransformerEmbedder
    from cyberresearch.processing.store import ChunkStore
    from cyberresearch.rag.retrieval import EmbeddingModelMismatch, Retriever
    from cyberresearch.web.app import Services, create_app

    provider = args.provider or settings.rag.provider
    if provider == "gemini":
        from cyberresearch.rag.gemini_client import GeminiGenerator, key_is_set

        if not key_is_set():
            print("GEMINI_API_KEY is not set. Add it to .env (see .env.example), or use the free local model: --provider ollama")
            return 2
        generator = GeminiGenerator(settings.rag)
    elif provider == "ollama":
        from cyberresearch.rag.ollama_client import OllamaGenerator

        generator = OllamaGenerator(settings.rag)
    else:
        from cyberresearch.rag.llm import ClaudeClient

        if not settings.anthropic_api_key_present:
            print("ANTHROPIC_API_KEY is not set. Add it to .env (see .env.example), "
                  "or use the free local model: --provider ollama")
            return 2
        generator = ClaudeClient(settings.rag)

    database_url = resolve_database_url(settings)

    def connect():
        return PostgresStore.connect(database_url).conn

    embedder = SentenceTransformerEmbedder(settings.embedding)
    with connect() as conn:
        ChunkStore(conn, settings.embedding.dimension).init_schema()  # databases older than hybrid search
        try:
            Retriever(conn, embedder, settings.rag).check_embedding_model()
        except EmbeddingModelMismatch as exc:
            print(f"Embedding model mismatch: {exc}. Re-run `python main.py process --reprocess-pdfs`.")
            return 2

    dist = PROJECT_ROOT / "frontend" / "dist"
    if not dist.is_dir():
        print("frontend/dist not found: serving the API only. Build the website with `npm run build` in frontend/.")
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print(f"WARNING: listening on {args.host}. The API has no login; put authentication in front of it.")
    app = create_app(Services(settings, provider, generator, embedder, connect), dist)
    # One process only: conversations and limits live in memory.
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0
