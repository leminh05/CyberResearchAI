"""FastAPI application: a thin HTTP layer over the existing RAG engine.

All retrieval, generation and citation validation stays in ``cyberresearch.rag``.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from uuid import UUID

import psycopg
from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from cyberresearch.config import Settings
from cyberresearch.rag.engine import RAGEngine
from cyberresearch.rag.retrieval import Retriever
from cyberresearch.web.limits import ConversationStore, RateLimiter
from cyberresearch.web.schemas import ChatRequest, ChatResponse, to_chat_response

logger = logging.getLogger(__name__)

MAX_BODY_BYTES = 64 * 1024
CONTENT_SECURITY_POLICY = (
    "default-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
)
# Swagger UI loads its assets from a CDN, which the policy above would block.
NO_CSP_PATHS = ("/docs", "/redoc")


@dataclass
class Services:
    """What the API needs; built once at startup, replaced by fakes in tests."""

    settings: Settings
    provider: str
    generator: Any  # AnswerGenerator + QueryTranslator + is_ready()
    embedder: Any
    connect: Callable[[], psycopg.Connection]  # a new connection per request


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, retry_after: int | None = None) -> None:
        self.status, self.code, self.message, self.retry_after = status, code, message, retry_after


def _error(status: int, code: str, message: str, headers: dict[str, str] | None = None) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message}}, status_code=status, headers=headers)


def create_app(services: Services, frontend_dist: Path | None = None) -> FastAPI:
    settings = services.settings
    app = FastAPI(title="CyberResearchAI", version="1.0")
    limiter = RateLimiter(settings.web.rate_limit_per_minute)
    slots = threading.BoundedSemaphore(settings.web.max_concurrent_requests)
    conversations = ConversationStore(
        lambda: RAGEngine(None, services.generator, settings.rag, translator=services.generator),
        settings.web.max_conversations,
        settings.web.conversation_ttl_minutes * 60,
    )

    @app.middleware("http")
    async def limit_body_and_set_headers(request: Request, call_next) -> Response:
        # shortcut: trusts Content-Length, so put a reverse proxy with a body limit in front of a public deployment.
        length = request.headers.get("content-length", "0")
        if length.isdigit() and int(length) > MAX_BODY_BYTES:
            return _error(413, "body_too_large", "The request is too large.")
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        if not request.url.path.startswith(NO_CSP_PATHS):
            response.headers["Content-Security-Policy"] = CONTENT_SECURITY_POLICY
        return response

    @app.exception_handler(ApiError)
    async def on_api_error(_: Request, exc: ApiError) -> JSONResponse:
        headers = {"Retry-After": str(exc.retry_after)} if exc.retry_after else None
        return _error(exc.status, exc.code, exc.message, headers)

    @app.exception_handler(RequestValidationError)
    async def on_invalid_request(_: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0]
        field = ".".join(str(part) for part in first["loc"] if part != "body")
        return _error(422, "invalid_request", f"{field}: {first['msg']}" if field else first["msg"])

    @app.exception_handler(Exception)
    async def on_unexpected_error(_: Request, exc: Exception) -> JSONResponse:
        # Details go to the log only: exception text can contain internal addresses.
        logger.error("unhandled web error", exc_info=exc)
        return _error(500, "internal_error", "Something went wrong on the server.")

    @app.get("/api/health")
    def health(response: Response) -> dict[str, Any]:
        papers = chunks = None
        try:
            with services.connect() as conn:
                papers = conn.execute("SELECT count(*) AS n FROM papers").fetchone()["n"]
                chunks = conn.execute("SELECT count(*) AS n FROM paper_chunks").fetchone()["n"]
        except psycopg.Error as exc:
            logger.warning("health check: database unavailable", extra={"error": type(exc).__name__})
        database_ok = papers is not None
        llm_ok = services.generator.is_ready()
        response.status_code = 200 if database_ok else 503
        return {
            "status": "ok" if database_ok and llm_ok else "degraded" if database_ok else "unavailable",
            "database": database_ok,
            "llm": llm_ok,
            "provider": services.provider,
            "papers": papers,
            "chunks": chunks,
        }

    # A plain "def": FastAPI runs it in a worker thread, so a slow LLM call
    # does not block other requests.
    @app.post("/api/chat", response_model=ChatResponse)
    def chat(body: ChatRequest, request: Request) -> ChatResponse:
        # shortcut: behind a reverse proxy every user shares the proxy's address and therefore one limit.
        wait = limiter.retry_after(request.client.host if request.client else "unknown")
        if wait:
            raise ApiError(429, "rate_limited", f"Too many questions. Try again in {wait} s.", wait)
        if not slots.acquire(blocking=False):
            raise ApiError(503, "busy", "The server is answering other questions. Try again shortly.", 30)
        try:
            if body.conversation_id:
                conversation_id, conversation = body.conversation_id, conversations.get(body.conversation_id)
                if conversation is None:
                    raise ApiError(404, "conversation_not_found", "This conversation has expired. Start a new one.")
            else:
                conversation_id, conversation = conversations.create()
            if not conversation.lock.acquire(blocking=False):
                raise ApiError(409, "conversation_busy", "This conversation is still answering the previous question.")
            try:
                with services.connect() as conn:
                    conversation.engine.retriever = Retriever(conn, services.embedder, settings.rag)
                    rag_response = conversation.engine.ask(body.question)
            except psycopg.Error as exc:
                logger.error("chat: database error", extra={"error": type(exc).__name__})
                raise ApiError(503, "database_unavailable", "The paper database is not reachable.") from exc
            finally:
                conversation.lock.release()
        finally:
            slots.release()
        # The engine's error text can contain internal addresses; it logged the details already.
        if rag_response.answer.status == "error":
            raise ApiError(502, "llm_failed", "The answer could not be generated. Try again.")
        return to_chat_response(rag_response, conversation_id)

    @app.delete("/api/conversations/{conversation_id}", status_code=204)
    def delete_conversation(conversation_id: UUID) -> None:
        conversations.delete(conversation_id)  # idempotent: unknown ids are already gone

    if frontend_dist and frontend_dist.is_dir():
        app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
    return app
