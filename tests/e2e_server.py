"""Server for the browser end-to-end tests (frontend/e2e).

Everything is real (website, FastAPI, embeddings, pgvector search over the
database in DATABASE_URL, citation validation) except the LLM: this scripted
generator quotes the first retrieved passage, so tests are fast and repeatable.
"""

from __future__ import annotations

import re
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import uvicorn  # noqa: E402

from cyberresearch.config import WebSettings, load_settings  # noqa: E402
from cyberresearch.db import resolve_database_url  # noqa: E402
from cyberresearch.ingestion.storage import PostgresStore  # noqa: E402
from cyberresearch.processing.embeddings import SentenceTransformerEmbedder  # noqa: E402
from cyberresearch.rag.llm import LLMError, LLMResult  # noqa: E402
from cyberresearch.web.app import Services, create_app  # noqa: E402

FIRST_EVIDENCE = re.compile(r'<evidence id="E1"[^>]*>\nPaper: [^\n]*\n(.*?)\n</evidence>', re.DOTALL)
PORT = 8765


class ScriptedGenerator:
    def is_ready(self) -> bool:
        return True

    def translate_query(self, question: str) -> str:
        return question

    def generate(self, system: str, user_message: str, schema: dict) -> LLMResult:
        question = user_message.rsplit("<question>", 1)[-1]
        if "simulate llm failure" in question:
            raise LLMError("scripted failure at http://internal-address:11434")
        passage = FIRST_EVIDENCE.search(user_message).group(1)
        quote = " ".join(passage.split()[:12])
        if "simulate fabricated quote" in question:
            quote = "this sentence was invented by the model and is in no paper"
        return LLMResult(
            {
                "answer_status": "answered",
                "summary": "Scripted summary of the retrieved passage.",
                "claims": [{
                    "statement": "Scripted finding taken from the first retrieved passage.",
                    "kind": "evidence",
                    "citations": [{"evidence_id": "E1", "quote": quote}],
                }],
                "ambiguity_note": "",
                "limitations": "",
                "follow_up_suggestions": ["Scripted follow-up?"],
            },
            "end_turn",
            "scripted-generator",
        )


def main() -> None:
    settings = replace(load_settings(), web=WebSettings(rate_limit_per_minute=1000))
    database_url = resolve_database_url(settings)
    services = Services(
        settings, "scripted", ScriptedGenerator(), SentenceTransformerEmbedder(settings.embedding),
        lambda: PostgresStore.connect(database_url).conn,
    )
    uvicorn.run(create_app(services, ROOT / "frontend" / "dist"), host="127.0.0.1", port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
