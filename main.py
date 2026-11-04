"""CyberResearchAI entry point.

Usage (PowerShell, from the project root with .venv activated):
    python main.py check                                  # verify environment and configuration
    python main.py init-db                                # create database tables
    python main.py daily [--sources arxiv] [--max-records 50]
    python main.py backfill --start 2023-01-01 [--end 2024-12-31]
    python main.py status                                 # counts, checkpoints, recent logs
    python main.py process [--limit 20] [--no-download]   # PDFs -> chunks -> embeddings
    python main.py search "intrusion detection with GNN"  # semantic search over chunks
    python main.py benchmark [--sizes 1000 10000]         # plain Python vs PySpark
    python main.py chat                                   # terminal research chatbot (Claude)
    python main.py ask "How are GNNs used for IDS?"       # one question, then exit
    python main.py rag-eval [--retrieval-only]            # run the test questions
    python main.py evaluate [--part data retrieval performance answers]  # Phase 4 metrics
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys

sys.path.insert(0, "src")

from cyberresearch.config import PROJECT_ROOT, load_settings  # noqa: E402
from cyberresearch.evaluation.cli import add_evaluation_commands, run_evaluation_command  # noqa: E402
from cyberresearch.ingestion.cli import add_ingestion_commands, run_ingestion_command  # noqa: E402
from cyberresearch.logging_setup import setup_logging  # noqa: E402
from cyberresearch.processing.cli import add_processing_commands, run_processing_command  # noqa: E402
from cyberresearch.rag.cli import RAG_COMMANDS, add_rag_commands, run_rag_command  # noqa: E402

PROCESSING_COMMANDS = {"process", "search", "benchmark"}


def _java_version() -> str | None:
    """Return the first line of ``java -version``, or None if Java is missing."""
    if shutil.which("java") is None:
        return None
    result = subprocess.run(["java", "-version"], capture_output=True, text=True)
    lines = (result.stderr or result.stdout).splitlines()
    return lines[0] if lines else None


def check_environment() -> int:
    """Print environment status. Returns 0 when all required checks pass."""
    print(f"Python      : {sys.version.split()[0]}", end="")
    if sys.version_info[:2] != (3, 11):
        print("  [WARN] project targets Python 3.11")
    else:
        print("  [OK]")

    java = _java_version()
    print(f"Java        : {java or 'not found'}  [required by PySpark 3.5 (Java 8/11/17); PySpark 4.x needs 17+]")

    env_file = PROJECT_ROOT / ".env"
    print(f".env file   : {'present' if env_file.exists() else 'missing (copy .env.example)'}")

    try:
        settings = load_settings()
    except Exception as exc:  # report any config error clearly
        print(f"Config      : INVALID - {exc}")
        return 1
    print(f"Config      : OK ({len(settings.sources)} sources, {len(settings.topics)} topics)")
    print(f"Database URL: {'set' if settings.database_url else 'not set (embedded dev DB will be used)'}")
    for name in ("semantic_scholar_api_key", "openalex_api_key"):
        # Only report whether a key exists; never print the value.
        print(f"{name:<24}: {'set' if getattr(settings, name) else 'not set (optional)'}")
    print(f"{'anthropic_api_key':<24}: {'set' if settings.anthropic_api_key_present else 'not set (needed for --provider claude)'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="CyberResearchAI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check", help="verify environment and configuration")
    add_ingestion_commands(subparsers)
    add_processing_commands(subparsers)
    add_rag_commands(subparsers)
    add_evaluation_commands(subparsers)
    args = parser.parse_args()

    if args.command == "check":
        return check_environment()
    settings = load_settings()
    setup_logging(settings.log_level, settings.log_dir, settings.log_file_name)
    if args.command in PROCESSING_COMMANDS:
        return run_processing_command(args, settings)
    if args.command in RAG_COMMANDS:
        return run_rag_command(args, settings)
    if args.command == "evaluate":
        return run_evaluation_command(args, settings)
    return run_ingestion_command(args, settings)


if __name__ == "__main__":
    sys.exit(main())
