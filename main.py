"""CyberResearchAI entry point.

Usage (PowerShell, from the project root with .venv activated):
    python main.py check     # verify environment and configuration
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys

sys.path.insert(0, "src")

from cyberresearch.config import PROJECT_ROOT, load_settings  # noqa: E402


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
    print(f"Java        : {java or 'not found'}  [needed from the PySpark phase]")

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
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="CyberResearchAI")
    parser.add_argument("command", choices=["check"], help="command to run")
    args = parser.parse_args()
    if args.command == "check":
        return check_environment()
    return 1


if __name__ == "__main__":
    sys.exit(main())
