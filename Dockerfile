# CyberResearchAI application image (CLI: ingestion, processing, RAG chatbot).
# Pinned to Debian 12 (bookworm): Debian 13 no longer ships OpenJDK 17, the
# Java version PySpark 3.5 officially supports.
FROM python:3.11-slim-bookworm

# Java 17 for PySpark (the image does not depend on the host's Java 8).
RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-17-jre-headless \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch first, so sentence-transformers does not pull the much
# larger CUDA build; then the project requirements.
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY config ./config
COPY src ./src
COPY main.py ./

# Run as a non-root user.
RUN useradd --create-home appuser && mkdir -p /app/data /app/logs /cache/huggingface \
    && chown -R appuser /app /cache
USER appuser

ENV PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
ENTRYPOINT ["python", "main.py"]
CMD ["check"]
