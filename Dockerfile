# CyberResearchAI application image (CLI: ingestion, processing, RAG chatbot; website).

# Build the React website; only its static files go into the final image.
FROM node:22-slim AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

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

# Spark's Kafka connector, matching the installed PySpark and its Kafka client.
RUN python -c "import pyspark, urllib.request as u; v = pyspark.__version__; jars = pyspark.__path__[0] + '/jars/'; \
[u.urlretrieve('https://repo1.maven.org/maven2/' + p, jars + p.rsplit('/', 1)[1]) for p in ( \
f'org/apache/spark/spark-sql-kafka-0-10_2.12/{v}/spark-sql-kafka-0-10_2.12-{v}.jar', \
f'org/apache/spark/spark-token-provider-kafka-0-10_2.12/{v}/spark-token-provider-kafka-0-10_2.12-{v}.jar', \
'org/apache/kafka/kafka-clients/3.4.1/kafka-clients-3.4.1.jar', \
'org/apache/commons/commons-pool2/2.11.1/commons-pool2-2.11.1.jar')]"

COPY config ./config
COPY src ./src
COPY main.py ./
COPY --from=frontend /frontend/dist ./frontend/dist

# Run as a non-root user.
RUN useradd --create-home appuser && mkdir -p /app/data /app/logs /cache/huggingface \
    && chown -R appuser /app /cache
USER appuser

ENV PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
ENTRYPOINT ["python", "main.py"]
CMD ["check"]
