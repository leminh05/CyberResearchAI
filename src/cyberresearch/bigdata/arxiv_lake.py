"""Bronze / Silver / Gold data lake over the Kaggle arXiv metadata snapshot.

  Bronze: every raw record as Parquet, unchanged, plus ingestion metadata.
  Silver: cybersecurity papers (category cs.CR), cleaned and de-duplicated.
  Gold:   statistics for the report.

``python_silver`` applies the Silver rules in plain Python, which gives a
Python vs Spark comparison on real data with results that must be identical.
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from email.utils import parsedate_to_datetime
from typing import Any

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType, StructField, StructType

from cyberresearch.ingestion.normalize import clean_text, normalize_arxiv_id, normalize_doi
from cyberresearch.ingestion.schema import PaperRecord
from cyberresearch.ingestion.storage import PostgresStore

SNAPSHOT_SOURCE = "arxiv_kaggle"
SECURITY_CATEGORY = "cs.CR"
WITHDRAWN = r"(?i)\bthis (paper|article|submission|manuscript) (has been|is|was) withdrawn\b"
_NEW_ID = r"^(\d{2})\d{2}\.\d+"  # 2401.00001
_OLD_ID = r"^[a-z.\-A-Z]+/(\d{2})\d+"  # cs/0101001

ARXIV_SCHEMA = StructType([
    StructField("id", StringType()),
    StructField("submitter", StringType()),
    StructField("authors", StringType()),
    StructField("title", StringType()),
    StructField("comments", StringType()),
    StructField("journal-ref", StringType()),
    StructField("doi", StringType()),
    StructField("report-no", StringType()),
    StructField("categories", StringType()),
    StructField("license", StringType()),
    StructField("abstract", StringType()),
    StructField("update_date", StringType()),
    StructField("versions", ArrayType(StructType([StructField("version", StringType()), StructField("created", StringType())]))),
    StructField("authors_parsed", ArrayType(ArrayType(StringType()))),
    StructField("_corrupt_record", StringType()),
])


def read_raw(spark: SparkSession, path: str) -> DataFrame:
    # An explicit schema avoids a full scan of the file to infer one.
    return spark.read.schema(ARXIV_SCHEMA).option("mode", "PERMISSIVE").json(path)


def _clean(column: Any) -> Any:
    return F.trim(F.regexp_replace(column, r"\s+", " "))


def bronze(raw: DataFrame) -> DataFrame:
    return raw.withColumn("_ingested_at", F.current_timestamp()).withColumn("_source_file", F.input_file_name())


def silver(raw: DataFrame) -> tuple[DataFrame, DataFrame]:
    """Returns (kept papers, rejected ids with a reason). Same rules as ``python_silver``."""
    security = (
        raw.filter(F.array_contains(F.split(F.col("categories"), " "), SECURITY_CATEGORY))
        .withColumn("title", _clean(F.col("title")))
        .withColumn("abstract", _clean(F.col("abstract")))
    )
    reason = (
        F.when(F.coalesce(F.col("title"), F.lit("")) == "", "no_title")
        .when(F.coalesce(F.col("abstract"), F.lit("")) == "", "no_abstract")
        .when(F.col("abstract").rlike(WITHDRAWN), "withdrawn")
    )
    marked = security.withColumn("_reject", reason)
    valid = (
        marked.filter(F.col("_reject").isNull())
        .drop("_reject")
        .withColumn("title_key", F.trim(F.regexp_replace(F.lower(F.col("title")), r"[^a-z0-9]+", " ")))
        .withColumn("_yy", F.coalesce(F.nullif(F.regexp_extract("id", _NEW_ID, 1), F.lit("")),
                                      F.nullif(F.regexp_extract("id", _OLD_ID, 1), F.lit(""))).cast("int"))
        .withColumn("year", F.when(F.col("_yy") >= 91, F.col("_yy") + 1900).otherwise(F.col("_yy") + 2000))
        .drop("_yy")
    )
    # Keep the earliest id of papers re-posted with the same title.
    ranked = valid.withColumn("_rank", F.row_number().over(Window.partitionBy("title_key").orderBy("id")))
    kept = ranked.filter(F.col("_rank") == 1).drop("_rank", "_corrupt_record")
    rejected = marked.filter(F.col("_reject").isNotNull()).select("id", F.col("_reject").alias("reason")).union(
        ranked.filter(F.col("_rank") > 1).select("id", F.lit("duplicate_title").alias("reason"))
    )
    return kept, rejected


def gold(kept: DataFrame) -> dict[str, DataFrame]:
    categories = kept.select(F.explode(F.split("categories", " ")).alias("category"))
    return {
        "by_year": kept.groupBy("year").agg(
            F.count("*").alias("papers"),
            F.sum(F.col("doi").isNotNull().cast("int")).alias("with_doi"),
        ).orderBy("year"),
        "cross_categories": categories.filter(F.col("category") != SECURITY_CATEGORY)
        .groupBy("category").count().orderBy(F.desc("count")),
    }


def python_silver(path: str) -> dict[str, Any]:
    """The Silver rules in plain Python, one line at a time."""
    best: dict[str, str] = {}
    reasons: Counter = Counter()
    records = 0
    withdrawn = re.compile(WITHDRAWN)
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            records += 1
            row = json.loads(line)
            if SECURITY_CATEGORY not in (row.get("categories") or "").split(" "):
                continue
            title = " ".join((row.get("title") or "").split())
            abstract = " ".join((row.get("abstract") or "").split())
            if not title:
                reasons["no_title"] += 1
            elif not abstract:
                reasons["no_abstract"] += 1
            elif withdrawn.search(abstract):
                reasons["withdrawn"] += 1
            else:
                key = re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()
                if key in best:
                    reasons["duplicate_title"] += 1
                best[key] = min(best.get(key, row["id"]), row["id"])
    return {"records": records, "kept": len(best), "rejected": dict(reasons)}


def silver_to_record(row: dict[str, Any]) -> PaperRecord:
    """Map one Silver paper to the common schema; a separate source keeps the API's own payloads."""
    arxiv_id = normalize_arxiv_id(row["id"]) or row["id"]
    versions = row.get("versions") or []
    created = versions[0]["created"] if versions else None
    published = parsedate_to_datetime(created).date() if created else None
    authors = [" ".join(p for p in (a[1], a[0], a[2]) if p) for a in row.get("authors_parsed") or [] if len(a) >= 3]
    return PaperRecord(
        source=SNAPSHOT_SOURCE,
        source_id=arxiv_id,
        title=row["title"],
        abstract=row["abstract"],
        authors=authors,
        year=row.get("year"),
        publication_date=published,
        doi=normalize_doi(row.get("doi")),
        arxiv_id=arxiv_id,
        venue=clean_text(row.get("journal-ref")),
        url=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
        is_open_access=True,
        categories=(row.get("categories") or "").split(),
        raw={"license": row.get("license")},
    )


def load_silver(spark: SparkSession, root: str, store: PostgresStore, limit: int | None = None) -> Counter:
    """Upsert Silver papers into PostgreSQL. Safe to repeat: known papers come back as unchanged."""
    silver_df = spark.read.parquet(f"{root.rstrip('/')}/silver/arxiv_security").orderBy(F.desc("id"))
    if limit:
        silver_df = silver_df.limit(limit)
    counts: Counter = Counter()
    for row in silver_df.toLocalIterator():
        with store.transaction():
            counts[store.upsert_record(silver_to_record(row.asDict(recursive=True))).value] += 1
    return counts


def _size_bytes(spark: SparkSession, path: str) -> tuple[int, int]:
    """(stored bytes, bytes including replicas)."""
    jvm_path = spark.sparkContext._jvm.org.apache.hadoop.fs.Path(path)
    summary = jvm_path.getFileSystem(spark.sparkContext._jsc.hadoopConfiguration()).getContentSummary(jvm_path)
    return summary.getLength(), summary.getSpaceConsumed()


def _reasons(rejected: DataFrame) -> dict[str, int]:
    return {r["reason"]: r["count"] for r in rejected.groupBy("reason").count().collect()}


def build_lake(spark: SparkSession, source: str, root: str, compare: bool = True) -> dict[str, Any]:
    """Run all layers, overwriting earlier output, and return a measured report."""
    root = root.rstrip("/")
    report: dict[str, Any] = {"source": source, "lake_root": root, "layers": {}}

    start = time.perf_counter()
    bronze(read_raw(spark, source)).write.mode("overwrite").parquet(f"{root}/bronze/arxiv")
    bronze_df = spark.read.parquet(f"{root}/bronze/arxiv")
    report["layers"]["bronze"] = {
        "seconds": round(time.perf_counter() - start, 1),
        "records": bronze_df.count(),
        "corrupt_records": bronze_df.filter(F.col("_corrupt_record").isNotNull()).count(),
    }

    start = time.perf_counter()
    kept, rejected = silver(bronze_df.drop("_ingested_at", "_source_file"))
    kept.write.mode("overwrite").parquet(f"{root}/silver/arxiv_security")
    rejected.write.mode("overwrite").parquet(f"{root}/silver/arxiv_rejected")
    silver_df = spark.read.parquet(f"{root}/silver/arxiv_security")
    report["layers"]["silver"] = {
        "seconds": round(time.perf_counter() - start, 1),
        "records": silver_df.count(),
        "rejected": _reasons(spark.read.parquet(f"{root}/silver/arxiv_rejected")),
    }

    start = time.perf_counter()
    tables = gold(silver_df)
    for name, table in tables.items():
        table.write.mode("overwrite").parquet(f"{root}/gold/{name}")
    report["layers"]["gold"] = {
        "seconds": round(time.perf_counter() - start, 1),
        "by_year": [r.asDict() for r in spark.read.parquet(f"{root}/gold/by_year").orderBy("year").collect()],
        "top_cross_categories": [
            r.asDict() for r in spark.read.parquet(f"{root}/gold/cross_categories").orderBy(F.desc("count")).limit(10).collect()
        ],
    }
    for layer in ("bronze", "silver", "gold"):
        stored, with_replicas = _size_bytes(spark, f"{root}/{layer}")
        report["layers"][layer].update(stored_mb=round(stored / 1e6, 1), with_replicas_mb=round(with_replicas / 1e6, 1))

    if compare:
        start = time.perf_counter()
        spark_kept, spark_rejected = silver(read_raw(spark, source))
        spark_result = {"kept": spark_kept.count(), "rejected": _reasons(spark_rejected)}
        spark_seconds = time.perf_counter() - start
        start = time.perf_counter()
        python_result = python_silver(source)
        python_seconds = time.perf_counter() - start
        report["python_vs_spark"] = {
            "task": "raw JSON to Silver result, one run each, same machine",
            "records": python_result["records"],
            "python_seconds": round(python_seconds, 1),
            "spark_seconds": round(spark_seconds, 1),
            "python": python_result,
            "spark": spark_result,
            "identical": (python_result["kept"], python_result["rejected"]) == (spark_result["kept"], spark_result["rejected"]),
        }
    return report
