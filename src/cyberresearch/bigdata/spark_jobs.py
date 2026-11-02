"""PySpark DataFrame jobs for metadata processing and batch chunking.

Each function takes and returns a DataFrame, so steps can be chained and
tested one at a time. Logic matches ``python_baseline.py`` rule for rule.

Important: ``local[*]`` runs Spark in ONE JVM on this machine. It uses all
CPU cores, but it is not a cluster, so timings here say nothing about
distributed performance.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T

from cyberresearch.bigdata.python_baseline import (
    ARXIV_PREFIX,
    ARXIV_VERSION,
    COMPLETENESS_FIELDS,
    DOI_PREFIX,
    EDGE_WHITESPACE,
    MAX_YEAR,
    MIN_YEAR,
    NON_ALNUM,
    OPENALEX_PREFIX,
    SOURCE_RANK,
    WHITESPACE,
)
from cyberresearch.config import SparkSettings

# Explicit schema: avoids an extra pass over the file for schema inference.
RAW_SCHEMA = T.StructType(
    [T.StructField(name, T.StringType()) for name in
     ("source", "source_id", "title", "doi", "arxiv_id", "s2_paper_id", "openalex_id", "year", "abstract", "venue")]
)


def create_spark_session(settings: SparkSettings, app_name: str = "CyberResearchAI") -> SparkSession:
    """Start a local Spark session configured for a laptop."""
    # Workers must use the same Python interpreter (the project's .venv) and
    # be able to import this package for UDFs (src/ is not installed).
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    src_dir = str(Path(__file__).resolve().parents[2])
    paths = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    if src_dir not in paths:
        os.environ["PYTHONPATH"] = os.pathsep.join([src_dir, *paths])
    spark = (
        SparkSession.builder.master(settings.master)
        .appName(app_name)
        .config("spark.driver.memory", settings.driver_memory)
        .config("spark.sql.shuffle.partitions", str(settings.shuffle_partitions))
        .config("spark.ui.enabled", "false")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def _strip(column: Any) -> Any:
    return F.regexp_replace(column, EDGE_WHITESPACE, "")


def _blank_to_null(column: Any) -> Any:
    return F.when(column == "", None).otherwise(column)


def read_raw(spark: SparkSession, path: str) -> DataFrame:
    """Read the JSON-lines benchmark/raw dataset."""
    return spark.read.schema(RAW_SCHEMA).json(path)


def normalize_metadata(df: DataFrame) -> DataFrame:
    """Normalize titles, DOIs, arXiv/OpenAlex IDs and years with built-in functions.

    Built-in functions run inside the JVM; no Python UDF is needed here,
    which is what makes Spark efficient for this kind of work.
    """
    title = _blank_to_null(_strip(F.regexp_replace(F.coalesce(F.col("title"), F.lit("")), WHITESPACE, " ")))
    doi = F.regexp_replace(F.lower(_strip(F.coalesce(F.col("doi"), F.lit("")))), DOI_PREFIX, "")
    arxiv = F.regexp_replace(F.lower(_strip(F.coalesce(F.col("arxiv_id"), F.lit("")))), ARXIV_PREFIX, "")
    year_text = F.substring(_strip(F.coalesce(F.col("year"), F.lit(""))), 1, 4)
    return (
        df.withColumn("title", title)
        .withColumn("normalized_title", _blank_to_null(_strip(F.regexp_replace(
            F.lower(F.coalesce(F.col("title"), F.lit(""))), NON_ALNUM, " "))))
        .withColumn("doi", F.when(doi.startswith("10."), doi))
        .withColumn("arxiv_id", _blank_to_null(F.regexp_replace(arxiv, ARXIV_VERSION, "")))
        .withColumn("openalex_id", _blank_to_null(F.regexp_replace(
            _strip(F.coalesce(F.col("openalex_id"), F.lit(""))), OPENALEX_PREFIX, "")))
        .withColumn("s2_paper_id", _blank_to_null(_strip(F.coalesce(F.col("s2_paper_id"), F.lit("")))))
        .withColumn("year", F.when(year_text.rlike("^[0-9]{4}$"), year_text.cast("int")))
        .withColumn("abstract", _blank_to_null(_strip(F.coalesce(F.col("abstract"), F.lit("")))))
        .withColumn("venue", _blank_to_null(_strip(F.coalesce(F.col("venue"), F.lit("")))))
    )


def filter_records(df: DataFrame) -> DataFrame:
    """Keep records that have an identity and a plausible year."""
    has_identity = (
        F.col("normalized_title").isNotNull() | F.col("doi").isNotNull() | F.col("arxiv_id").isNotNull()
        | F.col("s2_paper_id").isNotNull() | F.col("openalex_id").isNotNull()
    )
    year_ok = F.col("year").isNull() | F.col("year").between(MIN_YEAR, MAX_YEAR)
    return df.filter(has_identity & year_ok)


def deduplicate(df: DataFrame) -> DataFrame:
    """Keep the most complete record per deduplication key.

    A window function ranks the records of each key (a shuffle groups equal
    keys on the same partition), then ``row_number() == 1`` keeps the best one.
    This is a single-key rule; linking records through *different* keys
    (A shares a DOI with B, B shares an arXiv ID with C) would need a
    connected-components algorithm, which the database stage handles.
    """
    key = F.coalesce(
        F.concat(F.lit("arxiv_id:"), F.col("arxiv_id")),
        F.concat(F.lit("doi:"), F.col("doi")),
        F.concat(F.lit("s2_paper_id:"), F.col("s2_paper_id")),
        F.concat(F.lit("openalex_id:"), F.col("openalex_id")),
        F.concat(F.lit("title:"), F.col("normalized_title"), F.lit("|"),
                 F.coalesce(F.col("year").cast("string"), F.lit(""))),
    )
    completeness = sum(F.col(c).isNotNull().cast("int") for c in COMPLETENESS_FIELDS)
    source_rank = F.coalesce(
        *[F.when(F.col("source") == name, F.lit(rank)) for name, rank in SOURCE_RANK.items()], F.lit(9)
    )
    window = Window.partitionBy("dedup_key").orderBy(
        F.col("_completeness").desc(), F.col("_source_rank").asc(), F.col("source_id").asc()
    )
    return (
        df.withColumn("dedup_key", key)
        .withColumn("_completeness", completeness)
        .withColumn("_source_rank", source_rank)
        .withColumn("_row", F.row_number().over(window))
        .filter(F.col("_row") == 1)
        .drop("_row", "_completeness", "_source_rank")
    )


def quality_report(normalized: DataFrame, filtered: DataFrame, deduped: DataFrame) -> dict[str, Any]:
    """Data quality metrics in one aggregation pass per DataFrame."""
    agg = normalized.agg(
        F.count(F.lit(1)).alias("total"),
        *[F.sum(F.col(c).isNull().cast("int")).alias(c) for c in COMPLETENESS_FIELDS],
    ).first()
    total = agg["total"]
    after_filter = filtered.count()
    after_dedup = deduped.count()
    by_source = {r["source"]: r["count"] for r in deduped.groupBy("source").count().collect()}
    return {
        "input_records": total,
        "after_filter": after_filter,
        "after_dedup": after_dedup,
        "removed_by_filter": total - after_filter,
        "duplicates_removed": after_filter - after_dedup,
        "null_rate": {c: round(agg[c] / total, 4) if total else 0.0 for c in COMPLETENESS_FIELDS},
        "by_source": dict(sorted(by_source.items())),
    }


def run_pipeline(spark: SparkSession, path: str) -> tuple[DataFrame, dict[str, Any]]:
    """Read -> normalize -> filter -> deduplicate -> quality report."""
    normalized = normalize_metadata(read_raw(spark, path)).cache()  # reused by 3 actions
    filtered = filter_records(normalized)
    deduped = deduplicate(filtered)
    report = quality_report(normalized, filtered, deduped)
    normalized.unpersist()
    return deduped, report


# ------------------------------------------------------- batch PDF chunking

CHUNK_SCHEMA = T.ArrayType(
    T.StructType(
        [
            T.StructField("chunk_index", T.IntegerType()),
            T.StructField("section", T.StringType()),
            T.StructField("page_start", T.IntegerType()),
            T.StructField("page_end", T.IntegerType()),
            T.StructField("word_count", T.IntegerType()),
            T.StructField("text", T.StringType()),
        ]
    )
)


def chunk_documents(pages: DataFrame, max_words: int = 220, overlap_words: int = 40,
                    skip_sections: tuple[str, ...] = ()) -> DataFrame:
    """Batch-chunk extracted pages: one output row per chunk.

    Input columns: ``paper_id``, ``page_number``, ``blocks`` (array<string>).
    Pages are grouped per paper (shuffle), then the same Python cleaning,
    section and chunking code used by the single-paper pipeline runs inside
    a UDF on the executors. Python UDFs cost serialization between the JVM
    and Python workers - the price for reusing existing Python logic.
    """

    def _chunk(page_rows: list[Any]) -> list[tuple]:
        from cyberresearch.processing.chunking import chunk_paragraphs
        from cyberresearch.processing.cleaning import clean_pages
        from cyberresearch.processing.extract import PageText
        from cyberresearch.processing.sections import segment

        page_texts = [PageText(r["page_number"], list(r["blocks"] or [])) for r in sorted(page_rows, key=lambda r: r["page_number"])]
        chunks = chunk_paragraphs(segment(clean_pages(page_texts)), max_words, overlap_words, skip_sections)
        return [(c.chunk_index, c.section, c.page_start, c.page_end, c.word_count, c.text) for c in chunks]

    chunk_udf = F.udf(_chunk, CHUNK_SCHEMA)
    grouped = pages.groupBy("paper_id").agg(
        F.collect_list(F.struct("page_number", "blocks")).alias("pages")
    )
    return (
        grouped.withColumn("chunk", F.explode(chunk_udf(F.col("pages"))))
        .select("paper_id", "chunk.*")
    )
