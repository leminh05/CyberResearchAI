"""Benchmark: plain Python vs PySpark (local mode) on the same JSON-lines files.

What is measured, per dataset size:
* Python  - wall time for read + normalize + filter + dedup + quality report,
            and peak Python heap (tracemalloc).
* Spark   - session startup time (measured once, separately), then wall time
            for the same steps; JVM heap in use after the job (approximate,
            depends on garbage collection).
* both    - results must be identical (same surviving records), otherwise
            the comparison would be meaningless.

Limitations, stated in the report: Spark runs in local mode (one machine,
one JVM), so this measures single-machine overhead and multi-core use, NOT
cluster scalability.
"""

from __future__ import annotations

import gc
import os
import platform
import time
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from cyberresearch.bigdata import python_baseline
from cyberresearch.bigdata.dataset import scale_records, write_jsonl
from cyberresearch.config import SparkSettings


@dataclass
class BenchmarkRow:
    records: int
    file_mb: float
    python_seconds: float
    python_peak_mb: float
    spark_seconds: float
    spark_jvm_used_mb: float | None
    after_dedup: int
    duplicates_removed: int
    results_match: bool


def _jvm_used_mb(spark: Any) -> float | None:
    try:
        runtime = spark.sparkContext._jvm.java.lang.Runtime.getRuntime()
        return round((runtime.totalMemory() - runtime.freeMemory()) / 1024 / 1024, 1)
    except Exception:  # private API; not critical
        return None


def run_benchmark(
    base_records: list[dict[str, Any]],
    sizes: list[int],
    data_dir: Path,
    spark_settings: SparkSettings,
    repeats: int = 3,
) -> dict[str, Any]:
    """Run both implementations on every size; returns rows + environment info."""
    from cyberresearch.bigdata import spark_jobs  # imports pyspark

    files = {}
    for size in sizes:
        path = data_dir / f"papers_{size}.jsonl"
        write_jsonl(scale_records(base_records, size), path)
        files[size] = path

    from pyspark.sql import SparkSession

    # Startup can only be measured when this function creates the session; an
    # already running session (e.g. in tests) is reused and left running.
    owns_session = SparkSession.getActiveSession() is None
    start = time.perf_counter()
    spark = spark_jobs.create_spark_session(spark_settings, "CyberResearchAI-benchmark")
    spark_startup: float | None = time.perf_counter() - start if owns_session else None

    # Warm-up on the smallest file: JIT compilation and Python worker start-up
    # would otherwise be charged to whichever size runs first.
    spark_jobs.run_pipeline(spark, files[sizes[0]].as_uri())

    rows: list[BenchmarkRow] = []
    try:
        for size in sizes:
            path = files[size]
            py_times, spark_times = [], []
            for _ in range(repeats):
                gc.collect()
                t0 = time.perf_counter()
                py_records, py_report = python_baseline.run_pipeline(path)
                py_times.append(time.perf_counter() - t0)

                t0 = time.perf_counter()
                spark_df, spark_report = spark_jobs.run_pipeline(spark, path.as_uri())
                spark_times.append(time.perf_counter() - t0)

            # Memory is measured in a separate run: tracemalloc slows Python
            # down considerably and would distort the timing above.
            gc.collect()
            tracemalloc.start()
            python_baseline.run_pipeline(path)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            spark_keys = {(r["source"], r["source_id"]) for r in spark_df.select("source", "source_id").collect()}
            py_keys = {(r["source"], r["source_id"]) for r in py_records}
            rows.append(
                BenchmarkRow(
                    records=size,
                    file_mb=round(path.stat().st_size / 1024 / 1024, 2),
                    python_seconds=round(min(py_times), 3),
                    python_peak_mb=round(peak / 1024 / 1024, 1),
                    spark_seconds=round(min(spark_times), 3),
                    spark_jvm_used_mb=_jvm_used_mb(spark),
                    after_dedup=py_report["after_dedup"],
                    duplicates_removed=py_report["duplicates_removed"],
                    results_match=py_keys == spark_keys and py_report == spark_report,
                )
            )
    finally:
        if owns_session:
            spark.stop()

    return {
        "environment": {
            "python": platform.python_version(),
            "os": f"{platform.system()} {platform.release()}",
            "cpu_count": os.cpu_count(),
            "spark_master": spark_settings.master,
            "spark_driver_memory": spark_settings.driver_memory,
            "repeats": repeats,
            "timing": "best of repeats, after one warm-up run",
        },
        "spark_startup_seconds": round(spark_startup, 2) if spark_startup is not None else None,
        "base_records": len(base_records),
        "rows": [asdict(r) for r in rows],
    }


def format_markdown(result: dict[str, Any]) -> str:
    """Render benchmark results as a Markdown report."""
    env = result["environment"]
    lines = [
        "# Benchmark: plain Python vs PySpark (local mode)",
        "",
        f"- Machine: {env['os']}, {env['cpu_count']} logical CPUs, Python {env['python']}",
        f"- Spark: `{env['spark_master']}` (single JVM on this machine, **not a cluster**), "
        f"driver memory {env['spark_driver_memory']}",
        "- Spark session startup (paid once): "
        + (f"**{result['spark_startup_seconds']} s**" if result["spark_startup_seconds"] is not None
           else "not measured (existing session reused)"),
        f"- Timing: {env['timing']} ({env['repeats']} repeats)",
        f"- Base dataset: {result['base_records']} real source records from `paper_sources`; "
        "larger sizes are synthetic copies with new IDs and ~20% injected duplicates",
        "",
        "| Records | File (MB) | Python (s) | Spark (s) | Spark / Python | Python peak heap (MB) "
        "| Spark JVM used (MB) | Kept after dedup | Duplicates removed | Same result |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for r in result["rows"]:
        ratio = r["spark_seconds"] / r["python_seconds"] if r["python_seconds"] else float("nan")
        lines.append(
            f"| {r['records']:,} | {r['file_mb']} | {r['python_seconds']} | {r['spark_seconds']} | "
            f"{ratio:.2f}x | {r['python_peak_mb']} | {r['spark_jvm_used_mb']} | {r['after_dedup']:,} | "
            f"{r['duplicates_removed']:,} | {'yes' if r['results_match'] else 'NO'} |"
        )
    return "\n".join(lines) + "\n"
