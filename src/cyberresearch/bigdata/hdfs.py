"""Check that Spark can really store data in the data lake (HDFS or a local path)."""

from __future__ import annotations

import uuid
from typing import Any

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def round_trip(spark: SparkSession, root: str, rows: int = 100_000) -> dict[str, Any]:
    """Write Parquet under ``root``, read it back, compare and clean up.

    Raises ``ValueError`` if what was read differs from what was written.
    """
    target = f"{root.rstrip('/')}/_check/{uuid.uuid4().hex}"
    written = spark.range(rows).withColumn("value", F.col("id") * 2)
    expected = written.agg(F.count("*").alias("n"), F.sum("value").alias("total")).first()
    written.write.parquet(target)

    jvm = spark.sparkContext._jvm
    path = jvm.org.apache.hadoop.fs.Path(target)
    fs = path.getFileSystem(spark.sparkContext._jsc.hadoopConfiguration())
    try:
        actual = spark.read.parquet(target).agg(F.count("*").alias("n"), F.sum("value").alias("total")).first()
        if (actual["n"], actual["total"]) != (expected["n"], expected["total"]):
            raise ValueError(f"read back {actual} but wrote {expected}")
        files = [s for s in fs.listStatus(path) if s.getPath().getName().endswith(".parquet")]
        return {
            "filesystem": str(fs.getUri()),
            "rows": actual["n"],
            "parquet_files": len(files),
            "replication": max((s.getReplication() for s in files), default=0),
        }
    finally:
        fs.delete(path, True)
