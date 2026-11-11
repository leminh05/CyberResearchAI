"""Spark DataFrame transformations and the Python vs Spark benchmark."""

import json
import os
import sys
from pathlib import Path

import pytest

from cyberresearch.bigdata import python_baseline
from cyberresearch.bigdata.dataset import flatten_source_record, scale_records, write_jsonl
from cyberresearch.config import SparkSettings
from fakes import fixture_json

pytestmark = pytest.mark.spark

RAW = [
    # same paper three times: arXiv URL+version, arXiv: prefix, and an S2 record
    {"source": "arxiv", "source_id": "2401.01234", "title": "Deep  Learning\n for IDS", "doi": "10.1109/TIFS.1",
     "arxiv_id": "http://arxiv.org/abs/2401.01234v2", "year": "2024-01-03", "abstract": "A", "venue": None},
    {"source": "semantic_scholar", "source_id": "abc", "title": "Deep Learning for IDS", "doi": None,
     "arxiv_id": "arXiv:2401.01234", "s2_paper_id": "abc", "year": 2024, "abstract": None, "venue": "CCS"},
    # same DOI in different formats
    {"source": "openalex", "source_id": "W1", "title": "Botnets", "doi": "https://doi.org/10.5/ABC",
     "openalex_id": "https://openalex.org/W1", "year": 2023, "abstract": "x", "venue": "J"},
    {"source": "semantic_scholar", "source_id": "def", "title": "BOTNETS", "doi": " doi:10.5/abc ",
     "s2_paper_id": "def", "year": "2023"},
    # title+year duplicates (no identifiers)
    {"source": "openalex", "source_id": "W2", "title": "Zero-Day Attacks!", "year": 2022},
    {"source": "openalex", "source_id": "W3", "title": "zero day attacks", "year": 2022, "abstract": "more"},
    # filtered out: no title and no ids / implausible year
    {"source": "openalex", "source_id": "W4", "title": "   ", "year": 2022},
    {"source": "openalex", "source_id": "W5", "title": "Ancient paper", "year": 1700},
    {"source": "openalex", "source_id": "W6", "title": "Bad DOI", "doi": "not-a-doi", "year": "n/a"},
]


@pytest.fixture
def raw_path(tmp_path: Path) -> Path:
    path = tmp_path / "raw.jsonl"
    write_jsonl(RAW, path)
    return path


@pytest.mark.skipif(sys.platform == "win32" and not os.getenv("HADOOP_HOME"),
                    reason="Spark cannot write local files on Windows without winutils; `hdfs-check` covers this in Docker")
def test_lake_round_trip_cleans_up(spark, tmp_path) -> None:
    from cyberresearch.bigdata.hdfs import round_trip

    result = round_trip(spark, tmp_path.as_uri(), rows=100)
    assert result["rows"] == 100 and result["parquet_files"] >= 1
    assert list((tmp_path / "_check").iterdir()) == []


def test_normalize_metadata(spark, raw_path: Path) -> None:
    from cyberresearch.bigdata.spark_jobs import normalize_metadata, read_raw

    rows = {r["source_id"]: r for r in normalize_metadata(read_raw(spark, raw_path.as_uri())).collect()}
    assert rows["2401.01234"]["title"] == "Deep Learning for IDS"
    assert rows["2401.01234"]["normalized_title"] == "deep learning for ids"
    assert rows["2401.01234"]["arxiv_id"] == "2401.01234" == rows["abc"]["arxiv_id"]
    assert rows["2401.01234"]["year"] == 2024 and rows["abc"]["year"] == 2024
    assert rows["W1"]["doi"] == "10.5/abc" == rows["def"]["doi"]
    assert rows["W1"]["openalex_id"] == "W1"
    assert rows["W4"]["title"] is None and rows["W6"]["doi"] is None and rows["W6"]["year"] is None


def test_filter_and_deduplicate(spark, raw_path: Path) -> None:
    from cyberresearch.bigdata.spark_jobs import deduplicate, filter_records, normalize_metadata, read_raw

    filtered = filter_records(normalize_metadata(read_raw(spark, raw_path.as_uri())))
    assert {r["source_id"] for r in filtered.collect()} == {r["source_id"] for r in RAW} - {"W4", "W5"}
    kept = {r["source_id"]: r["dedup_key"] for r in deduplicate(filtered).collect()}
    assert kept == {
        "2401.01234": "arxiv_id:2401.01234",  # arXiv beats S2: equal completeness, better source rank
        "W1": "doi:10.5/abc",                 # more complete than the S2 copy
        "W3": "title:zero day attacks|2022",  # has an abstract, W2 does not
        "W6": "title:bad doi|",
    }


def test_spark_matches_python_baseline(spark, raw_path: Path) -> None:
    from cyberresearch.bigdata.spark_jobs import run_pipeline

    spark_df, spark_report = run_pipeline(spark, raw_path.as_uri())
    py_records, py_report = python_baseline.run_pipeline(raw_path)
    assert spark_report == py_report
    assert {(r["source"], r["source_id"]) for r in spark_df.collect()} == {(r["source"], r["source_id"]) for r in py_records}
    assert py_report["input_records"] == 9 and py_report["removed_by_filter"] == 2
    assert py_report["duplicates_removed"] == 3


def test_spark_matches_python_on_real_api_payloads(spark, tmp_path: Path) -> None:
    from cyberresearch.bigdata.spark_jobs import run_pipeline

    base = [flatten_source_record("openalex", w["id"] or "x", w) for w in fixture_json("openalex_works.json")["results"]]
    base += [flatten_source_record("semantic_scholar", p["paperId"], p) for p in fixture_json("s2_bulk.json")["data"]]
    path = tmp_path / "scaled.jsonl"
    write_jsonl(scale_records(base, 500, duplicate_rate=0.3), path)
    _, spark_report = run_pipeline(spark, path.as_uri())
    _, py_report = python_baseline.run_pipeline(path)
    assert spark_report == py_report
    assert py_report["duplicates_removed"] > 0


def test_batch_chunk_documents(spark) -> None:
    from cyberresearch.bigdata.spark_jobs import chunk_documents

    body = " ".join(f"w{i}." if i % 10 == 9 else f"w{i}" for i in range(150))
    pages = spark.createDataFrame(
        [
            (1, 2, ["more text " + body]),
            (1, 1, ["1 Introduction\n" + body]),
            (2, 1, ["Abstract—" + body]),
            (2, 2, ["References\n[1] Something cited, 2020 " * 5]),
        ],
        "paper_id long, page_number int, blocks array<string>",
    )
    chunks = chunk_documents(pages, max_words=100, overlap_words=20, skip_sections=("References",)).collect()
    by_paper = {}
    for c in chunks:
        by_paper.setdefault(c["paper_id"], []).append(c)
    assert set(by_paper) == {1, 2}
    assert {c["section"] for c in by_paper[1]} == {"1 Introduction"}
    assert by_paper[1][0]["page_start"] == 1 and by_paper[1][-1]["page_end"] == 2  # pages sorted correctly
    assert {c["section"] for c in by_paper[2]} == {"Abstract"}  # references skipped


def test_scale_records_is_deterministic_with_duplicates() -> None:
    base = [{"source": "openalex", "source_id": f"W{i}", "title": f"T{i}", "doi": f"10.1/{i}",
             "openalex_id": f"https://openalex.org/W{i}", "year": 2024} for i in range(10)]
    a, b = scale_records(base, 1000), scale_records(base, 1000)
    assert a == b and len(a) == 1000
    assert len({r["source_id"] for r in a}) == 1000  # every record has its own source id


def test_benchmark_small_sizes(spark, tmp_path: Path) -> None:
    from cyberresearch.bigdata.benchmark import format_markdown, run_benchmark

    base = [flatten_source_record("openalex", w["id"] or "x", w) for w in fixture_json("openalex_works.json")["results"]]
    result = run_benchmark(base, [50, 300], tmp_path, SparkSettings(master="local[2]", shuffle_partitions=2), repeats=1)
    # The shared test session already exists: startup cannot be measured and
    # the benchmark must not stop a session it does not own.
    assert result["spark_startup_seconds"] is None
    assert spark.range(3).count() == 3
    assert [r["records"] for r in result["rows"]] == [50, 300]
    assert all(r["results_match"] for r in result["rows"])
    assert all(r["python_seconds"] > 0 and r["spark_seconds"] > 0 for r in result["rows"])
    report = format_markdown(result)
    assert "not a cluster" in report
    json.dumps(result)  # serializable
