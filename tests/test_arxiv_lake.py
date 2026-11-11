"""Silver and Gold rules of the arXiv lake, and agreement between Spark and plain Python."""

import json

import pytest

from cyberresearch.bigdata.arxiv_lake import SNAPSHOT_SOURCE, gold, python_silver, read_raw, silver, silver_to_record

pytestmark = pytest.mark.spark
ABSTRACT = "  We study   intrusion\n detection. "


def record(arxiv_id, title="A Paper", categories="cs.CR", abstract=ABSTRACT, doi=None):
    return {"id": arxiv_id, "title": title, "categories": categories, "abstract": abstract, "doi": doi,
            "versions": [{"version": "v1", "created": "x"}], "authors_parsed": [["Lee", "Ann", ""]]}


RECORDS = [
    record("2401.00001", "Graph  IDS\n Methods", "cs.CR cs.NI", doi="10.1/a"),
    record("cs/0101001", "Old Paper", "cs.CR"),
    record("9912.0001", "Nineties Paper", "cs.CR"),
    record("2402.00002", "Not Security", "math.CO"),
    record("2402.00003", "Near cs.CR", "cs.CRX"),
    record("2403.00004", "   ", "cs.CR"),
    record("2403.00005", "No Abstract", "cs.CR", abstract=" "),
    record("2403.00006", "Gone", "cs.CR", abstract="This paper has been withdrawn by the author."),
    record("2405.00007", "graph ids: methods!", "cs.CR"),  # same title as the first paper
]


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "arxiv.json"
    path.write_text("\n".join(json.dumps(r) for r in RECORDS) + "\n", encoding="utf-8")
    return str(path)


def test_silver_rules(spark, source) -> None:
    kept, rejected = silver(read_raw(spark, source))
    rows = {r["id"]: r for r in kept.collect()}
    assert set(rows) == {"2401.00001", "cs/0101001", "9912.0001"}
    assert rows["2401.00001"]["title"] == "Graph IDS Methods"
    assert rows["2401.00001"]["abstract"] == "We study intrusion detection."
    assert {i: rows[i]["year"] for i in rows} == {"2401.00001": 2024, "cs/0101001": 2001, "9912.0001": 1999}
    assert {r["id"]: r["reason"] for r in rejected.collect()} == {
        "2403.00004": "no_title", "2403.00005": "no_abstract", "2403.00006": "withdrawn", "2405.00007": "duplicate_title",
    }


def test_spark_and_python_agree(spark, source) -> None:
    kept, rejected = silver(read_raw(spark, source))
    result = python_silver(source)
    assert result["records"] == len(RECORDS) and result["kept"] == kept.count()
    assert result["rejected"] == {r["reason"]: r["count"] for r in rejected.groupBy("reason").count().collect()}


def test_gold_tables(spark, source) -> None:
    kept, _ = silver(read_raw(spark, source))
    tables = gold(kept)
    assert {r["year"]: (r["papers"], r["with_doi"]) for r in tables["by_year"].collect()} == {
        1999: (1, 0), 2001: (1, 0), 2024: (1, 1),
    }
    assert {r["category"]: r["count"] for r in tables["cross_categories"].collect()} == {"cs.NI": 1}


def test_silver_row_becomes_a_record() -> None:
    record = silver_to_record({
        "id": "2401.00001v2", "title": "Graph IDS", "abstract": "We study IDS.", "year": 2024, "doi": "https://doi.org/10.1/A",
        "journal-ref": None, "license": "http://creativecommons.org/licenses/by/4.0/", "categories": "cs.CR cs.NI",
        "versions": [{"version": "v1", "created": "Mon, 2 Apr 2007 19:18:42 GMT"}],
        "authors_parsed": [["Lee", "Ann", ""], ["Tran", "Binh", "Jr."], ["broken"]],
    })
    assert record.source == SNAPSHOT_SOURCE and record.source_id == record.arxiv_id == "2401.00001"
    assert record.authors == ["Ann Lee", "Binh Tran Jr."] and record.doi == "10.1/a"
    assert record.publication_date.isoformat() == "2007-04-02" and record.categories == ["cs.CR", "cs.NI"]
    assert record.is_valid() and record.raw == {"license": "http://creativecommons.org/licenses/by/4.0/"}
