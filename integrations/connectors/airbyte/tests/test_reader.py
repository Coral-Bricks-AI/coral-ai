"""Tests for coralbricks.connectors.airbyte.read_airbyte_output."""

from __future__ import annotations

from pathlib import Path

import pytest

from coralbricks.connectors.airbyte import read_airbyte_output

FIXTURES = Path(__file__).parent / "fixtures"


def test_reads_local_json_format_with_nested_data() -> None:
    records = read_airbyte_output(FIXTURES / "stories.jsonl", text_field="title")
    assert len(records) == 3
    r = records[0]
    assert r["id"] == "ab-0001"  # fallback to _airbyte_ab_id
    assert "$AAPL" in r["text"]
    assert r["source"] == "stories"
    assert r["metadata"]["by"] == "pg"
    assert r["metadata"]["_airbyte"]["_airbyte_ab_id"] == "ab-0001"
    assert "_airbyte_data" not in r["metadata"]["_airbyte"]


def test_reads_modern_s3_format_with_meta() -> None:
    records = read_airbyte_output(
        FIXTURES / "stories_20260420_00001.jsonl", text_field="title"
    )
    assert len(records) == 2
    assert records[0]["id"] == "0197d5c4-0001-0001-0001-000000000001"
    assert records[0]["source"] == "stories"  # trailing digit-only segments stripped
    assert records[0]["metadata"]["_airbyte"]["_airbyte_generation_id"] == 0
    assert records[0]["metadata"]["_airbyte"]["_airbyte_meta"] == {"changes": []}


def test_directory_reads_all_jsonl_recursively() -> None:
    records = read_airbyte_output(FIXTURES, text_field="title")
    # 3 stories (local) + 2 stories (s3) + 2 items
    assert len(records) == 7


def test_directory_stream_filter_narrows_files() -> None:
    records = read_airbyte_output(FIXTURES, stream="stories", text_field="title")
    assert len(records) == 5
    assert all(r["source"] == "stories" for r in records)


def test_text_field_concatenates_sequence() -> None:
    records = read_airbyte_output(
        FIXTURES / "stories.jsonl", text_field=["title", "url"]
    )
    assert "$AAPL" in records[0]["text"]
    assert "https://example.com/1" in records[0]["text"]


def test_text_field_accepts_callable() -> None:
    records = read_airbyte_output(
        FIXTURES / "stories.jsonl",
        text_field=lambda d: f"[{d['by']}] {d['title']}",
    )
    assert records[0]["text"].startswith("[pg]")


def test_id_field_from_data_column() -> None:
    records = read_airbyte_output(
        FIXTURES / "stories.jsonl", text_field="title", id_field="id"
    )
    assert records[0]["id"] == "39678900"


def test_id_field_callable() -> None:
    records = read_airbyte_output(
        FIXTURES / "stories.jsonl",
        text_field="title",
        id_field=lambda d: f"hn-{d['id']}",
    )
    assert records[0]["id"] == "hn-39678900"


def test_empty_and_whitespace_lines_ignored(tmp_path: Path) -> None:
    f = tmp_path / "sparse.jsonl"
    f.write_text(
        "\n"
        "   \n"
        '{"_airbyte_ab_id":"x","_airbyte_emitted_at":1,"_airbyte_data":{"title":"hi"}}\n'
        "\n"
    )
    records = read_airbyte_output(f, text_field="title")
    assert len(records) == 1


def test_invalid_json_raises_with_location(tmp_path: Path) -> None:
    f = tmp_path / "broken.jsonl"
    f.write_text("{not json\n")
    with pytest.raises(ValueError, match="invalid JSON"):
        read_airbyte_output(f, text_field="title")


def test_non_object_line_raises(tmp_path: Path) -> None:
    f = tmp_path / "scalar.jsonl"
    f.write_text('"just a string"\n')
    with pytest.raises(ValueError, match="expected a JSON object"):
        read_airbyte_output(f, text_field="title")


def test_nonexistent_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_airbyte_output(tmp_path / "nope.jsonl")


def test_flat_columns_fallback_when_no_airbyte_data(tmp_path: Path) -> None:
    f = tmp_path / "flat.jsonl"
    f.write_text(
        '{"_airbyte_raw_id":"r1","_airbyte_extracted_at":"2026-04-20",'
        '"id":9,"title":"flat row"}\n'
    )
    records = read_airbyte_output(f, text_field="title")
    assert records[0]["text"] == "flat row"
    assert records[0]["id"] == "r1"
    assert records[0]["metadata"]["id"] == 9
    # The flat-column extractor must not leak airbyte envelope keys into
    # the row-level metadata.
    assert "_airbyte_raw_id" not in records[0]["metadata"]
    assert records[0]["metadata"]["_airbyte"]["_airbyte_raw_id"] == "r1"


def test_missing_text_field_becomes_empty_string() -> None:
    records = read_airbyte_output(FIXTURES / "stories.jsonl", text_field="does_not_exist")
    assert all(r["text"] == "" for r in records)


def test_missing_id_field_column_raises() -> None:
    with pytest.raises(KeyError, match="id_field"):
        read_airbyte_output(
            FIXTURES / "stories.jsonl",
            text_field="title",
            id_field="does_not_exist",
        )


def test_returns_empty_list_when_directory_has_no_jsonl(tmp_path: Path) -> None:
    (tmp_path / "other.txt").write_text("ignore me")
    records = read_airbyte_output(tmp_path, text_field="title")
    assert records == []


def test_stream_name_overrides_filename_source() -> None:
    records = read_airbyte_output(
        FIXTURES / "items.jsonl", stream="my_events", text_field="title"
    )
    assert all(r["source"] == "my_events" for r in records)
