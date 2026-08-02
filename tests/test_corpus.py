import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from python_doc_rag.config import ChunkingConfig
from python_doc_rag.corpus import (
    build_corpus,
    enumerate_html_files,
    write_chunks_jsonl_atomic,
)
from python_doc_rag.models import SearchChunk
from python_doc_rag.sites.python_docs.constants import TARGET_CATEGORIES


def _write_page(path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""<!doctype html>
<html lang="ja"><head><title>{title}</title></head>
<body><main><section id="section"><h1>{title}</h1>
<p>これは検索用コーパスに含める十分な長さの日本語本文です。</p>
<h2 id="short">短い節</h2><p>短い</p>
</section></main></body></html>""",
        encoding="utf-8",
    )


def _chunk(text: str = "日本語のチャンク") -> SearchChunk:
    return SearchChunk(
        text=text,
        page_title="ページ",
        section_title="節",
        source_url="https://docs.python.org/ja/3.13/tutorial/index.html#section",
        category="tutorial",
        chunk_index=0,
        start_index=0,
    )


def test_enumeration_selects_categories_in_fixed_order(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    _write_page(root / "tutorial" / "z.html", "Z")
    _write_page(root / "tutorial" / "nested" / "a.html", "A")
    _write_page(root / "library" / "b.html", "B")
    _write_page(root / "c-api" / "ignored.html", "対象外")
    _write_page(root / "about.html", "対象外")

    first = enumerate_html_files(root)
    second = enumerate_html_files(root)
    relative_paths = [
        path.relative_to(first.document_root).as_posix() for path in first.files
    ]

    assert relative_paths == [
        "library/b.html",
        "tutorial/nested/a.html",
        "tutorial/z.html",
    ]
    assert first.files == second.files
    assert first.missing_categories == ("reference", "howto", "faq")


def test_test_mode_limits_each_category(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    for category in ("tutorial", "library"):
        for index in range(4):
            _write_page(root / category / f"{index}.html", f"{category}-{index}")

    enumeration = enumerate_html_files(
        root,
        test_mode=True,
        max_files_per_category=2,
    )
    selected_categories = [
        path.relative_to(enumeration.document_root).parts[0]
        for path in enumeration.files
    ]

    assert selected_categories.count("tutorial") == 2
    assert selected_categories.count("library") == 2
    assert enumeration.available_html_count == 8


def test_enumeration_does_not_follow_symlinks_outside_root(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    outside = tmp_path / "outside"
    _write_page(root / "tutorial" / "safe.html", "安全")
    _write_page(outside / "escaped.html", "ルート外")
    (root / "tutorial" / "external-directory").symlink_to(
        outside,
        target_is_directory=True,
    )
    (root / "tutorial" / "external-file.html").symlink_to(outside / "escaped.html")

    enumeration = enumerate_html_files(root)
    relative_paths = [
        path.relative_to(enumeration.document_root).as_posix()
        for path in enumeration.files
    ]

    assert relative_paths == ["tutorial/safe.html"]


def test_jsonl_preserves_japanese_and_replaces_destination(tmp_path: Path) -> None:
    output_path = tmp_path / "chunks.jsonl"
    output_path.write_text("古い内容\n", encoding="utf-8")

    count = write_chunks_jsonl_atomic([_chunk()], output_path)

    raw_jsonl = output_path.read_text(encoding="utf-8")
    saved = json.loads(raw_jsonl)
    assert count == 1
    assert "日本語のチャンク" in raw_jsonl
    assert saved == _chunk().to_dict()
    assert not list(tmp_path.glob(".chunks.jsonl.*.tmp"))


def test_jsonl_failure_preserves_existing_destination(tmp_path: Path) -> None:
    output_path = tmp_path / "chunks.jsonl"
    output_path.write_text("正常な既存ファイル\n", encoding="utf-8")

    def failing_chunks() -> Iterator[SearchChunk]:
        yield _chunk()
        raise RuntimeError("fixture failure")

    with pytest.raises(RuntimeError, match="fixture failure"):
        write_chunks_jsonl_atomic(failing_chunks(), output_path)

    assert output_path.read_text(encoding="utf-8") == "正常な既存ファイル\n"
    assert not list(tmp_path.glob(".chunks.jsonl.*.tmp"))


def test_build_continues_after_bad_file_and_manifest_matches(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    _write_page(root / "tutorial" / "good.html", "良いページ")
    broken_path = root / "tutorial" / "broken.html"
    broken_path.write_bytes(b"\xff\xfe\x00")
    output_jsonl = tmp_path / "processed" / "chunks.jsonl"
    manifest_path = tmp_path / "processed" / "manifest.json"

    result = build_corpus(
        root,
        output_jsonl,
        manifest_path,
        archive_sha256="fixture-sha256",
        chunking_config=ChunkingConfig(chunk_size=1000, chunk_overlap=150),
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    jsonl_rows = [
        json.loads(line)
        for line in output_jsonl.read_text(encoding="utf-8").splitlines()
    ]

    assert result.detected_html_count == 2
    assert result.successful_html_count == 1
    assert result.failed_html_count == 1
    assert result.extracted_section_count == 1
    assert result.generated_chunk_count == 1
    assert result.excluded_section_count == 1
    assert result.failures[0].source_path == "tutorial/broken.html"
    assert "UnicodeDecodeError" in result.failures[0].reason
    assert len(jsonl_rows) == result.generated_chunk_count
    assert jsonl_rows[0]["text"].startswith("これは検索用コーパス")
    assert manifest["target_categories"] == list(TARGET_CATEGORIES)
    assert manifest["detected_html_count"] == result.detected_html_count
    assert manifest["successful_html_count"] == result.successful_html_count
    assert manifest["failed_html_count"] == result.failed_html_count
    assert manifest["extracted_section_count"] == result.extracted_section_count
    assert manifest["generated_chunk_count"] == result.generated_chunk_count
    assert manifest["excluded_empty_or_short_section_count"] == 1
    assert manifest["archive_sha256"] == "fixture-sha256"
    assert manifest["chunk_size"] == 1000
    assert manifest["chunk_overlap"] == 150
    assert manifest["failures"][0]["source_path"] == "tutorial/broken.html"
