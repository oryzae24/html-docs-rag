import json
from pathlib import Path

import pytest

from python_doc_rag.models import SearchChunk
from python_doc_rag.technical_retrieval import (
    FieldBM25Retriever,
    SymbolRetriever,
    WeightedRankFusionRetriever,
    extract_identifiers,
    identifier_variants,
    load_symbol_sidecar,
    symbol_record_for,
    write_symbol_sidecar_atomic,
)


def _chunk(
    name: str,
    *,
    text: str,
    section_title: str = "節",
    page_title: str = "ページ",
    index: int = 0,
) -> SearchChunk:
    return SearchChunk(
        text=text,
        page_title=page_title,
        section_title=section_title,
        source_url=f"https://docs.python.org/ja/3.13/library/{name}.html#{name}",
        category="library",
        chunk_index=index,
        start_index=index * 100,
        extra_metadata={"preserved": name},
    )


def test_symbol_extraction_preserves_qualified_calls_and_underscores() -> None:
    identifiers = extract_identifiers(
        "IOBase.isatty() と ensure_ascii、argparse.ArgumentParserを使う。"
    )

    assert "IOBase.isatty()" in identifiers
    assert "ensure_ascii" in identifiers
    assert "argparse.ArgumentParser" in identifiers


def test_identifier_variants_are_general_and_keep_python_grammar() -> None:
    variants = identifier_variants("contextlib.aclosing()")

    assert variants[:3] == (
        "contextlib.aclosing()",
        "contextlib.aclosing",
        "aclosing",
    )
    assert "aclosing()" in variants
    assert identifier_variants("not-an-identifier") == ()


def test_symbol_record_uses_heading_anchor_and_body_without_hard_coding() -> None:
    chunk = _chunk(
        "custom_api",
        text="pkg.Widget.run() は value_name を受け取ります。",
        section_title="Widget.run()",
    )

    record = symbol_record_for(chunk)

    assert "pkg.Widget.run()" in record.identifiers
    assert "Widget.run()" in record.identifiers
    assert "custom_api" in record.identifiers
    assert "isatty" not in record.identifiers


def test_symbol_sidecar_is_deterministic_and_rejects_artifact_mismatch(
    tmp_path: Path,
) -> None:
    chunks = [
        _chunk("io", text="IOBase.isatty() の説明", index=0),
        _chunk("json", text="json.dumps() の説明", index=1),
    ]
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"

    first_summary = write_symbol_sidecar_atomic(chunks, first)
    second_summary = write_symbol_sidecar_atomic(chunks, second)

    assert first.read_bytes() == second.read_bytes()
    assert first_summary.sha256 == second_summary.sha256
    assert load_symbol_sidecar(chunks, first) == tuple(
        symbol_record_for(chunk) for chunk in chunks
    )
    changed = [
        SearchChunk.from_dict(chunks[0].to_dict() | {"text": "改変"}),
        chunks[1],
    ]
    with pytest.raises(ValueError, match="mismatch"):
        load_symbol_sidecar(changed, first)


def test_symbol_sidecar_rejects_wrong_dimension_and_invalid_json(
    tmp_path: Path,
) -> None:
    chunks = [_chunk("io", text="isatty()", index=0)]
    path = tmp_path / "symbols.jsonl"
    write_symbol_sidecar_atomic(chunks, path)
    path.write_text(path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="line 2"):
        load_symbol_sidecar(chunks, path)


def test_symbol_retriever_ranks_exact_match_and_preserves_metadata() -> None:
    suffix = _chunk("suffix", text="obj.isatty()", index=0)
    exact = _chunk("exact", text="IOBase.isatty()", index=1)
    chunks = [suffix, exact]
    retriever = SymbolRetriever(
        chunks, [symbol_record_for(chunk) for chunk in chunks]
    )

    results = retriever.search("IOBase.isatty()とは？", top_k=2)

    assert results[0].chunk == exact
    assert results[0].chunk.extra_metadata == {"preserved": "exact"}
    assert retriever.retrieve("unknown_identifier", limit=3) == ()


@pytest.mark.parametrize(
    ("field", "query", "expected"),
    [
        ("section_title", "非同期終了", "section"),
        ("page_title", "データモデル", "page"),
        ("body", "後始末を保証", "body"),
    ],
)
def test_field_bm25_isolates_each_text_field(
    field: str, query: str, expected: str
) -> None:
    chunks = [
        _chunk("section", text="無関係", section_title="非同期終了", index=0),
        _chunk("page", text="無関係", page_title="データモデル", index=1),
        _chunk("body", text="後始末を保証します", index=2),
    ]
    retriever = FieldBM25Retriever(chunks, field=field)

    results = retriever.search(query, top_k=3)

    assert results[0].chunk.extra_metadata["preserved"] == expected


def test_identifier_field_requires_aligned_records() -> None:
    chunk = _chunk("io", text="isatty()", index=0)
    with pytest.raises(ValueError, match="requires symbol records"):
        FieldBM25Retriever([chunk], field="identifiers")


class _FixedRetriever:
    def __init__(self, chunks: list[SearchChunk]) -> None:
        self.chunks = chunks
        self.calls = 0

    def retrieve(self, question: str, *, limit: int) -> tuple[SearchChunk, ...]:
        del question
        self.calls += 1
        return tuple(self.chunks[:limit])


def test_weighted_field_rrf_is_deterministic_and_reuses_searchers() -> None:
    first = _chunk("first", text="本文", index=0)
    second = _chunk("second", text="本文", index=1)
    identifier = _FixedRetriever([second, first])
    body = _FixedRetriever([first, second])
    searcher = WeightedRankFusionRetriever(
        [
            ("identifiers", identifier, 3.0),
            ("body", body, 1.0),
        ],
        rrf_k=10,
        candidate_k=30,
    )

    first_run = searcher.search("質問", top_k=2)
    second_run = searcher.search("質問", top_k=2)

    assert [result.chunk for result in first_run] == [second, first]
    assert [result.chunk for result in second_run] == [second, first]
    assert searcher.field_weights == {"identifiers": 3.0, "body": 1.0}
    assert identifier.calls == body.calls == 2


def test_symbol_sidecar_contains_no_chunk_text_or_internal_secret(
    tmp_path: Path,
) -> None:
    chunk = _chunk("safe", text="pkg.safe_call() private prose", index=0)
    path = tmp_path / "symbols.jsonl"

    write_symbol_sidecar_atomic([chunk], path)
    data = json.loads(path.read_text(encoding="utf-8"))

    assert "text" not in data
    assert "private prose" not in path.read_text(encoding="utf-8")
