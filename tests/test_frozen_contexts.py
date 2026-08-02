import json
from pathlib import Path

import pytest

from python_doc_rag.frozen_contexts import (
    FrozenContextRecord,
    FrozenContextRetriever,
    chunk_tuple_sha256,
    load_frozen_contexts,
    save_frozen_contexts_atomic,
)
from python_doc_rag.models import SearchChunk


def _chunk(name: str) -> SearchChunk:
    return SearchChunk(
        text=f"本文{name} https://untrusted.invalid/path",
        page_title=f"ページ{name}",
        section_title=f"節{name}",
        source_url=f"https://docs.python.org/ja/3.13/library/{name}.html#{name}",
        category="library",
        chunk_index=0,
        start_index=0,
        extra_metadata={"diagnostic": name},
    )


def _record() -> FrozenContextRecord:
    chunks = tuple(_chunk(str(index)) for index in range(5))
    return FrozenContextRecord(
        id="q1",
        question="質問",
        dataset="answerability",
        chunks=chunks,
        rerank_scores=(5.0, 4.0, 3.0, 2.0, 1.0),
        original_ranks=(2, 1, 4, 3, 5),
    )


def test_frozen_context_round_trip_keeps_exact_tuple_and_sanitizes_prompt(
    tmp_path: Path,
) -> None:
    path = tmp_path / "contexts.json"
    record = _record()
    save_frozen_contexts_atomic([record], path, settings={"openai_api_used": False})

    loaded = load_frozen_contexts(path)

    assert loaded == (record,)
    payload = json.loads(path.read_text(encoding="utf-8"))
    visible = payload["records"][0]["prompt_visible_contexts"]
    assert "docs.python.org" not in json.dumps(visible, ensure_ascii=False)
    assert "untrusted.invalid" not in json.dumps(visible, ensure_ascii=False)
    assert "[URL除去済み]" in visible[0]["text"]
    assert payload["records"][0]["chunk_tuple_sha256"] == chunk_tuple_sha256(
        record.chunks
    )


def test_frozen_retriever_returns_same_tuple_and_rejects_truncation() -> None:
    record = _record()
    retriever = FrozenContextRetriever([record])

    assert retriever.retrieve("質問", limit=5) is record.chunks
    assert retriever.calls == 1
    with pytest.raises(ValueError, match="truncate"):
        retriever.retrieve("質問", limit=4)
    with pytest.raises(ValueError, match="absent"):
        retriever.retrieve("別質問", limit=5)


def test_frozen_context_rejects_chunk_or_visible_prompt_tampering(
    tmp_path: Path,
) -> None:
    path = tmp_path / "contexts.json"
    save_frozen_contexts_atomic([_record()], path, settings={})
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["records"][0]["chunks"][0]["text"] = "改変"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_frozen_contexts(path)

    save_frozen_contexts_atomic([_record()], path, settings={})
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["records"][0]["prompt_visible_contexts"][0]["text"] = "改変"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="prompt-visible"):
        load_frozen_contexts(path)


def test_frozen_context_dataset_filter_and_duplicate_question_rejection(
    tmp_path: Path,
) -> None:
    answerability = _record()
    rag = FrozenContextRecord(
        id="q2",
        question="別質問",
        dataset="rag-quality",
        chunks=answerability.chunks,
        rerank_scores=answerability.rerank_scores,
        original_ranks=answerability.original_ranks,
    )
    path = tmp_path / "contexts.json"
    save_frozen_contexts_atomic([answerability, rag], path, settings={})

    assert load_frozen_contexts(path, dataset="rag-quality") == (rag,)
    with pytest.raises(ValueError, match="duplicate frozen question"):
        FrozenContextRetriever([answerability, answerability])
