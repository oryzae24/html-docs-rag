import json
from pathlib import Path

import pytest

from python_doc_rag.config import ChunkingConfig
from python_doc_rag.corpus import write_chunks_jsonl_atomic
from python_doc_rag.generation import GenerationConfig
from python_doc_rag.models import SearchChunk, SearchResult
from python_doc_rag.parent_retrieval import (
    PARENT_RETRIEVAL_REVISION,
    ParentDocumentRetriever,
    ParentStore,
    create_child_chunks,
    parent_id_for_chunk,
    text_sha256,
)
from python_doc_rag.pipeline import RagPipeline
from python_doc_rag.vector_store import load_chunks_jsonl


def parent(
    name: str,
    *,
    url: str | None = None,
    chunk_index: int = 0,
    start_index: int = 0,
    text: str | None = None,
) -> SearchChunk:
    return SearchChunk(
        text=text or f"{name}の親本文",
        page_title=f"{name}ページ",
        section_title=f"{name}節",
        source_url=url or f"https://trusted.invalid/{name}",
        category="library",
        chunk_index=chunk_index,
        start_index=start_index,
    )


def child_for(item: SearchChunk, *, text: str | None = None) -> SearchChunk:
    children, _ = create_child_chunks(
        [item],
        ChunkingConfig(chunk_size=20, chunk_overlap=5),
    )
    child = children[0]
    if text is None:
        return child
    data = child.to_dict()
    data["text"] = text
    return SearchChunk.from_dict(data)


def result(rank: int, item: SearchChunk, score: float | None = None) -> SearchResult:
    return SearchResult(
        rank,
        score if score is not None else 1.0 / rank,
        item,
        item.page_title,
        item.section_title,
        item.source_url,
        item.category,
    )


class Searcher:
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        self.calls.append((query, top_k))
        return self.results[:top_k]


def test_parent_id_is_stable_and_uses_all_identity_fields(tmp_path: Path) -> None:
    base = parent("a")
    same = SearchChunk.from_dict(json.loads(json.dumps(base.to_dict())))
    assert parent_id_for_chunk(base) == parent_id_for_chunk(same)
    assert parent_id_for_chunk(base) != parent_id_for_chunk(
        parent("a", url="https://trusted.invalid/other")
    )
    assert parent_id_for_chunk(base) != parent_id_for_chunk(
        parent("a", chunk_index=1)
    )
    assert parent_id_for_chunk(base) != parent_id_for_chunk(
        parent("a", start_index=1)
    )

    path = tmp_path / "parent.jsonl"
    write_chunks_jsonl_atomic([base], path)
    assert parent_id_for_chunk(load_chunks_jsonl(path)[0]) == parent_id_for_chunk(base)


def test_parent_store_rejects_duplicate_parent_id() -> None:
    item = parent("a")
    duplicate = SearchChunk.from_dict(item.to_dict() | {"text": "別本文"})
    with pytest.raises(ValueError, match="duplicate parent_id"):
        ParentStore([item, duplicate])


def test_child_generation_short_long_overlap_offsets_and_metadata(
    tmp_path: Path,
) -> None:
    short = parent("short", text="短い本文")
    long = parent("long", text="0123456789" * 8)
    config = ChunkingConfig(chunk_size=20, chunk_overlap=5)
    first, summary = create_child_chunks([short, long], config)
    second, _ = create_child_chunks([short, long], config)

    assert first[0].text == short.text
    assert first[0].start_index == 0
    long_children = first[1:]
    assert len(long_children) > 1
    assert all(item.text for item in first)
    assert long_children[0].text[-5:] == long_children[1].text[:5]
    assert all(
        long.text[item.start_index : item.start_index + len(item.text)] == item.text
        for item in long_children
    )
    assert [item.extra_metadata["child_index"] for item in long_children] == list(
        range(len(long_children))
    )
    metadata = long_children[0].extra_metadata
    assert metadata["parent_id"] == parent_id_for_chunk(long)
    assert metadata["parent_text_sha256"] == text_sha256(long.text)
    assert metadata["parent_source_url"] == long.source_url
    assert metadata["child_size"] == 20
    assert metadata["child_overlap"] == 5
    assert metadata["parent_retrieval_revision"] == PARENT_RETRIEVAL_REVISION
    assert summary.parent_count == 2
    assert summary.child_count == len(first)
    assert summary.maximum_children_per_parent == len(long_children)

    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    write_chunks_jsonl_atomic(first, first_path)
    write_chunks_jsonl_atomic(second, second_path)
    assert first_path.read_bytes() == second_path.read_bytes()


@pytest.mark.parametrize(
    ("size", "overlap"),
    [(0, 0), (10, -1), (10, 10), (10, 11)],
)
def test_child_config_validation(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        ChunkingConfig(size, overlap)


def test_parent_store_lookup_and_child_validation() -> None:
    item = parent("a", text="abcdefghij")
    store = ParentStore([item])
    child = child_for(item)
    parent_id, resolved = store.resolve_child(child)
    assert resolved is item
    assert store.lookup(parent_id) is item
    assert store.validate_children([child]) == 1

    data = child.to_dict()
    data["parent_id"] = "f" * 64
    with pytest.raises(ValueError, match="unresolved"):
        store.resolve_child(SearchChunk.from_dict(data))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("parent_source_url", "https://wrong.invalid", "parent_source_url"),
        ("parent_text_sha256", "0" * 64, "parent_text_sha256"),
    ],
)
def test_parent_store_rejects_mapping_mismatch(
    field: str, value: object, message: str
) -> None:
    item = parent("a")
    data = child_for(item).to_dict()
    data[field] = value
    with pytest.raises(ValueError, match=message):
        ParentStore([item]).resolve_child(SearchChunk.from_dict(data))


def test_parent_store_rejects_child_source_and_text_mismatch() -> None:
    item = parent("a", text="親の正しい本文")
    store = ParentStore([item])
    data = child_for(item).to_dict()
    data["source_url"] = "https://wrong.invalid"
    with pytest.raises(ValueError, match="source_url"):
        store.resolve_child(SearchChunk.from_dict(data))
    with pytest.raises(ValueError, match="child text"):
        store.resolve_child(child_for(item, text="改ざん"))


def test_parent_store_jsonl_is_loaded_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "parents.jsonl"
    write_chunks_jsonl_atomic([parent("a")], path)
    calls = 0

    def load(received: Path) -> list[SearchChunk]:
        nonlocal calls
        calls += 1
        return load_chunks_jsonl(received)

    monkeypatch.setattr("python_doc_rag.parent_retrieval.load_chunks_jsonl", load)
    store = ParentStore.from_jsonl(path)
    store.lookup(parent_id_for_chunk(parent("a")))
    store.lookup(parent_id_for_chunk(parent("a")))
    assert calls == 1


def test_retriever_deduplicates_parent_and_preserves_first_child_score() -> None:
    first_parent = parent("first", text="abcdefghij" * 5)
    second_parent = parent(
        "second",
        url=first_parent.source_url,
        chunk_index=1,
        text="klmnopqrst" * 5,
    )
    first_children, _ = create_child_chunks(
        [first_parent], ChunkingConfig(20, 5)
    )
    second_child = child_for(second_parent)
    searcher = Searcher(
        [
            result(1, first_children[0], 0.9),
            result(2, first_children[1], 0.8),
            result(3, second_child, 0.7),
        ]
    )
    retriever = ParentDocumentRetriever(
        searcher,
        ParentStore([first_parent, second_parent]),
        child_candidate_k=30,
    )

    results = retriever.search("質問", top_k=10)
    retrieved = retriever.retrieve("質問", limit=1)

    assert [item.rank for item in results] == [1, 2]
    assert [item.chunk for item in results] == [first_parent, second_parent]
    assert [item.score for item in results] == [0.9, 0.7]
    assert retrieved == (first_parent,)
    assert searcher.calls == [("質問", 30), ("質問", 30)]
    trace = retriever.last_trace
    assert trace.child_candidate_count == 3
    assert trace.unique_parent_candidate_count == 2
    assert trace.maximum_children_for_one_parent == 2
    assert trace.matches[0].matched_child_rank == 1
    assert trace.matches[0].matched_child_text == first_children[0].text
    assert trace.matches[0].matched_child_index == 0
    assert trace.matches[0].child_hits_for_parent == 2


def test_retriever_candidate_shortage_ties_and_invalid_limits() -> None:
    first = parent("first")
    second = parent("second")
    searcher = Searcher([result(1, child_for(first), 0.5), result(2, child_for(second), 0.5)])
    retriever = ParentDocumentRetriever(
        searcher, ParentStore([first, second]), child_candidate_k=2
    )
    assert retriever.retrieve("質問", limit=10) == (first, second)
    assert retriever.retrieve("質問", limit=10) == (first, second)
    with pytest.raises(ValueError, match="top_k"):
        retriever.search("質問", top_k=0)
    with pytest.raises(ValueError, match="limit"):
        retriever.retrieve("質問", limit=0)
    with pytest.raises(ValueError, match="child_candidate_k"):
        ParentDocumentRetriever(searcher, ParentStore([first]), child_candidate_k=0)


def test_retriever_rejects_missing_child_metadata() -> None:
    item = parent("a")
    retriever = ParentDocumentRetriever(
        Searcher([result(1, item)]),
        ParentStore([item]),
        child_candidate_k=1,
    )
    with pytest.raises(ValueError, match="child metadata missing"):
        retriever.search("質問")


def test_pipeline_generator_and_citation_use_only_resolved_parent() -> None:
    item = parent(
        "secure",
        text="child-hit。parent-only-context。",
        url="https://trusted.invalid/parent-source",
    )
    child = child_for(item)
    searcher = Searcher([result(1, child)])
    retriever = ParentDocumentRetriever(
        searcher,
        ParentStore([item]),
        child_candidate_k=5,
    )

    class Generator:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def generate(self, prompt: str, *, max_new_tokens: int) -> str:
            del max_new_tokens
            self.calls.append(prompt)
            return "回答[S1]"

    class Tokenizer:
        def encode(
            self, text: str, *, add_special_tokens: bool, truncation: bool
        ) -> list[int]:
            assert not add_special_tokens
            assert not truncation
            return list(range(len(text)))

    generator = Generator()
    pipeline = RagPipeline(
        retriever=retriever,
        generator=generator,
        tokenizer=Tokenizer(),
        config=GenerationConfig(max_prompt_tokens=100_000),
    )
    outcome = pipeline.answer("質問")

    assert outcome.retrieved_chunks == (item,)
    assert outcome.sources[0].url == item.source_url
    prompt = generator.calls[0]
    assert item.text in prompt
    assert item.source_url not in prompt
    assert child.extra_metadata["parent_id"] not in prompt
    assert "child_start_index" not in prompt
    assert "parent_retrieval_revision" not in prompt
    assert pipeline.last_prompt_token_counts
