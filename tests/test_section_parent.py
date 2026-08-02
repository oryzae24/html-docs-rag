import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from python_doc_rag.config import ChunkingConfig
from python_doc_rag.generation import (
    IdentityPromptSerializer,
    build_generation_prompt,
)
from python_doc_rag.models import DocumentSection, SearchChunk, SearchResult
from python_doc_rag.reranking import RerankingRetriever
from python_doc_rag.section_parent import (
    SectionContextResolver,
    SectionParentRetriever,
    SectionStore,
    create_section_children,
    load_sections_jsonl,
    section_id_for,
    write_sections_jsonl_atomic,
)


def section(name: str, *, text: str | None = None) -> DocumentSection:
    return DocumentSection(
        text=text or f"{name}の完全な節本文です。",
        page_title=f"{name}ページ",
        section_title=f"{name}節",
        source_path=f"library/{name}.html",
        source_url=f"https://docs.python.org/ja/3.13/library/{name}.html#{name}",
        anchor=name,
        category="library",
        python_version="3.13",
    )


def result(rank: int, item: SearchChunk) -> SearchResult:
    return SearchResult(
        rank=rank,
        score=1.0 / rank,
        chunk=item,
        page_title=item.page_title,
        section_title=item.section_title,
        source_url=item.source_url,
        category=item.category,
    )


class Searcher:
    def __init__(self, results: Sequence[SearchResult]) -> None:
        self.results = list(results)

    def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        del query
        return self.results[:top_k]


class CharacterTokenizer:
    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        truncation: bool,
    ) -> list[int]:
        assert not add_special_tokens
        assert not truncation
        return list(range(len(text)))


class LengthScorer:
    def score(self, pairs, *, batch_size):
        assert batch_size == 2
        return [float(len(document)) for _question, document in pairs]


def test_section_id_and_persistence_are_stable(tmp_path: Path) -> None:
    item = section("stable")
    copy = DocumentSection(**json.loads(json.dumps(item.to_dict())))
    assert section_id_for(item) == section_id_for(copy)
    assert section_id_for(item) != section("other")

    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    assert write_sections_jsonl_atomic([item], first) == 1
    assert write_sections_jsonl_atomic([item], second) == 1
    assert first.read_bytes() == second.read_bytes()
    assert load_sections_jsonl(first)[0].section == item


def test_section_persistence_and_store_reject_duplicates(tmp_path: Path) -> None:
    item = section("duplicate")
    with pytest.raises(ValueError, match="duplicate section_id"):
        write_sections_jsonl_atomic([item, item], tmp_path / "sections.jsonl")

    path = tmp_path / "valid.jsonl"
    write_sections_jsonl_atomic([item], path)
    records = load_sections_jsonl(path)
    with pytest.raises(ValueError, match="duplicate section_id"):
        SectionStore([*records, *records])


def test_child_mapping_and_section_retrieval_preserve_parent_metadata(
    tmp_path: Path,
) -> None:
    item = section("mapped", text="段落です。" * 30)
    children, summary = create_section_children([item], ChunkingConfig(40, 10))
    path = tmp_path / "sections.jsonl"
    write_sections_jsonl_atomic([item], path)
    store = SectionStore.from_jsonl(path)
    assert store.validate_children(children) == len(children)
    assert summary.section_count == 1
    assert summary.child_count > 1

    retriever = SectionParentRetriever(
        Searcher([result(1, children[0]), result(2, children[1])]),
        store,
        child_candidate_k=30,
    )
    ranked = retriever.search("質問", top_k=5)

    assert len(ranked) == 1
    assert ranked[0].chunk.text == item.text
    assert ranked[0].source_url == item.source_url
    assert retriever.last_trace.maximum_children_for_one_parent == 2


def test_section_store_rejects_unresolved_and_text_mismatch(tmp_path: Path) -> None:
    item = section("secure", text="正しい節本文です。" * 5)
    children, _ = create_section_children([item], ChunkingConfig(30, 5))
    path = tmp_path / "sections.jsonl"
    write_sections_jsonl_atomic([item], path)
    store = SectionStore.from_jsonl(path)

    unresolved = children[0].to_dict()
    unresolved["section_id"] = "f" * 64
    with pytest.raises(ValueError, match="unresolved"):
        store.resolve_child(SearchChunk.from_dict(unresolved))

    changed = children[0].to_dict()
    changed["text"] = "改ざん"
    with pytest.raises(ValueError, match="child text"):
        store.resolve_child(SearchChunk.from_dict(changed))

    changed_url = children[0].to_dict()
    changed_url["source_url"] = "https://docs.python.org/ja/3.13/library/other.html"
    with pytest.raises(ValueError, match="source_url"):
        store.resolve_child(SearchChunk.from_dict(changed_url))


def test_context_resolver_uses_full_section_when_it_fits(tmp_path: Path) -> None:
    candidate = _parent_candidate(section("full", text="短い完全な節です。"), tmp_path)
    resolver = SectionContextResolver()
    resolved = resolver(
        "質問",
        [candidate],
        tokenizer=CharacterTokenizer(),
        max_prompt_tokens=10_000,
        prompt_serializer=IdentityPromptSerializer(),
        initial_prompt_builder=_simple_prompt,
        retry_prompt_builder=_simple_prompt,
    )

    assert resolved[0].text == candidate.text
    assert resolved[0].extra_metadata["context_scope"] == "full_section"


def test_context_resolver_uses_matched_paragraph_window_and_hides_metadata(
    tmp_path: Path,
) -> None:
    original_text = "前の段落" * 20 + "\n\n" + "一致する段落" * 10 + "\n\n" + "後の段落" * 20
    item = section("window", text=original_text)
    candidate = _parent_candidate(item, tmp_path, child_index=4)
    matched_start = int(candidate.extra_metadata["matched_child_start"])
    matched_end = int(candidate.extra_metadata["matched_child_end"])
    resolver = SectionContextResolver()
    resolved = resolver(
        "質問",
        [candidate],
        tokenizer=CharacterTokenizer(),
        max_prompt_tokens=180,
        prompt_serializer=IdentityPromptSerializer(),
        initial_prompt_builder=_simple_prompt,
        retry_prompt_builder=_simple_prompt,
    )

    window = resolved[0]
    assert window.extra_metadata["context_scope"] == "section_window"
    assert original_text[matched_start:matched_end] in window.text
    assert item.text == original_text
    prompt = build_generation_prompt("質問", resolved)
    assert candidate.source_url not in prompt
    assert candidate.extra_metadata["section_id"] not in prompt
    assert "matched_child_start" not in prompt
    assert "context_window_start" not in prompt


def test_section_parent_and_reranker_preserve_section_citation_metadata(
    tmp_path: Path,
) -> None:
    short = section("short", text="短い節本文です。")
    long = section("long", text="長い節本文です。" * 20)
    path = tmp_path / "sections.jsonl"
    write_sections_jsonl_atomic([short, long], path)
    children, _ = create_section_children(
        [short, long],
        ChunkingConfig(400, 100),
    )
    store = SectionStore.from_jsonl(path)
    parents = SectionParentRetriever(
        Searcher([result(1, children[0]), result(2, children[1])]),
        store,
        child_candidate_k=30,
    )
    reranker = RerankingRetriever(
        parents,
        LengthScorer(),
        candidate_k=30,
        batch_size=2,
    )

    ranked = reranker.search("質問", top_k=2)

    assert ranked[0].chunk.text == long.text
    assert ranked[0].source_url == long.source_url
    assert ranked[0].chunk.extra_metadata["section_parent_revision"] == (
        "section-parent-v1"
    )


def _parent_candidate(
    item: DocumentSection,
    tmp_path: Path,
    *,
    child_index: int = 0,
) -> SearchChunk:
    path = tmp_path / f"{item.anchor}.jsonl"
    write_sections_jsonl_atomic([item], path)
    children, _ = create_section_children([item], ChunkingConfig(40, 10))
    store = SectionStore.from_jsonl(path)
    child = children[min(child_index, len(children) - 1)]
    section_id, _stored, start, end = store.resolve_child(child)
    return store.parent_chunk(
        section_id,
        matched_child_start=start,
        matched_child_end=end,
        matched_child_rank=1,
    )


def _simple_prompt(question: str, contexts: Sequence[SearchChunk]) -> str:
    return question + "|" + "|".join(context.text for context in contexts)
