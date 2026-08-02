"""Existing-chunk parent retrieval with stable, validated child mappings."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from langchain_text_splitters import RecursiveCharacterTextSplitter

from python_doc_rag.config import ChunkingConfig
from python_doc_rag.models import SearchChunk, SearchResult
from python_doc_rag.vector_store import load_chunks_jsonl

PARENT_RETRIEVAL_REVISION = "existing-chunk-parent-v1"
PARENT_ID_ALGORITHM = (
    "sha256(canonical-json:{source_url,chunk_index,start_index};utf-8)"
)
_SEPARATORS = ["\n\n", "\n", "。", "、", " ", ""]
_CHILD_REQUIRED_METADATA = {
    "parent_id",
    "parent_source_url",
    "parent_chunk_index",
    "parent_start_index",
    "parent_text_sha256",
    "child_index",
    "child_start_index",
    "child_size",
    "child_overlap",
    "parent_retrieval_revision",
}


class ChildSearcher(Protocol):
    """Search interface used by the parent adapter."""

    def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        """Return child results in deterministic rank order."""
        ...


@dataclass(frozen=True, slots=True)
class ParentMappingSummary:
    """Counts produced while deriving child chunks from parent chunks."""

    parent_count: int
    child_count: int
    average_children_per_parent: float
    maximum_children_per_parent: int
    parent_id_collision_count: int
    unresolved_child_count: int


@dataclass(frozen=True, slots=True)
class ParentMatchDiagnostic:
    """One selected parent and the highest-ranked matching child."""

    parent_rank: int
    matched_child_rank: int
    matched_child_text: str
    matched_child_index: int
    parent_id: str
    child_hits_for_parent: int


@dataclass(frozen=True, slots=True)
class ParentRetrievalTrace:
    """Diagnostics retained outside the Generator boundary."""

    child_candidate_count: int
    unique_parent_candidate_count: int
    maximum_children_for_one_parent: int
    matches: tuple[ParentMatchDiagnostic, ...]

    @property
    def compression_ratio(self) -> float:
        """Return unique parents divided by returned child candidates."""
        if self.child_candidate_count == 0:
            return 0.0
        return self.unique_parent_candidate_count / self.child_candidate_count


def parent_id_for_chunk(parent: SearchChunk) -> str:
    """Return a stable SHA-256 ID from citation identity fields."""
    payload = {
        "source_url": parent.source_url,
        "chunk_index": parent.chunk_index,
        "start_index": parent.start_index,
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def text_sha256(text: str) -> str:
    """Return the SHA-256 of the exact UTF-8 text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def create_child_chunks(
    parents: Sequence[SearchChunk],
    config: ChunkingConfig,
) -> tuple[list[SearchChunk], ParentMappingSummary]:
    """Split only parent text and attach a complete validated mapping."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        add_start_index=True,
        separators=_SEPARATORS,
    )
    children: list[SearchChunk] = []
    seen_parent_ids: set[str] = set()
    child_counts: list[int] = []

    for parent in parents:
        parent_id = parent_id_for_chunk(parent)
        if parent_id in seen_parent_ids:
            raise ValueError(f"duplicate parent_id: {parent_id}")
        seen_parent_ids.add(parent_id)
        if not parent.text:
            raise ValueError(f"parent text must not be empty: {parent_id}")

        if len(parent.text) <= config.chunk_size:
            pieces = ((parent.text, 0),)
        else:
            documents = splitter.create_documents([parent.text])
            pieces = tuple(
                (document.page_content, int(document.metadata["start_index"]))
                for document in documents
                if document.page_content
            )
        if not pieces:
            raise ValueError(f"parent produced no child chunks: {parent_id}")
        covered = [False] * len(parent.text)
        for child_text, child_start in pieces:
            for position in range(child_start, child_start + len(child_text)):
                covered[position] = True
        if any(
            not character.isspace() and not covered[position]
            for position, character in enumerate(parent.text)
        ):
            raise ValueError(f"child splitting omitted parent content: {parent_id}")

        parent_hash = text_sha256(parent.text)
        for child_index, (child_text, child_start) in enumerate(pieces):
            if not child_text:
                raise ValueError(f"empty child chunk for parent: {parent_id}")
            if child_start < 0 or child_start + len(child_text) > len(parent.text):
                raise ValueError(f"invalid child_start_index for parent: {parent_id}")
            if parent.text[child_start : child_start + len(child_text)] != child_text:
                raise ValueError(f"child text is not an exact parent substring: {parent_id}")
            metadata = {
                "parent_id": parent_id,
                "parent_source_url": parent.source_url,
                "parent_chunk_index": parent.chunk_index,
                "parent_start_index": parent.start_index,
                "parent_text_sha256": parent_hash,
                "child_index": child_index,
                "child_start_index": child_start,
                "child_size": config.chunk_size,
                "child_overlap": config.chunk_overlap,
                "parent_retrieval_revision": PARENT_RETRIEVAL_REVISION,
            }
            children.append(
                SearchChunk(
                    text=child_text,
                    page_title=parent.page_title,
                    section_title=parent.section_title,
                    source_url=parent.source_url,
                    category=parent.category,
                    chunk_index=child_index,
                    start_index=child_start,
                    extra_metadata=metadata,
                )
            )
        child_counts.append(len(pieces))

    child_count = len(children)
    parent_count = len(parents)
    return children, ParentMappingSummary(
        parent_count=parent_count,
        child_count=child_count,
        average_children_per_parent=(
            child_count / parent_count if parent_count else 0.0
        ),
        maximum_children_per_parent=max(child_counts, default=0),
        parent_id_collision_count=0,
        unresolved_child_count=0,
    )


class ParentStore:
    """Load immutable citation-ready parents once and resolve validated children."""

    def __init__(self, parents: Iterable[SearchChunk]) -> None:
        by_id: dict[str, SearchChunk] = {}
        for parent in parents:
            parent_id = parent_id_for_chunk(parent)
            if parent_id in by_id:
                raise ValueError(f"duplicate parent_id: {parent_id}")
            by_id[parent_id] = parent
        if not by_id:
            raise ValueError("parent store must contain at least one parent")
        self._parents = by_id

    @classmethod
    def from_jsonl(cls, path: Path) -> ParentStore:
        """Load one parent metadata JSONL snapshot."""
        return cls(load_chunks_jsonl(path))

    @property
    def parent_count(self) -> int:
        """Return the number of stable parent mappings."""
        return len(self._parents)

    def lookup(self, parent_id: str) -> SearchChunk:
        """Resolve exactly one parent or reject the child reference."""
        try:
            return self._parents[parent_id]
        except KeyError as error:
            raise ValueError(f"unresolved parent_id: {parent_id}") from error

    def resolve_child(self, child: SearchChunk) -> tuple[str, SearchChunk]:
        """Validate all child-to-parent security fields before returning a parent."""
        metadata = child.extra_metadata
        missing = sorted(_CHILD_REQUIRED_METADATA.difference(metadata))
        if missing:
            raise ValueError("child metadata missing: " + ", ".join(missing))
        parent_id = _required_string(metadata, "parent_id")
        if len(parent_id) != 64 or any(
            character not in "0123456789abcdef" for character in parent_id
        ):
            raise ValueError("child parent_id must be a 64-character lowercase hex")
        revision = _required_string(metadata, "parent_retrieval_revision")
        if revision != PARENT_RETRIEVAL_REVISION:
            raise ValueError(f"unsupported parent retrieval revision: {revision}")
        parent = self.lookup(parent_id)
        if parent_id_for_chunk(parent) != parent_id:
            raise ValueError("child parent_id does not match parent identity")
        _required_string(metadata, "parent_source_url")
        _required_non_negative_int(metadata, "parent_chunk_index")
        _required_non_negative_int(metadata, "parent_start_index")
        parent_text_hash = _required_string(metadata, "parent_text_sha256")
        if len(parent_text_hash) != 64 or any(
            character not in "0123456789abcdef"
            for character in parent_text_hash
        ):
            raise ValueError(
                "child parent_text_sha256 must be a 64-character lowercase hex"
            )
        expected = {
            "parent_source_url": parent.source_url,
            "parent_chunk_index": parent.chunk_index,
            "parent_start_index": parent.start_index,
            "parent_text_sha256": text_sha256(parent.text),
        }
        for key, value in expected.items():
            if metadata[key] != value:
                raise ValueError(f"child {key} does not match parent")
        if child.source_url != parent.source_url:
            raise ValueError("child source_url does not match parent")
        if (
            child.page_title != parent.page_title
            or child.section_title != parent.section_title
            or child.category != parent.category
        ):
            raise ValueError("child citation metadata does not match parent")
        child_index = _required_non_negative_int(metadata, "child_index")
        child_start = _required_non_negative_int(metadata, "child_start_index")
        _required_positive_int(metadata, "child_size")
        child_overlap = _required_non_negative_int(metadata, "child_overlap")
        if child_overlap >= metadata["child_size"]:
            raise ValueError("child_overlap must be smaller than child_size")
        if child.start_index != child_start:
            raise ValueError("child start_index does not match child metadata")
        if child.chunk_index != child_index:
            raise ValueError("child chunk_index does not match child metadata")
        if child_start + len(child.text) > len(parent.text):
            raise ValueError("child range is outside parent text")
        if parent.text[child_start : child_start + len(child.text)] != child.text:
            raise ValueError("child text does not match parent text")
        return parent_id, parent

    def validate_children(self, children: Iterable[SearchChunk]) -> int:
        """Validate every child reference and return its count."""
        count = 0
        for child in children:
            self.resolve_child(child)
            count += 1
        return count


class ParentDocumentRetriever:
    """Search child chunks and return de-duplicated existing parent chunks."""

    def __init__(
        self,
        child_searcher: ChildSearcher,
        parent_store: ParentStore,
        *,
        child_candidate_k: int,
    ) -> None:
        _validate_positive_int(child_candidate_k, name="child_candidate_k")
        self._child_searcher = child_searcher
        self._parent_store = parent_store
        self._child_candidate_k = child_candidate_k
        self._last_trace = ParentRetrievalTrace(0, 0, 0, ())

    @property
    def child_candidate_k(self) -> int:
        """Return the number of child candidates requested per query."""
        return self._child_candidate_k

    @property
    def last_trace(self) -> ParentRetrievalTrace:
        """Return diagnostics for the most recent search or retrieve call."""
        return self._last_trace

    def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        """Return parents ranked by their first (highest-ranked) child."""
        _validate_positive_int(top_k, name="top_k")
        child_results = self._child_searcher.search(
            query,
            top_k=self._child_candidate_k,
        )
        resolved: list[tuple[SearchResult, str, SearchChunk]] = []
        counts: Counter[str] = Counter()
        for child_result in child_results:
            parent_id, parent = self._parent_store.resolve_child(child_result.chunk)
            counts[parent_id] += 1
            resolved.append((child_result, parent_id, parent))

        results: list[SearchResult] = []
        diagnostics: list[ParentMatchDiagnostic] = []
        seen: set[str] = set()
        for child_result, parent_id, parent in resolved:
            if parent_id in seen:
                continue
            seen.add(parent_id)
            rank = len(results) + 1
            results.append(
                SearchResult(
                    rank=rank,
                    score=child_result.score,
                    chunk=parent,
                    page_title=parent.page_title,
                    section_title=parent.section_title,
                    source_url=parent.source_url,
                    category=parent.category,
                )
            )
            diagnostics.append(
                ParentMatchDiagnostic(
                    parent_rank=rank,
                    matched_child_rank=child_result.rank,
                    matched_child_text=child_result.chunk.text,
                    matched_child_index=int(
                        child_result.chunk.extra_metadata["child_index"]
                    ),
                    parent_id=parent_id,
                    child_hits_for_parent=counts[parent_id],
                )
            )
            if len(results) == top_k:
                break

        self._last_trace = ParentRetrievalTrace(
            child_candidate_count=len(child_results),
            unique_parent_candidate_count=len(counts),
            maximum_children_for_one_parent=max(counts.values(), default=0),
            matches=tuple(diagnostics),
        )
        return results

    def retrieve(self, question: str, *, limit: int) -> tuple[SearchChunk, ...]:
        """Return parent chunks in child-derived rank order."""
        _validate_positive_int(limit, name="limit")
        return tuple(result.chunk for result in self.search(question, top_k=limit))


def _required_string(metadata: dict[str, Any], name: str) -> str:
    value = metadata[name]
    if not isinstance(value, str) or not value:
        raise ValueError(f"child {name} must be a non-empty string")
    return value


def _required_non_negative_int(metadata: dict[str, Any], name: str) -> int:
    value = metadata[name]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"child {name} must be a non-negative integer")
    return value


def _required_positive_int(metadata: dict[str, Any], name: str) -> int:
    value = metadata[name]
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"child {name} must be a positive integer")
    return value


def _validate_positive_int(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be at least 1")
