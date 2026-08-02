"""Full-DocumentSection parent retrieval with token-aware context windows."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

from langchain_text_splitters import RecursiveCharacterTextSplitter

from python_doc_rag.config import ChunkingConfig
from python_doc_rag.generation import (
    PromptSerializer,
    TokenizerProtocol,
    count_prompt_tokens,
)
from python_doc_rag.models import DocumentSection, SearchChunk, SearchResult

SECTION_PARENT_REVISION = "section-parent-v1"
SECTION_ID_ALGORITHM = "sha256(canonical-json:section-metadata;utf-8)"
_SEPARATORS = ["\n\n", "\n", "。", "、", " ", ""]
_CHILD_REQUIRED_METADATA = {
    "section_id",
    "section_text_sha256",
    "section_source_path",
    "section_source_url",
    "section_anchor",
    "section_python_version",
    "child_index",
    "child_start_index",
    "child_size",
    "child_overlap",
    "section_parent_revision",
}


class ChildSearcher(Protocol):
    """Search interface used by the section-parent adapter."""

    def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        """Return child results in deterministic rank order."""
        ...


@dataclass(frozen=True, slots=True)
class StoredSection:
    """One immutable section with stable identity and text hash."""

    section_id: str
    text_sha256: str
    section: DocumentSection

    def to_dict(self) -> dict[str, Any]:
        """Return a flat JSON-compatible section record."""
        return {
            "section_id": self.section_id,
            "text_sha256": self.text_sha256,
            **self.section.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class SectionBuildSummary:
    """Counts produced while splitting sections into search-only children."""

    section_count: int
    child_count: int
    average_children_per_section: float
    maximum_children_per_section: int
    duplicate_section_count: int
    unresolved_child_count: int


@dataclass(frozen=True, slots=True)
class SectionMatchDiagnostic:
    """One selected section and its highest-ranked matching child."""

    parent_rank: int
    matched_child_rank: int
    section_id: str
    matched_child_start: int
    matched_child_end: int
    child_hits_for_section: int


@dataclass(frozen=True, slots=True)
class SectionRetrievalTrace:
    """Internal diagnostics retained outside the Generator boundary."""

    child_candidate_count: int
    unique_parent_candidate_count: int
    maximum_children_for_one_parent: int
    scoring_seconds: float
    matches: tuple[SectionMatchDiagnostic, ...]

    @property
    def compression_ratio(self) -> float:
        """Return unique sections divided by child candidates."""
        if self.child_candidate_count == 0:
            return 0.0
        return self.unique_parent_candidate_count / self.child_candidate_count


@dataclass(frozen=True, slots=True)
class SectionContextDiagnostic:
    """One full-section or window decision made under the prompt budget."""

    rank: int
    context_scope: str
    section_id: str
    window_start: int
    window_end: int


def section_id_for(section: DocumentSection) -> str:
    """Return a stable SHA-256 over all non-text section metadata."""
    payload = {
        "anchor": section.anchor,
        "category": section.category,
        "page_title": section.page_title,
        "python_version": section.python_version,
        "section_title": section.section_title,
        "source_path": section.source_path,
        "source_url": section.source_url,
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def text_sha256(text: str) -> str:
    """Return the SHA-256 of exact UTF-8 section text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_sections_jsonl_atomic(
    sections: Iterable[DocumentSection],
    path: Path,
) -> int:
    """Validate and atomically persist deterministic section records."""
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    seen: set[str] = set()
    count = 0
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            for section in sections:
                stored = _stored_section(section)
                if stored.section_id in seen:
                    raise ValueError(f"duplicate section_id: {stored.section_id}")
                seen.add(stored.section_id)
                json.dump(stored.to_dict(), stream, ensure_ascii=False)
                stream.write("\n")
                count += 1
            stream.flush()
            os.fsync(stream.fileno())
        if count == 0:
            raise ValueError("section artifact must contain at least one section")
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return count


def load_sections_jsonl(path: Path) -> list[StoredSection]:
    """Load and fully validate an immutable section snapshot."""
    records: list[StoredSection] = []
    with path.expanduser().open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                records.append(_stored_section_from_dict(data))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid section at line {line_number}: {error}"
                ) from error
    if not records:
        raise ValueError("section artifact must contain at least one section")
    return records


def create_section_children(
    sections: Sequence[DocumentSection],
    config: ChunkingConfig,
) -> tuple[list[SearchChunk], SectionBuildSummary]:
    """Split section text into search-only children with validated mappings."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        add_start_index=True,
        separators=_SEPARATORS,
    )
    children: list[SearchChunk] = []
    seen_ids: set[str] = set()
    child_counts: list[int] = []
    for section in sections:
        section_id = section_id_for(section)
        if section_id in seen_ids:
            raise ValueError(f"duplicate section_id: {section_id}")
        seen_ids.add(section_id)
        if not section.text:
            raise ValueError(f"section text must not be empty: {section_id}")
        if len(section.text) <= config.chunk_size:
            pieces = ((section.text, 0),)
        else:
            pieces = tuple(
                (document.page_content, int(document.metadata["start_index"]))
                for document in splitter.create_documents([section.text])
                if document.page_content
            )
        if not pieces:
            raise ValueError(f"section produced no child chunks: {section_id}")
        _validate_child_coverage(section.text, pieces, section_id=section_id)
        section_hash = text_sha256(section.text)
        for child_index, (child_text, child_start) in enumerate(pieces):
            if section.text[child_start : child_start + len(child_text)] != child_text:
                raise ValueError(f"child text is not a section substring: {section_id}")
            children.append(
                SearchChunk(
                    text=child_text,
                    page_title=section.page_title,
                    section_title=section.section_title,
                    source_url=section.source_url,
                    category=section.category,
                    chunk_index=child_index,
                    start_index=child_start,
                    extra_metadata={
                        "section_id": section_id,
                        "section_text_sha256": section_hash,
                        "section_source_path": section.source_path,
                        "section_source_url": section.source_url,
                        "section_anchor": section.anchor,
                        "section_python_version": section.python_version,
                        "child_index": child_index,
                        "child_start_index": child_start,
                        "child_size": config.chunk_size,
                        "child_overlap": config.chunk_overlap,
                        "section_parent_revision": SECTION_PARENT_REVISION,
                    },
                )
            )
        child_counts.append(len(pieces))
    return children, SectionBuildSummary(
        section_count=len(sections),
        child_count=len(children),
        average_children_per_section=(
            len(children) / len(sections) if sections else 0.0
        ),
        maximum_children_per_section=max(child_counts, default=0),
        duplicate_section_count=0,
        unresolved_child_count=0,
    )


class SectionStore:
    """Resolve search-only children to immutable full DocumentSection parents."""

    def __init__(self, records: Iterable[StoredSection]) -> None:
        by_id: dict[str, StoredSection] = {}
        ordinals: dict[str, int] = {}
        for ordinal, record in enumerate(records):
            expected = _stored_section(record.section)
            if record != expected:
                raise ValueError("stored section identity or text hash mismatch")
            if record.section_id in by_id:
                raise ValueError(f"duplicate section_id: {record.section_id}")
            by_id[record.section_id] = record
            ordinals[record.section_id] = ordinal
        if not by_id:
            raise ValueError("section store must contain at least one section")
        self._by_id = by_id
        self._ordinals = ordinals

    @classmethod
    def from_jsonl(cls, path: Path) -> SectionStore:
        """Load a section snapshot once."""
        return cls(load_sections_jsonl(path))

    @property
    def section_count(self) -> int:
        """Return the number of immutable section parents."""
        return len(self._by_id)

    def resolve_child(
        self,
        child: SearchChunk,
    ) -> tuple[str, StoredSection, int, int]:
        """Fail closed on every child-to-section identity and text field."""
        metadata = child.extra_metadata
        missing = sorted(_CHILD_REQUIRED_METADATA.difference(metadata))
        if missing:
            raise ValueError("child metadata missing: " + ", ".join(missing))
        section_id = _required_hash(metadata, "section_id")
        try:
            stored = self._by_id[section_id]
        except KeyError as error:
            raise ValueError(f"unresolved section_id: {section_id}") from error
        if metadata["section_parent_revision"] != SECTION_PARENT_REVISION:
            raise ValueError("unsupported section parent revision")
        section = stored.section
        expected = {
            "section_text_sha256": stored.text_sha256,
            "section_source_path": section.source_path,
            "section_source_url": section.source_url,
            "section_anchor": section.anchor,
            "section_python_version": section.python_version,
        }
        for key, value in expected.items():
            if metadata[key] != value:
                raise ValueError(f"child {key} does not match section")
        if child.source_url != section.source_url:
            raise ValueError("child source_url does not match section")
        if (
            child.page_title != section.page_title
            or child.section_title != section.section_title
            or child.category != section.category
        ):
            raise ValueError("child citation metadata does not match section")
        child_index = _required_non_negative_int(metadata, "child_index")
        child_start = _required_non_negative_int(metadata, "child_start_index")
        child_size = _required_positive_int(metadata, "child_size")
        child_overlap = _required_non_negative_int(metadata, "child_overlap")
        if child_overlap >= child_size:
            raise ValueError("child_overlap must be smaller than child_size")
        if child.chunk_index != child_index or child.start_index != child_start:
            raise ValueError("child indices do not match child metadata")
        child_end = child_start + len(child.text)
        if section.text[child_start:child_end] != child.text:
            raise ValueError("child text does not match section text")
        return section_id, stored, child_start, child_end

    def validate_children(self, children: Iterable[SearchChunk]) -> int:
        """Validate every child reference and return its count."""
        count = 0
        for child in children:
            self.resolve_child(child)
            count += 1
        return count

    def parent_chunk(
        self,
        section_id: str,
        *,
        matched_child_start: int,
        matched_child_end: int,
        matched_child_rank: int,
    ) -> SearchChunk:
        """Return a full-section candidate carrying only internal diagnostics."""
        stored = self._by_id[section_id]
        section = stored.section
        return SearchChunk(
            text=section.text,
            page_title=section.page_title,
            section_title=section.section_title,
            source_url=section.source_url,
            category=section.category,
            chunk_index=self._ordinals[section_id],
            start_index=0,
            extra_metadata={
                "section_id": section_id,
                "section_text_sha256": stored.text_sha256,
                "matched_child_start": matched_child_start,
                "matched_child_end": matched_child_end,
                "matched_child_rank": matched_child_rank,
                "section_parent_revision": SECTION_PARENT_REVISION,
            },
        )


class SectionParentRetriever:
    """Search child chunks and return de-duplicated full section candidates."""

    def __init__(
        self,
        child_searcher: ChildSearcher,
        section_store: SectionStore,
        *,
        child_candidate_k: int,
    ) -> None:
        _validate_positive_int(child_candidate_k, name="child_candidate_k")
        self._child_searcher = child_searcher
        self._section_store = section_store
        self._child_candidate_k = child_candidate_k
        self._last_trace = SectionRetrievalTrace(0, 0, 0, 0.0, ())

    @property
    def last_trace(self) -> SectionRetrievalTrace:
        """Return the latest child-to-section retrieval diagnostics."""
        return self._last_trace

    def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        """Rank sections by their first highest-ranked child hit."""
        _validate_positive_int(top_k, name="top_k")
        started_at = perf_counter()
        child_results = self._child_searcher.search(
            query,
            top_k=self._child_candidate_k,
        )
        resolved: list[tuple[SearchResult, str, int, int]] = []
        counts: Counter[str] = Counter()
        for child_result in child_results:
            section_id, _stored, child_start, child_end = (
                self._section_store.resolve_child(child_result.chunk)
            )
            counts[section_id] += 1
            resolved.append((child_result, section_id, child_start, child_end))
        results: list[SearchResult] = []
        diagnostics: list[SectionMatchDiagnostic] = []
        seen: set[str] = set()
        for child_result, section_id, child_start, child_end in resolved:
            if section_id in seen:
                continue
            seen.add(section_id)
            rank = len(results) + 1
            parent = self._section_store.parent_chunk(
                section_id,
                matched_child_start=child_start,
                matched_child_end=child_end,
                matched_child_rank=child_result.rank,
            )
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
                SectionMatchDiagnostic(
                    parent_rank=rank,
                    matched_child_rank=child_result.rank,
                    section_id=section_id,
                    matched_child_start=child_start,
                    matched_child_end=child_end,
                    child_hits_for_section=counts[section_id],
                )
            )
            if len(results) == top_k:
                break
        self._last_trace = SectionRetrievalTrace(
            child_candidate_count=len(child_results),
            unique_parent_candidate_count=len(counts),
            maximum_children_for_one_parent=max(counts.values(), default=0),
            scoring_seconds=perf_counter() - started_at,
            matches=tuple(diagnostics),
        )
        return results

    def retrieve(self, question: str, *, limit: int) -> tuple[SearchChunk, ...]:
        """Return full-section candidates for token-aware context resolution."""
        return tuple(result.chunk for result in self.search(question, top_k=limit))


class SectionContextResolver:
    """Resolve full sections or matched-child windows under the exact budget."""

    def __init__(self) -> None:
        self._last_diagnostics: tuple[SectionContextDiagnostic, ...] = ()

    @property
    def last_diagnostics(self) -> tuple[SectionContextDiagnostic, ...]:
        """Return scope decisions from the latest question."""
        return self._last_diagnostics

    def __call__(
        self,
        question: str,
        candidates: Sequence[SearchChunk],
        *,
        tokenizer: TokenizerProtocol,
        max_prompt_tokens: int,
        prompt_serializer: PromptSerializer,
        initial_prompt_builder: Callable[[str, Sequence[SearchChunk]], str],
        retry_prompt_builder: Callable[[str, Sequence[SearchChunk]], str],
    ) -> tuple[SearchChunk, ...]:
        """Prefer full sections, then deterministic paragraph-boundary windows."""
        selected: list[SearchChunk] = []
        diagnostics: list[SectionContextDiagnostic] = []
        for candidate in candidates:
            metadata = candidate.extra_metadata
            if metadata.get("section_parent_revision") != SECTION_PARENT_REVISION:
                raise ValueError("section context candidate has unsupported revision")
            section_id = _required_hash(metadata, "section_id")
            if _contexts_fit(
                question,
                (*selected, candidate),
                tokenizer=tokenizer,
                max_prompt_tokens=max_prompt_tokens,
                prompt_serializer=prompt_serializer,
                initial_prompt_builder=initial_prompt_builder,
                retry_prompt_builder=retry_prompt_builder,
            ):
                resolved = _context_chunk(
                    candidate,
                    text=candidate.text,
                    scope="full_section",
                    start=0,
                    end=len(candidate.text),
                )
            else:
                resolved = _largest_fitting_window(
                    question,
                    candidate,
                    selected,
                    tokenizer=tokenizer,
                    max_prompt_tokens=max_prompt_tokens,
                    prompt_serializer=prompt_serializer,
                    initial_prompt_builder=initial_prompt_builder,
                    retry_prompt_builder=retry_prompt_builder,
                )
                if resolved is None:
                    continue
            selected.append(resolved)
            diagnostics.append(
                SectionContextDiagnostic(
                    rank=len(selected),
                    context_scope=str(resolved.extra_metadata["context_scope"]),
                    section_id=section_id,
                    window_start=int(resolved.extra_metadata["context_window_start"]),
                    window_end=int(resolved.extra_metadata["context_window_end"]),
                )
            )
        if candidates and not selected:
            raise RuntimeError("no section context fits within max_prompt_tokens")
        self._last_diagnostics = tuple(diagnostics)
        return tuple(selected)


def _largest_fitting_window(
    question: str,
    candidate: SearchChunk,
    selected: Sequence[SearchChunk],
    *,
    tokenizer: TokenizerProtocol,
    max_prompt_tokens: int,
    prompt_serializer: PromptSerializer,
    initial_prompt_builder: Callable[[str, Sequence[SearchChunk]], str],
    retry_prompt_builder: Callable[[str, Sequence[SearchChunk]], str],
) -> SearchChunk | None:
    metadata = candidate.extra_metadata
    matched_start = _required_non_negative_int(metadata, "matched_child_start")
    matched_end = _required_positive_int(metadata, "matched_child_end")
    if matched_start >= matched_end or matched_end > len(candidate.text):
        raise ValueError("matched child range is outside section text")
    spans = _paragraph_spans(candidate.text)
    overlapping = [
        index
        for index, (start, end) in enumerate(spans)
        if start < matched_end and end > matched_start
    ]
    if not overlapping:
        raise ValueError("matched child does not overlap a section paragraph")
    left_index = min(overlapping)
    right_index = max(overlapping)

    def make_window(left: int, right: int) -> SearchChunk:
        start = spans[left][0]
        end = spans[right][1]
        return _context_chunk(
            candidate,
            text=candidate.text[start:end],
            scope="section_window",
            start=start,
            end=end,
        )

    window = make_window(left_index, right_index)
    if not _contexts_fit(
        question,
        (*selected, window),
        tokenizer=tokenizer,
        max_prompt_tokens=max_prompt_tokens,
        prompt_serializer=prompt_serializer,
        initial_prompt_builder=initial_prompt_builder,
        retry_prompt_builder=retry_prompt_builder,
    ):
        window = _context_chunk(
            candidate,
            text=candidate.text[matched_start:matched_end],
            scope="section_window",
            start=matched_start,
            end=matched_end,
        )
        if not _contexts_fit(
            question,
            (*selected, window),
            tokenizer=tokenizer,
            max_prompt_tokens=max_prompt_tokens,
            prompt_serializer=prompt_serializer,
            initial_prompt_builder=initial_prompt_builder,
            retry_prompt_builder=retry_prompt_builder,
        ):
            return None
        return window

    left_open = left_index > 0
    right_open = right_index + 1 < len(spans)
    while left_open or right_open:
        options: list[tuple[int, int, str]] = []
        if left_open:
            options.append((spans[left_index][0] - spans[left_index - 1][0], 0, "left"))
        if right_open:
            options.append((spans[right_index + 1][1] - spans[right_index][1], 1, "right"))
        _size, _tie, side = min(options)
        proposed_left = left_index - 1 if side == "left" else left_index
        proposed_right = right_index + 1 if side == "right" else right_index
        proposed = make_window(proposed_left, proposed_right)
        if _contexts_fit(
            question,
            (*selected, proposed),
            tokenizer=tokenizer,
            max_prompt_tokens=max_prompt_tokens,
            prompt_serializer=prompt_serializer,
            initial_prompt_builder=initial_prompt_builder,
            retry_prompt_builder=retry_prompt_builder,
        ):
            window = proposed
            left_index = proposed_left
            right_index = proposed_right
        elif side == "left":
            left_open = False
        else:
            right_open = False
        left_open = left_open and left_index > 0
        right_open = right_open and right_index + 1 < len(spans)
    return window


def _contexts_fit(
    question: str,
    contexts: Sequence[SearchChunk],
    *,
    tokenizer: TokenizerProtocol,
    max_prompt_tokens: int,
    prompt_serializer: PromptSerializer,
    initial_prompt_builder: Callable[[str, Sequence[SearchChunk]], str],
    retry_prompt_builder: Callable[[str, Sequence[SearchChunk]], str],
) -> bool:
    prompts = (
        prompt_serializer.serialize(initial_prompt_builder(question, contexts)),
        prompt_serializer.serialize(retry_prompt_builder(question, contexts)),
    )
    return all(
        count_prompt_tokens(tokenizer, prompt) <= max_prompt_tokens
        for prompt in prompts
    )


def _context_chunk(
    candidate: SearchChunk,
    *,
    text: str,
    scope: str,
    start: int,
    end: int,
) -> SearchChunk:
    metadata = dict(candidate.extra_metadata)
    metadata.update(
        {
            "context_scope": scope,
            "context_window_start": start,
            "context_window_end": end,
        }
    )
    return SearchChunk(
        text=text,
        page_title=candidate.page_title,
        section_title=candidate.section_title,
        source_url=candidate.source_url,
        category=candidate.category,
        chunk_index=candidate.chunk_index,
        start_index=start,
        extra_metadata=metadata,
    )


def _paragraph_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for match in re.finditer(r"(?:\A|\n[ \t]*\n)(.*?)(?=\n[ \t]*\n|\Z)", text, re.DOTALL):
        raw_start, raw_end = match.span(1)
        value = match.group(1)
        leading = len(value) - len(value.lstrip())
        trailing = len(value) - len(value.rstrip())
        start = raw_start + leading
        end = raw_end - trailing
        if start < end:
            spans.append((start, end))
    return spans or [(0, len(text))]


def _stored_section(section: DocumentSection) -> StoredSection:
    return StoredSection(
        section_id=section_id_for(section),
        text_sha256=text_sha256(section.text),
        section=section,
    )


def _stored_section_from_dict(data: Any) -> StoredSection:
    if not isinstance(data, dict):
        raise TypeError("section record must be an object")
    section = DocumentSection(
        text=_required_record_string(data, "text"),
        page_title=_required_record_string(data, "page_title"),
        section_title=_required_record_string(data, "section_title"),
        source_path=_required_record_string(data, "source_path"),
        source_url=_required_record_string(data, "source_url"),
        anchor=_required_record_string(data, "anchor", allow_empty=True),
        category=_required_record_string(data, "category"),
        python_version=_required_record_string(data, "python_version"),
    )
    expected = _stored_section(section)
    if data.get("section_id") != expected.section_id:
        raise ValueError("section_id does not match canonical metadata")
    if data.get("text_sha256") != expected.text_sha256:
        raise ValueError("section text SHA-256 mismatch")
    return expected


def _validate_child_coverage(
    text: str,
    pieces: Sequence[tuple[str, int]],
    *,
    section_id: str,
) -> None:
    covered = [False] * len(text)
    for child_text, start in pieces:
        if start < 0 or start + len(child_text) > len(text):
            raise ValueError(f"child range outside section: {section_id}")
        for position in range(start, start + len(child_text)):
            covered[position] = True
    if any(
        not character.isspace() and not covered[position]
        for position, character in enumerate(text)
    ):
        raise ValueError(f"child splitting omitted section content: {section_id}")


def _required_record_string(
    data: dict[str, Any],
    name: str,
    *,
    allow_empty: bool = False,
) -> str:
    value = data[name]
    if not isinstance(value, str) or (not allow_empty and not value):
        raise TypeError(f"{name} must be a string")
    return value


def _required_hash(metadata: dict[str, Any], name: str) -> str:
    value = metadata.get(name)
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a 64-character lowercase hex")
    return value


def _required_non_negative_int(metadata: dict[str, Any], name: str) -> int:
    value = metadata.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _required_positive_int(metadata: dict[str, Any], name: str) -> int:
    value = metadata.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_positive_int(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be at least 1")
