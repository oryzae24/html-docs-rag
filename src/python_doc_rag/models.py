"""Serializable data models shared by ingestion and the RAG pipeline."""

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any
from urllib.parse import urlsplit

type JsonScalar = None | bool | int | float | str
type JsonValue = JsonScalar | tuple["JsonValue", ...] | Mapping[str, "JsonValue"]


@dataclass(frozen=True, slots=True)
class SourceDocument:
    """One immutable fetched HTML document with portable source metadata."""

    source_url: str
    canonical_url: str
    content: str
    content_sha256: str
    source_kind: str
    logical_path: str
    category: str
    fetched_at: str | None = None
    metadata: Mapping[str, JsonValue] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        """Validate identity and defensively freeze JSON-compatible metadata."""
        for label, value in (
            ("source_url", self.source_url),
            ("canonical_url", self.canonical_url),
        ):
            _validate_absolute_http_url(value, label)
        if not isinstance(self.content, str):
            raise TypeError("content must be a UTF-8 decoded string")
        expected_hash = hashlib.sha256(self.content.encode("utf-8")).hexdigest()
        if self.content_sha256 != expected_hash:
            raise ValueError("content_sha256 does not match content")
        logical_path = PurePosixPath(self.logical_path.replace("\\", "/"))
        if (
            not self.logical_path
            or logical_path.is_absolute()
            or ".." in logical_path.parts
            or logical_path == PurePosixPath(".")
        ):
            raise ValueError("logical_path must be a safe relative POSIX path")
        if not self.source_kind.strip() or not self.category.strip():
            raise ValueError("source_kind and category must not be empty")
        if self.fetched_at is not None and not isinstance(self.fetched_at, str):
            raise TypeError("fetched_at must be a string or None")
        frozen = _freeze_json_mapping(self.metadata)
        object.__setattr__(self, "metadata", frozen)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation without local paths."""
        return {
            "source_url": self.source_url,
            "canonical_url": self.canonical_url,
            "content": self.content,
            "content_sha256": self.content_sha256,
            "source_kind": self.source_kind,
            "logical_path": self.logical_path,
            "category": self.category,
            "fetched_at": self.fetched_at,
            "metadata": _thaw_json(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class DocumentSection:
    """A heading-scoped section extracted from a source document."""

    text: str
    page_title: str
    section_title: str
    source_path: str
    source_url: str
    anchor: str
    category: str
    python_version: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SearchChunk:
    """A searchable text chunk with citation metadata."""

    text: str
    page_title: str
    section_title: str
    source_url: str
    category: str
    chunk_index: int
    start_index: int
    extra_metadata: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        """Reject citation metadata that could expose a local path."""
        _validate_absolute_http_url(self.source_url, "source_url")

    def to_dict(self) -> dict[str, Any]:
        """Return the original flat JSON representation, including extra metadata."""
        data = {
            "text": self.text,
            "page_title": self.page_title,
            "section_title": self.section_title,
            "source_url": self.source_url,
            "category": self.category,
            "chunk_index": self.chunk_index,
            "start_index": self.start_index,
        }
        data.update(self.extra_metadata)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SearchChunk":
        """Restore a chunk while preserving unrecognized metadata fields."""
        required = {
            "text",
            "page_title",
            "section_title",
            "source_url",
            "category",
            "chunk_index",
            "start_index",
        }
        missing = sorted(required.difference(data))
        if missing:
            raise KeyError(", ".join(missing))

        string_fields = (
            "text",
            "page_title",
            "section_title",
            "source_url",
            "category",
        )
        for name in string_fields:
            if not isinstance(data[name], str):
                raise TypeError(f"{name} must be a string")
        for name in ("chunk_index", "start_index"):
            if not isinstance(data[name], int) or isinstance(data[name], bool):
                raise TypeError(f"{name} must be an integer")

        return cls(
            text=data["text"],
            page_title=data["page_title"],
            section_title=data["section_title"],
            source_url=data["source_url"],
            category=data["category"],
            chunk_index=data["chunk_index"],
            start_index=data["start_index"],
            extra_metadata={
                key: value for key, value in data.items() if key not in required
            },
        )


@dataclass(frozen=True, slots=True)
class SearchResult:
    """One ranked vector-search result with citation-ready metadata."""

    rank: int
    score: float
    chunk: SearchChunk
    page_title: str
    section_title: str
    source_url: str
    category: str


@dataclass(frozen=True, slots=True)
class CitationSource:
    """A source entry derived exclusively from a retrieved chunk."""

    label: str
    page_title: str
    section_title: str
    url: str


@dataclass(frozen=True, slots=True)
class CitedAnswer:
    """A generated answer with separately controlled citation metadata."""

    answer_text: str
    sources: tuple[CitationSource, ...]
    retrieved_chunks: tuple[SearchChunk, ...]
    generation_attempts: int

    def __post_init__(self) -> None:
        """Reject structurally invalid answer values."""
        if not isinstance(self.answer_text, str) or not self.answer_text.strip():
            raise ValueError("answer_text must be a non-empty string")
        if (
            not isinstance(self.generation_attempts, int)
            or isinstance(self.generation_attempts, bool)
            or self.generation_attempts < 0
        ):
            raise ValueError("generation_attempts must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class AbstainedAnswer:
    """A normal no-answer outcome with no user-visible source list."""

    reason_code: str
    retrieved_chunks: tuple[SearchChunk, ...]
    generation_attempts: int

    def __post_init__(self) -> None:
        """Reject arbitrary reasons and invalid generation counters."""
        if self.reason_code not in {
            "insufficient_evidence",
            "no_retrieval_results",
        }:
            raise ValueError(f"unsupported abstention reason: {self.reason_code}")
        if (
            not isinstance(self.generation_attempts, int)
            or isinstance(self.generation_attempts, bool)
            or self.generation_attempts < 0
        ):
            raise ValueError("generation_attempts must be a non-negative integer")


AnswerOutcome = CitedAnswer | AbstainedAnswer


def _validate_absolute_http_url(value: str, label: str) -> None:
    """Require one absolute HTTP(S) URL with a network location."""
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{label} must be an absolute HTTP(S) URL")


def _freeze_json_mapping(value: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    if not isinstance(value, Mapping):
        raise TypeError("metadata must be a mapping")
    frozen: dict[str, JsonValue] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise TypeError("metadata keys must be non-empty strings")
        frozen[key] = _freeze_json(item)
    return MappingProxyType(frozen)


def _freeze_json(value: object) -> JsonValue:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("metadata floats must be finite")
        return value
    if isinstance(value, Mapping):
        return _freeze_json_mapping(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_json(item) for item in value)
    raise TypeError(f"metadata value is not JSON-compatible: {type(value).__name__}")


def _thaw_json(value: JsonValue) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value
