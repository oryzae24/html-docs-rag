"""Site-neutral protocols and immutable results for HTML ingestion."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from python_doc_rag.models import DocumentSection, SearchChunk, SourceDocument


class DocumentLoader(Protocol):
    """Load immutable source documents in deterministic order."""

    def load(self) -> Iterable[SourceDocument]:
        """Yield decoded source documents without exposing local paths."""
        ...


@dataclass(frozen=True, slots=True)
class DocumentParseResult:
    """Sections and portable diagnostics produced from one source document."""

    sections: tuple[DocumentSection, ...]
    excluded_section_count: int = 0
    excluded_node_count: int = 0
    code_block_count: int = 0
    table_count: int = 0
    used_fallback: bool = False


class HtmlDocumentParser(Protocol):
    """Convert one fetched HTML document into heading-scoped sections."""

    def parse(self, document: SourceDocument) -> DocumentParseResult:
        """Parse one document without changing its trusted source URL."""
        ...


@dataclass(frozen=True, slots=True)
class IngestionFailure:
    """One page-local parser failure that does not stop the dataset."""

    logical_path: str
    reason: str


@dataclass(frozen=True, slots=True)
class IngestionResult:
    """In-memory output and diagnostics from parsing through chunking."""

    documents: tuple[SourceDocument, ...]
    sections: tuple[DocumentSection, ...]
    chunks: tuple[SearchChunk, ...]
    failures: tuple[IngestionFailure, ...]
    excluded_section_count: int
    excluded_node_count: int
    code_block_count: int
    table_count: int
    fallback_page_count: int
