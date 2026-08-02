"""Deterministic symbol and field-aware retrieval for technical documents."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlsplit

from python_doc_rag.models import SearchChunk, SearchResult
from python_doc_rag.retrieval import CodeAwareNgramTokenizer, Retriever

TECHNICAL_RETRIEVAL_REVISION = "technical-field-retrieval-v1"
_IDENTIFIER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"[A-Za-z_][A-Za-z0-9_]*"
    r"(?:\.[A-Za-z_][A-Za-z0-9_]*)*"
    r"(?:\[[0-9]+\])?"
    r"(?:\(\))?"
)


@dataclass(frozen=True, slots=True)
class SymbolRecord:
    """Sidecar fields tied to one immutable baseline chunk."""

    source_url: str
    chunk_index: int
    start_index: int
    text_sha256: str
    identifiers: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-compatible record."""
        return {
            "source_url": self.source_url,
            "chunk_index": self.chunk_index,
            "start_index": self.start_index,
            "text_sha256": self.text_sha256,
            "identifiers": list(self.identifiers),
        }


@dataclass(frozen=True, slots=True)
class SymbolArtifactSummary:
    """Counts and hash for a persisted sidecar."""

    chunk_count: int
    identifier_occurrence_count: int
    unique_identifier_count: int
    sha256: str


def extract_identifiers(text: str) -> tuple[str, ...]:
    """Extract Python-shaped identifiers without changing dots or underscores."""
    return tuple(dict.fromkeys(match.group() for match in _IDENTIFIER_PATTERN.finditer(text)))


def identifier_variants(identifier: str) -> tuple[str, ...]:
    """Return call, qualified suffix, and casefold variants deterministically."""
    if not _IDENTIFIER_PATTERN.fullmatch(identifier):
        return ()
    call_suffix = identifier.endswith("()")
    without_call = identifier[:-2] if call_suffix else identifier
    without_subscript = re.sub(r"\[[0-9]+\]$", "", without_call)
    values = [identifier, without_call, without_subscript]
    parts = without_subscript.split(".")
    for offset in range(len(parts)):
        suffix = ".".join(parts[offset:])
        values.append(suffix)
        if call_suffix:
            values.append(f"{suffix}()")
    values.extend(parts)
    expanded: list[str] = []
    for value in dict.fromkeys(item for item in values if item):
        expanded.append(value)
        folded = value.casefold()
        if folded != value:
            expanded.append(folded)
    return tuple(expanded)


def symbol_record_for(chunk: SearchChunk) -> SymbolRecord:
    """Derive sidecar fields only from parser-produced chunk content."""
    parsed = urlsplit(chunk.source_url)
    url_text = unquote(f"{parsed.path} {parsed.fragment}")
    fields = "\n".join(
        (chunk.page_title, chunk.section_title, url_text, chunk.text)
    )
    identifiers: list[str] = []
    for identifier in extract_identifiers(fields):
        identifiers.extend(identifier_variants(identifier))
    return SymbolRecord(
        source_url=chunk.source_url,
        chunk_index=chunk.chunk_index,
        start_index=chunk.start_index,
        text_sha256=_text_sha256(chunk.text),
        identifiers=tuple(dict.fromkeys(identifiers)),
    )


def write_symbol_sidecar_atomic(
    chunks: Sequence[SearchChunk],
    path: Path,
) -> SymbolArtifactSummary:
    """Write a validated symbol sidecar through a sibling temporary file."""
    if not chunks:
        raise ValueError("symbol sidecar requires at least one chunk")
    destination = path.expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    occurrences = 0
    unique: set[str] = set()
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            for chunk in chunks:
                record = symbol_record_for(chunk)
                json.dump(record.to_dict(), stream, ensure_ascii=False)
                stream.write("\n")
                occurrences += len(record.identifiers)
                unique.update(record.identifiers)
            stream.flush()
            os.fsync(stream.fileno())
        sha256 = _sha256(Path(temporary_name))
        os.replace(temporary_name, destination)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return SymbolArtifactSummary(len(chunks), occurrences, len(unique), sha256)


def load_symbol_sidecar(
    chunks: Sequence[SearchChunk],
    path: Path,
) -> tuple[SymbolRecord, ...]:
    """Load a sidecar and reject order, identity, content, or extraction drift."""
    records: list[SymbolRecord] = []
    with path.expanduser().open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                records.append(_record_from_mapping(data))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid symbol sidecar at line {line_number}: {error}"
                ) from error
    if len(records) != len(chunks):
        raise ValueError("symbol sidecar count does not match chunk count")
    for position, (chunk, record) in enumerate(zip(chunks, records, strict=True)):
        expected = symbol_record_for(chunk)
        if record != expected:
            raise ValueError(f"symbol sidecar mismatch at chunk position {position}")
    return tuple(records)


class SymbolRetriever:
    """Rank exact and variant identifier matches from a reusable sidecar."""

    def __init__(
        self,
        chunks: Sequence[SearchChunk],
        records: Sequence[SymbolRecord],
    ) -> None:
        _validate_aligned(chunks, records)
        self._chunks = tuple(chunks)
        postings: dict[str, list[int]] = {}
        for position, record in enumerate(records):
            for identifier in record.identifiers:
                postings.setdefault(identifier, []).append(position)
        self._postings = postings

    def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        """Return exact matches before qualified/suffix variants."""
        _validate_search(query, top_k)
        query_identifiers = extract_identifiers(query)
        exact = tuple(dict.fromkeys(query_identifiers))
        variants = tuple(
            dict.fromkeys(
                variant
                for identifier in exact
                for variant in identifier_variants(identifier)
            )
        )
        scores: Counter[int] = Counter()
        exact_hits: Counter[int] = Counter()
        for identifier in variants:
            for position in self._postings.get(identifier, ()):
                scores[position] += 1
        for identifier in exact:
            for position in self._postings.get(identifier, ()):
                exact_hits[position] += 1
        positions = sorted(
            scores,
            key=lambda position: (
                -exact_hits[position],
                -scores[position],
                position,
            ),
        )[:top_k]
        return [
            _result(
                rank,
                float(exact_hits[position] * 1000 + scores[position]),
                self._chunks[position],
            )
            for rank, position in enumerate(positions, start=1)
        ]

    def retrieve(self, question: str, *, limit: int) -> tuple[SearchChunk, ...]:
        """Return original chunks without adding sidecar metadata."""
        return tuple(result.chunk for result in self.search(question, top_k=limit))


class FieldBM25Retriever:
    """Search one isolated technical-document field with BM25."""

    def __init__(
        self,
        chunks: Sequence[SearchChunk],
        *,
        field: str,
        records: Sequence[SymbolRecord] | None = None,
        tokenizer: CodeAwareNgramTokenizer | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not chunks:
            raise ValueError("field corpus must contain at least one chunk")
        if field not in {"identifiers", "section_title", "page_title", "body"}:
            raise ValueError(f"unsupported retrieval field: {field}")
        if field == "identifiers":
            if records is None:
                raise ValueError("identifier field requires symbol records")
            _validate_aligned(chunks, records)
        self._chunks = tuple(chunks)
        self._field = field
        self._records = tuple(records or ())
        self._tokenizer = tokenizer or CodeAwareNgramTokenizer((2,))
        self._k1 = k1
        self._b = b
        self._lengths: list[int] = []
        postings: dict[str, list[tuple[int, int]]] = {}
        for position, chunk in enumerate(self._chunks):
            tokens = self._document_tokens(position, chunk)
            frequencies = Counter(tokens)
            self._lengths.append(sum(frequencies.values()))
            for token, frequency in frequencies.items():
                postings.setdefault(token, []).append((position, frequency))
        self._postings = postings
        self._average_length = sum(self._lengths) / len(self._lengths) or 1.0

    @property
    def field(self) -> str:
        """Return the isolated field name."""
        return self._field

    def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        """Return deterministic field-local BM25 results."""
        _validate_search(query, top_k)
        query_tokens = self._query_tokens(query)
        scores: Counter[int] = Counter()
        document_count = len(self._chunks)
        for token in tuple(dict.fromkeys(query_tokens)):
            postings = self._postings.get(token)
            if not postings:
                continue
            frequency = len(postings)
            inverse = math.log(
                1.0 + (document_count - frequency + 0.5) / (frequency + 0.5)
            )
            for position, term_frequency in postings:
                ratio = self._lengths[position] / self._average_length
                denominator = term_frequency + self._k1 * (
                    1.0 - self._b + self._b * ratio
                )
                scores[position] += inverse * (
                    term_frequency * (self._k1 + 1.0) / denominator
                )
        positions = sorted(scores, key=lambda item: (-scores[item], item))[:top_k]
        return [
            _result(rank, scores[position], self._chunks[position])
            for rank, position in enumerate(positions, start=1)
        ]

    def retrieve(self, question: str, *, limit: int) -> tuple[SearchChunk, ...]:
        """Return original chunks in field-local rank order."""
        return tuple(result.chunk for result in self.search(question, top_k=limit))

    def _document_tokens(
        self, position: int, chunk: SearchChunk
    ) -> tuple[str, ...]:
        if self._field == "identifiers":
            return self._records[position].identifiers
        selector: Callable[[SearchChunk], str] = {
            "section_title": lambda value: value.section_title,
            "page_title": lambda value: value.page_title,
            "body": lambda value: value.text,
        }[self._field]
        return self._tokenizer.tokenize(selector(chunk))

    def _query_tokens(self, query: str) -> tuple[str, ...]:
        if self._field != "identifiers":
            return self._tokenizer.tokenize(query)
        return tuple(
            dict.fromkeys(
                variant
                for identifier in extract_identifiers(query)
                for variant in identifier_variants(identifier)
            )
        )


class WeightedRankFusionRetriever:
    """Fuse field rankings without directly combining incomparable scores."""

    def __init__(
        self,
        retrievers: Sequence[tuple[str, Retriever, float]],
        *,
        rrf_k: int = 10,
        candidate_k: int = 30,
    ) -> None:
        if len(retrievers) < 2:
            raise ValueError("field rank fusion requires at least two retrievers")
        if rrf_k < 1 or candidate_k < 1:
            raise ValueError("rrf_k and candidate_k must be positive")
        if any(not name or weight <= 0 for name, _retriever, weight in retrievers):
            raise ValueError("field rank fusion names and weights must be positive")
        self._retrievers = tuple(retrievers)
        self._rrf_k = rrf_k
        self._candidate_k = candidate_k

    @property
    def field_weights(self) -> dict[str, float]:
        """Return the frozen field-weight mapping."""
        return {name: weight for name, _retriever, weight in self._retrievers}

    def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        """Return weighted RRF results with deterministic ties."""
        _validate_search(query, top_k)
        scores: Counter[tuple[str, int, int]] = Counter()
        chunks: dict[tuple[str, int, int], SearchChunk] = {}
        best_rank: dict[tuple[str, int, int], int] = {}
        first_seen: dict[tuple[str, int, int], int] = {}
        next_seen = 0
        for _name, retriever, weight in self._retrievers:
            seen: set[tuple[str, int, int]] = set()
            for rank, chunk in enumerate(
                retriever.retrieve(query, limit=self._candidate_k), start=1
            ):
                key = _chunk_key(chunk)
                if key in seen:
                    continue
                seen.add(key)
                chunks.setdefault(key, chunk)
                if key not in first_seen:
                    first_seen[key] = next_seen
                    next_seen += 1
                scores[key] += weight / (self._rrf_k + rank)
                best_rank[key] = min(best_rank.get(key, rank), rank)
        keys = sorted(
            scores,
            key=lambda key: (
                -scores[key],
                best_rank[key],
                first_seen[key],
                key,
            ),
        )[:top_k]
        return [
            _result(rank, scores[key], chunks[key])
            for rank, key in enumerate(keys, start=1)
        ]

    def retrieve(self, question: str, *, limit: int) -> tuple[SearchChunk, ...]:
        """Return fused original chunks for downstream reranking."""
        return tuple(result.chunk for result in self.search(question, top_k=limit))


def _record_from_mapping(value: object) -> SymbolRecord:
    if not isinstance(value, dict):
        raise TypeError("symbol record must be an object")
    identifiers = value["identifiers"]
    if not isinstance(identifiers, list) or not all(
        isinstance(item, str) and item for item in identifiers
    ):
        raise TypeError("identifiers must be a string list")
    record = SymbolRecord(
        source_url=value["source_url"],
        chunk_index=value["chunk_index"],
        start_index=value["start_index"],
        text_sha256=value["text_sha256"],
        identifiers=tuple(identifiers),
    )
    if not isinstance(record.source_url, str) or not record.source_url:
        raise TypeError("source_url must be a non-empty string")
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in (record.chunk_index, record.start_index)
    ):
        raise TypeError("chunk indices must be non-negative integers")
    if not re.fullmatch(r"[0-9a-f]{64}", record.text_sha256):
        raise ValueError("text_sha256 must be lowercase SHA-256")
    return record


def _validate_aligned(
    chunks: Sequence[SearchChunk], records: Sequence[SymbolRecord]
) -> None:
    if not chunks or len(chunks) != len(records):
        raise ValueError("chunks and symbol records must be non-empty and aligned")
    for chunk, record in zip(chunks, records, strict=True):
        if (
            record.source_url,
            record.chunk_index,
            record.start_index,
            record.text_sha256,
        ) != (*_chunk_key(chunk), _text_sha256(chunk.text)):
            raise ValueError("symbol record identity does not match chunk")


def _validate_search(query: str, top_k: int) -> None:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must not be empty or whitespace")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be at least 1")


def _chunk_key(chunk: SearchChunk) -> tuple[str, int, int]:
    return (chunk.source_url, chunk.chunk_index, chunk.start_index)


def _result(rank: int, score: float, chunk: SearchChunk) -> SearchResult:
    return SearchResult(
        rank=rank,
        score=float(score),
        chunk=chunk,
        page_title=chunk.page_title,
        section_title=chunk.section_title,
        source_url=chunk.source_url,
        category=chunk.category,
    )


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
