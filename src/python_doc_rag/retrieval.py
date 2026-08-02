"""Interfaces for retrieving document chunks for a question."""

import math
import re
from collections import Counter
from collections.abc import Sequence
from typing import Protocol

from python_doc_rag.models import SearchChunk, SearchResult
from python_doc_rag.vector_store import VectorIndex


class Retriever(Protocol):
    """Retrieve ranked chunks while preserving their source metadata."""

    def retrieve(self, question: str, *, limit: int) -> Sequence[SearchChunk]:
        """Return the chunks most relevant to a natural-language question."""
        ...


def search(
    vector_index: VectorIndex,
    query: str,
    *,
    top_k: int = 5,
) -> list[SearchResult]:
    """Search a loaded vector index through a small functional interface."""
    return vector_index.search(query, top_k=top_k)


class VectorIndexRetriever:
    """Adapt the existing vector index to the retrieval protocol."""

    def __init__(self, vector_index: VectorIndex) -> None:
        self._vector_index = vector_index

    def retrieve(self, question: str, *, limit: int) -> tuple[SearchChunk, ...]:
        """Return chunks in the ranking order produced by the vector index."""
        results = search(self._vector_index, question, top_k=limit)
        return tuple(result.chunk for result in results)


_CODE_IDENTIFIER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"[A-Za-z_][A-Za-z0-9_]*"
    r"(?:\.[A-Za-z_][A-Za-z0-9_]*)*"
    r"(?:\[[0-9]+\])?"
    r"(?:\(\))?"
)
_JAPANESE_RUN_PATTERN = re.compile(
    r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff々〆ヵヶー]+"
)


class CodeAwareNgramTokenizer:
    """Tokenize Japanese text while retaining complete Python identifiers."""

    def __init__(self, ngram_sizes: Sequence[int] = (2, 3)) -> None:
        sizes = tuple(dict.fromkeys(ngram_sizes))
        if not sizes or any(
            isinstance(size, bool) or not isinstance(size, int) or size < 1
            for size in sizes
        ):
            raise ValueError("ngram_sizes must contain positive integers")
        self._ngram_sizes = sizes

    @property
    def ngram_sizes(self) -> tuple[int, ...]:
        """Return the configured Japanese character n-gram sizes."""
        return self._ngram_sizes

    def tokenize(self, text: str) -> tuple[str, ...]:
        """Return code-aware ASCII tokens and Japanese character n-grams."""
        tokens: list[str] = []
        for match in _CODE_IDENTIFIER_PATTERN.finditer(text):
            tokens.extend(_identifier_variants(match.group()))
        for match in _JAPANESE_RUN_PATTERN.finditer(text):
            run = match.group()
            for size in self._ngram_sizes:
                tokens.extend(
                    run[offset : offset + size] for offset in range(len(run) - size + 1)
                )
        return tuple(tokens)


class BM25Retriever:
    """Search an in-memory corpus with BM25 and code-aware tokenization."""

    def __init__(
        self,
        chunks: Sequence[SearchChunk],
        *,
        tokenizer: CodeAwareNgramTokenizer | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not chunks:
            raise ValueError("BM25 corpus must contain at least one chunk")
        if isinstance(k1, bool) or not isinstance(k1, int | float) or k1 <= 0:
            raise ValueError("k1 must be greater than zero")
        if isinstance(b, bool) or not isinstance(b, int | float) or not 0 <= b <= 1:
            raise ValueError("b must be between zero and one")

        self._chunks = tuple(chunks)
        self._tokenizer = tokenizer or CodeAwareNgramTokenizer()
        self._k1 = float(k1)
        self._b = float(b)
        self._document_lengths: list[int] = []
        postings: dict[str, list[tuple[int, int]]] = {}
        for position, chunk in enumerate(self._chunks):
            frequencies = Counter(self._tokenizer.tokenize(_searchable_text(chunk)))
            self._document_lengths.append(sum(frequencies.values()))
            for token, frequency in frequencies.items():
                postings.setdefault(token, []).append((position, frequency))
        self._postings = postings
        average_length = sum(self._document_lengths) / len(self._document_lengths)
        self._average_document_length = average_length or 1.0

    @property
    def chunk_count(self) -> int:
        """Return the number of indexed chunks."""
        return len(self._chunks)

    @property
    def tokenizer(self) -> CodeAwareNgramTokenizer:
        """Return the tokenizer used to build this index."""
        return self._tokenizer

    def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        """Return BM25-ranked chunks with deterministic tie ordering."""
        _validate_question(query)
        _validate_positive_int(top_k, name="top_k")
        query_tokens = tuple(dict.fromkeys(self._tokenizer.tokenize(query)))
        if not query_tokens:
            raise ValueError("query produced no searchable tokens")

        scores = [0.0] * self.chunk_count
        document_count = self.chunk_count
        for token in query_tokens:
            postings = self._postings.get(token)
            if not postings:
                continue
            document_frequency = len(postings)
            inverse_document_frequency = math.log(
                1.0
                + (document_count - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            for position, term_frequency in postings:
                length_ratio = (
                    self._document_lengths[position] / self._average_document_length
                )
                denominator = term_frequency + self._k1 * (
                    1.0 - self._b + self._b * length_ratio
                )
                scores[position] += inverse_document_frequency * (
                    term_frequency * (self._k1 + 1.0) / denominator
                )

        limit = min(top_k, self.chunk_count)
        positions = sorted(
            (position for position, score in enumerate(scores) if score > 0.0),
            key=lambda position: (-scores[position], position),
        )[:limit]
        return [
            _search_result(rank, scores[position], self._chunks[position])
            for rank, position in enumerate(positions, start=1)
        ]

    def retrieve(self, question: str, *, limit: int) -> tuple[SearchChunk, ...]:
        """Return chunks in BM25 rank order."""
        return tuple(result.chunk for result in self.search(question, top_k=limit))


class ReciprocalRankFusionRetriever:
    """Fuse multiple retriever rankings with Reciprocal Rank Fusion."""

    def __init__(
        self,
        retrievers: Sequence[Retriever],
        *,
        rrf_k: int = 60,
        candidate_k: int = 20,
    ) -> None:
        if len(retrievers) < 2:
            raise ValueError("RRF requires at least two retrievers")
        _validate_positive_int(rrf_k, name="rrf_k")
        _validate_positive_int(candidate_k, name="candidate_k")
        self._retrievers = tuple(retrievers)
        self._rrf_k = rrf_k
        self._candidate_k = candidate_k

    @property
    def rrf_k(self) -> int:
        """Return the RRF rank constant."""
        return self._rrf_k

    @property
    def candidate_k(self) -> int:
        """Return the candidate count requested from each retriever."""
        return self._candidate_k

    def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        """Return fused results with RRF scores and preserved metadata."""
        _validate_positive_int(top_k, name="top_k")
        ranked = self._fuse(query, limit=top_k)
        return [
            _search_result(rank, score, chunk)
            for rank, (chunk, score) in enumerate(ranked, start=1)
        ]

    def retrieve(self, question: str, *, limit: int) -> tuple[SearchChunk, ...]:
        """Return chunks in fused rank order."""
        return tuple(chunk for chunk, _score in self._fuse(question, limit=limit))

    def _fuse(
        self,
        question: str,
        *,
        limit: int,
    ) -> list[tuple[SearchChunk, float]]:
        _validate_question(question)
        _validate_positive_int(limit, name="limit")
        scores: dict[tuple[str, int, int], float] = {}
        chunks: dict[tuple[str, int, int], SearchChunk] = {}
        first_seen: dict[tuple[str, int, int], int] = {}
        best_rank: dict[tuple[str, int, int], int] = {}
        next_seen = 0

        for retriever in self._retrievers:
            seen_in_retriever: set[tuple[str, int, int]] = set()
            candidates = retriever.retrieve(question, limit=self._candidate_k)
            for rank, chunk in enumerate(candidates[: self._candidate_k], start=1):
                key = _chunk_key(chunk)
                if key in seen_in_retriever:
                    continue
                seen_in_retriever.add(key)
                if key not in chunks:
                    chunks[key] = chunk
                    first_seen[key] = next_seen
                    next_seen += 1
                scores[key] = scores.get(key, 0.0) + 1.0 / (self._rrf_k + rank)
                best_rank[key] = min(best_rank.get(key, rank), rank)

        ordered_keys = sorted(
            scores,
            key=lambda key: (
                -scores[key],
                best_rank[key],
                first_seen[key],
                key,
            ),
        )[:limit]
        return [(chunks[key], scores[key]) for key in ordered_keys]


def _identifier_variants(identifier: str) -> tuple[str, ...]:
    variants: list[str] = [identifier]
    has_call = identifier.endswith("()")
    without_call = identifier[:-2] if has_call else identifier
    if has_call:
        variants.append(without_call)
    without_subscript = re.sub(r"\[[0-9]+\]$", "", without_call)
    if without_subscript != without_call:
        variants.append(without_subscript)

    parts = without_subscript.split(".")
    for offset in range(len(parts)):
        suffix = ".".join(parts[offset:])
        variants.append(suffix)
        if has_call:
            variants.append(f"{suffix}()")
    variants.extend(parts)

    expanded: list[str] = []
    for variant in dict.fromkeys(variants):
        expanded.append(variant)
        folded = variant.casefold()
        if folded != variant:
            expanded.append(folded)
    return tuple(expanded)


def _searchable_text(chunk: SearchChunk) -> str:
    return (
        f"ページタイトル: {chunk.page_title}\n"
        f"セクションタイトル: {chunk.section_title}\n"
        f"本文: {chunk.text}"
    )


def _chunk_key(chunk: SearchChunk) -> tuple[str, int, int]:
    return (chunk.source_url, chunk.chunk_index, chunk.start_index)


def _search_result(rank: int, score: float, chunk: SearchChunk) -> SearchResult:
    return SearchResult(
        rank=rank,
        score=score,
        chunk=chunk,
        page_title=chunk.page_title,
        section_title=chunk.section_title,
        source_url=chunk.source_url,
        category=chunk.category,
    )


def _validate_question(question: str) -> None:
    if not question.strip():
        raise ValueError("query must not be empty or whitespace")


def _validate_positive_int(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be at least 1")
