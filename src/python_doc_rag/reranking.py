"""Local cross-encoder reranking with preserved citation metadata."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol

import numpy as np

from python_doc_rag.models import SearchChunk, SearchResult

RERANKING_REVISION = "local-cross-encoder-rerank-v1"


@dataclass(frozen=True, slots=True)
class RerankerModelSpec:
    """Pinned model-card provenance for one bounded experiment candidate."""

    key: str
    model_name: str
    revision: str
    license: str
    model_card_url: str
    language_evidence: str


RERANKER_MODEL_SPECS = (
    RerankerModelSpec(
        key="bge-m3",
        model_name="BAAI/bge-reranker-v2-m3",
        revision="953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
        license="Apache-2.0",
        model_card_url="https://huggingface.co/BAAI/bge-reranker-v2-m3",
        language_evidence="Model card identifies the reranker as multilingual.",
    ),
    RerankerModelSpec(
        key="mmarco-minilm",
        model_name="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        revision="1427fd652930e4ba29e8149678df786c240d8825",
        license="Apache-2.0",
        model_card_url=(
            "https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
        ),
        language_evidence="Model card language metadata explicitly includes Japanese.",
    ),
)


class CandidateSearcher(Protocol):
    """Search interface that supplies candidates without reranking them."""

    def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        """Return candidates in original deterministic rank order."""
        ...


class PairScorer(Protocol):
    """Batch scoring boundary implemented by a local cross-encoder."""

    def score(
        self,
        pairs: Sequence[tuple[str, str]],
        *,
        batch_size: int,
    ) -> Sequence[float]:
        """Return one finite relevance score per query-document pair."""
        ...


@dataclass(frozen=True, slots=True)
class RerankMatchDiagnostic:
    """One reranked item with its score and original position."""

    rank: int
    original_rank: int
    rerank_score: float
    original_score: float
    source_url: str
    chunk_index: int
    start_index: int


@dataclass(frozen=True, slots=True)
class RerankTrace:
    """Diagnostics retained outside model-visible generation contexts."""

    candidate_count: int
    batch_size: int
    scoring_seconds: float
    matches: tuple[RerankMatchDiagnostic, ...]


class CrossEncoderPairScorer:
    """Load one Sentence Transformers CrossEncoder and reuse it across queries."""

    def __init__(self, model: Any) -> None:
        self._model = model

    @classmethod
    def from_pretrained(
        cls,
        spec: RerankerModelSpec,
        *,
        device: str,
        max_length: int = 512,
    ) -> CrossEncoderPairScorer:
        """Load a pinned local model without executing repository code."""
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(
            spec.model_name,
            revision=spec.revision,
            device=device,
            trust_remote_code=False,
            max_length=max_length,
        )
        return cls(model)

    def score(
        self,
        pairs: Sequence[tuple[str, str]],
        *,
        batch_size: int,
    ) -> tuple[float, ...]:
        """Score a complete candidate batch without exposing extra metadata."""
        _validate_positive_int(batch_size, name="batch_size")
        if not pairs:
            return ()
        values = self._model.predict(
            list(pairs),
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        array = np.asarray(values).reshape(-1)
        return tuple(float(value) for value in array)


class RerankingRetriever:
    """Rerank a fixed candidate set and return unchanged citation-ready chunks."""

    def __init__(
        self,
        candidate_searcher: CandidateSearcher,
        scorer: PairScorer,
        *,
        candidate_k: int,
        batch_size: int,
    ) -> None:
        _validate_positive_int(candidate_k, name="candidate_k")
        _validate_positive_int(batch_size, name="batch_size")
        self._candidate_searcher = candidate_searcher
        self._scorer = scorer
        self._candidate_k = candidate_k
        self._batch_size = batch_size
        self._last_trace = RerankTrace(0, batch_size, 0.0, ())

    @property
    def candidate_k(self) -> int:
        """Return the bounded number of candidates sent to the scorer."""
        return self._candidate_k

    @property
    def last_trace(self) -> RerankTrace:
        """Return diagnostics from the most recent query."""
        return self._last_trace

    def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        """Return top results sorted by score, then stable original rank."""
        _validate_query(query)
        _validate_positive_int(top_k, name="top_k")
        candidates = self._candidate_searcher.search(
            query,
            top_k=self._candidate_k,
        )
        pairs = tuple(
            (query, _reranker_document(result.chunk)) for result in candidates
        )
        started_at = perf_counter()
        scores = tuple(self._scorer.score(pairs, batch_size=self._batch_size))
        scoring_seconds = perf_counter() - started_at
        _validate_scores(scores, expected=len(candidates))
        ranked = sorted(
            zip(candidates, scores, strict=True),
            key=lambda item: (
                -item[1],
                item[0].rank,
                _chunk_key(item[0].chunk),
            ),
        )
        selected = ranked[:top_k]
        results: list[SearchResult] = []
        matches: list[RerankMatchDiagnostic] = []
        for rank, (original, rerank_score) in enumerate(selected, start=1):
            chunk = original.chunk
            results.append(
                SearchResult(
                    rank=rank,
                    score=rerank_score,
                    chunk=chunk,
                    page_title=chunk.page_title,
                    section_title=chunk.section_title,
                    source_url=chunk.source_url,
                    category=chunk.category,
                )
            )
            matches.append(
                RerankMatchDiagnostic(
                    rank=rank,
                    original_rank=original.rank,
                    rerank_score=rerank_score,
                    original_score=original.score,
                    source_url=chunk.source_url,
                    chunk_index=chunk.chunk_index,
                    start_index=chunk.start_index,
                )
            )
        self._last_trace = RerankTrace(
            candidate_count=len(candidates),
            batch_size=self._batch_size,
            scoring_seconds=scoring_seconds,
            matches=tuple(matches),
        )
        return results

    def retrieve(self, question: str, *, limit: int) -> tuple[SearchChunk, ...]:
        """Return unchanged chunks for the RAG pipeline."""
        return tuple(result.chunk for result in self.search(question, top_k=limit))


def model_spec_for_key(key: str) -> RerankerModelSpec:
    """Resolve one of the two bounded model candidates."""
    for spec in RERANKER_MODEL_SPECS:
        if spec.key == key:
            return spec
    raise ValueError(f"unsupported reranker model key: {key}")


def _reranker_document(chunk: SearchChunk) -> str:
    return (
        f"ページタイトル: {chunk.page_title}\n"
        f"セクションタイトル: {chunk.section_title}\n"
        f"本文: {chunk.text}"
    )


def _chunk_key(chunk: SearchChunk) -> tuple[str, int, int]:
    return (chunk.source_url, chunk.chunk_index, chunk.start_index)


def _validate_scores(scores: Sequence[float], *, expected: int) -> None:
    if len(scores) != expected:
        raise RuntimeError(
            f"reranker returned {len(scores)} scores for {expected} candidates"
        )
    if any(not math.isfinite(score) for score in scores):
        raise RuntimeError("reranker returned a non-finite score")


def _validate_query(query: str) -> None:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must not be empty or whitespace")


def _validate_positive_int(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be at least 1")
