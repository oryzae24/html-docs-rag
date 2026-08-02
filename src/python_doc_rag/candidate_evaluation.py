"""Candidate-recall metrics separated from downstream reranking quality."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any, Protocol
from urllib.parse import urldefrag

from python_doc_rag.evaluation import EvaluationQuestion
from python_doc_rag.models import SearchResult

CANDIDATE_EVALUATION_REVISION = "candidate-recall-evaluation-v1"


class CandidateSearcher(Protocol):
    """Search interface used before a reranker changes result order."""

    def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        """Return candidate-generation results."""
        ...


def evaluate_candidate_recall(
    searcher: CandidateSearcher,
    questions: Sequence[EvaluationQuestion],
    *,
    candidate_k: int = 30,
    hard_case_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Measure candidate presence and diversity before any reranking."""
    if not questions:
        raise ValueError("at least one evaluation question is required")
    if candidate_k < 30:
        raise ValueError("candidate_k must be at least 30")
    hard_ids = frozenset(hard_case_ids)
    rows: list[dict[str, Any]] = []
    for item in questions:
        started_at = perf_counter()
        results = tuple(searcher.search(item.question, top_k=candidate_k))
        elapsed = perf_counter() - started_at
        rank = _first_relevant_rank(results, item.expected_url_keywords)
        rows.append(
            {
                "id": item.id,
                "question": item.question,
                "query_type": item.query_type,
                "topic": item.topic,
                "hard_case": item.id in hard_ids,
                "expected_url_keywords": list(item.expected_url_keywords),
                "candidate_count": len(results),
                "recall_at_10": rank is not None and rank <= 10,
                "recall_at_20": rank is not None and rank <= 20,
                "recall_at_30": rank is not None and rank <= 30,
                "expected_url_in_candidates": rank is not None,
                "first_relevant_rank": rank,
                "unique_url_count": len({result.source_url for result in results}),
                "unique_page_count": len(
                    {urldefrag(result.source_url).url for result in results}
                ),
                "unique_section_count": len(
                    {(result.source_url, result.section_title) for result in results}
                ),
                "candidate_generation_seconds": elapsed,
                "results": [
                    {
                        "rank": result.rank,
                        "score": result.score,
                        "page_title": result.page_title,
                        "section_title": result.section_title,
                        "source_url": result.source_url,
                        "chunk_index": result.chunk.chunk_index,
                        "start_index": result.chunk.start_index,
                    }
                    for result in results
                ],
            }
        )
    return {
        "revision": CANDIDATE_EVALUATION_REVISION,
        "summary": _summarize(rows),
        "query_types": {
            query_type: _summarize(
                [row for row in rows if row["query_type"] == query_type]
            )
            for query_type in sorted(
                {str(row["query_type"]) for row in rows if row["query_type"]}
            )
        },
        "hard_cases": {
            str(row["id"]): {
                "recall_at_30": row["recall_at_30"],
                "first_relevant_rank": row["first_relevant_rank"],
            }
            for row in rows
            if row["hard_case"]
        },
        "questions": rows,
    }


def save_candidate_evaluation_atomic(
    payload: Mapping[str, Any],
    path: Path,
) -> None:
    """Atomically persist one UTF-8 candidate evaluation snapshot."""
    destination = path.expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, int | float]:
    count = len(rows)
    timings = [float(row["candidate_generation_seconds"]) for row in rows]
    if not count:
        return {
            "question_count": 0,
            "recall_at_10": 0.0,
            "recall_at_20": 0.0,
            "recall_at_30": 0.0,
            "expected_url_candidate_rate": 0.0,
            "average_unique_url_count": 0.0,
            "average_unique_page_count": 0.0,
            "average_unique_section_count": 0.0,
            "average_candidate_generation_seconds": 0.0,
            "median_candidate_generation_seconds": 0.0,
        }
    return {
        "question_count": count,
        "recall_at_10": _rate(rows, "recall_at_10"),
        "recall_at_20": _rate(rows, "recall_at_20"),
        "recall_at_30": _rate(rows, "recall_at_30"),
        "expected_url_candidate_rate": _rate(rows, "expected_url_in_candidates"),
        "average_unique_url_count": _mean(rows, "unique_url_count"),
        "average_unique_page_count": _mean(rows, "unique_page_count"),
        "average_unique_section_count": _mean(rows, "unique_section_count"),
        "average_candidate_generation_seconds": sum(timings) / count,
        "median_candidate_generation_seconds": median(timings),
    }


def _rate(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return sum(bool(row[field]) for row in rows) / len(rows)


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return sum(float(row[field]) for row in rows) / len(rows)


def _first_relevant_rank(
    results: Sequence[SearchResult], expected_keywords: Sequence[str]
) -> int | None:
    for rank, result in enumerate(results, start=1):
        if any(keyword in result.source_url for keyword in expected_keywords):
            return rank
    return None
