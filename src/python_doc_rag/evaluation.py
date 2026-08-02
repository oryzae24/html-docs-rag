"""URL-keyword ranking evaluation for document retrieval."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any, Protocol

from python_doc_rag.models import SearchResult

_QUERY_TYPES = frozenset({"exact_identifier", "conceptual", "operational"})


class SearchProtocol(Protocol):
    """Search interface required by the retrieval evaluator."""

    def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        """Return ranked search results for a question."""
        ...


@dataclass(frozen=True, slots=True)
class EvaluationQuestion:
    """A question and URL substrings accepted as relevant evidence."""

    question: str
    expected_url_keywords: tuple[str, ...]
    id: str | None = None
    query_type: str | None = None
    topic: str | None = None


@dataclass(frozen=True, slots=True)
class QuestionEvaluation:
    """Search results and URL-match flags for one evaluation question."""

    question: str
    expected_url_keywords: tuple[str, ...]
    results: tuple[SearchResult, ...]
    top_1_hit: bool
    top_3_hit: bool
    top_5_hit: bool
    top_10_hit: bool
    first_relevant_rank: int | None
    reciprocal_rank: float
    retrieval_seconds: float
    id: str | None = None
    query_type: str | None = None
    topic: str | None = None


@dataclass(frozen=True, slots=True)
class RankingMetrics:
    """Hit rates and reciprocal rank for one group of questions."""

    question_count: int
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    hit_at_10: float
    mrr_at_10: float

    def to_dict(self) -> dict[str, int | float]:
        """Return a JSON-serializable metric mapping."""
        return {
            "question_count": self.question_count,
            "hit_at_1": self.hit_at_1,
            "hit_at_3": self.hit_at_3,
            "hit_at_5": self.hit_at_5,
            "hit_at_10": self.hit_at_10,
            "mrr_at_10": self.mrr_at_10,
        }


@dataclass(frozen=True, slots=True)
class TopicMetrics:
    """Hit@5 for one topic."""

    question_count: int
    hit_at_5: float

    def to_dict(self) -> dict[str, int | float]:
        """Return a JSON-serializable metric mapping."""
        return {"question_count": self.question_count, "hit_at_5": self.hit_at_5}


@dataclass(frozen=True, slots=True)
class RetrievalEvaluation:
    """Aggregate hit rates and per-question retrieval details."""

    top_1_hit_rate: float
    top_3_hit_rate: float
    top_5_hit_rate: float
    top_10_hit_rate: float
    mrr_at_10: float
    question_count: int
    average_retrieval_seconds: float
    median_retrieval_seconds: float
    query_type_metrics: Mapping[str, RankingMetrics]
    topic_metrics: Mapping[str, TopicMetrics]
    unmatched_top_10: tuple[str, ...]
    questions: tuple[QuestionEvaluation, ...]


@dataclass(frozen=True, slots=True)
class RetrievalComparison:
    """Question-level rank movements between baseline and one experiment."""

    improved_count: int
    tied_count: int
    worsened_count: int
    entered_top_5: tuple[str, ...]
    left_top_5: tuple[str, ...]
    entered_top_10: tuple[str, ...]
    left_top_10: tuple[str, ...]


def compare_retrieval_reports(
    baseline: RetrievalEvaluation,
    experiment: RetrievalEvaluation,
) -> RetrievalComparison:
    """Compare aligned question IDs without changing either report."""
    baseline_by_id = _questions_by_stable_id(baseline.questions)
    experiment_by_id = _questions_by_stable_id(experiment.questions)
    if baseline_by_id.keys() != experiment_by_id.keys():
        raise ValueError("baseline and experiment question IDs must match")

    improved = 0
    tied = 0
    worsened = 0
    entered_top_5: list[str] = []
    left_top_5: list[str] = []
    entered_top_10: list[str] = []
    left_top_10: list[str] = []
    for question_id, baseline_item in baseline_by_id.items():
        experiment_item = experiment_by_id[question_id]
        baseline_rank = baseline_item.first_relevant_rank or float("inf")
        experiment_rank = experiment_item.first_relevant_rank or float("inf")
        if experiment_rank < baseline_rank:
            improved += 1
        elif experiment_rank > baseline_rank:
            worsened += 1
        else:
            tied += 1
        if baseline_rank > 5 >= experiment_rank:
            entered_top_5.append(question_id)
        if experiment_rank > 5 >= baseline_rank:
            left_top_5.append(question_id)
        if baseline_rank > 10 >= experiment_rank:
            entered_top_10.append(question_id)
        if experiment_rank > 10 >= baseline_rank:
            left_top_10.append(question_id)
    return RetrievalComparison(
        improved,
        tied,
        worsened,
        tuple(entered_top_5),
        tuple(left_top_5),
        tuple(entered_top_10),
        tuple(left_top_10),
    )


def load_evaluation_questions(path: Path) -> list[EvaluationQuestion]:
    """Load and validate UTF-8 JSONL evaluation questions."""
    questions: list[EvaluationQuestion] = []
    ids: set[str] = set()
    with path.expanduser().open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON in {path} at line {line_number}: {error.msg}"
                ) from error
            if not isinstance(data, dict):
                raise ValueError(
                    f"invalid evaluation question at line {line_number}: "
                    "expected a JSON object"
                )
            question = data.get("question")
            keywords = data.get("expected_url_keywords")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(
                    f"invalid evaluation question at line {line_number}: "
                    "question must not be empty"
                )
            if (
                not isinstance(keywords, list)
                or not keywords
                or not all(
                    isinstance(keyword, str) and keyword.strip() for keyword in keywords
                )
            ):
                raise ValueError(
                    f"invalid evaluation question at line {line_number}: "
                    "expected_url_keywords must be a non-empty string list"
                )
            metadata_values = tuple(
                data.get(name) for name in ("id", "query_type", "topic")
            )
            has_metadata = any(value is not None for value in metadata_values)
            if has_metadata and not all(
                isinstance(value, str) and value.strip() for value in metadata_values
            ):
                raise ValueError(
                    f"invalid evaluation question at line {line_number}: "
                    "id, query_type, and topic must all be non-empty strings"
                )
            question_id: str | None = None
            query_type: str | None = None
            topic: str | None = None
            if has_metadata:
                question_id, query_type, topic = (
                    str(value).strip() for value in metadata_values
                )
                if query_type not in _QUERY_TYPES:
                    raise ValueError(
                        f"invalid evaluation question at line {line_number}: "
                        f"unsupported query_type: {query_type}"
                    )
                if question_id in ids:
                    raise ValueError(f"duplicate evaluation question id: {question_id}")
                ids.add(question_id)
            questions.append(
                EvaluationQuestion(
                    question.strip(),
                    tuple(keyword.strip() for keyword in keywords),
                    question_id,
                    query_type,
                    topic,
                )
            )
    return questions


def evaluate_retrieval(
    searcher: SearchProtocol,
    questions: Sequence[EvaluationQuestion],
    *,
    top_k: int = 10,
) -> RetrievalEvaluation:
    """Calculate Hit@1/3/5/10 and MRR@10 while retaining ranked details."""
    if not questions:
        raise ValueError("at least one evaluation question is required")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be at least 1")

    details: list[QuestionEvaluation] = []
    for item in questions:
        started_at = perf_counter()
        results = tuple(searcher.search(item.question, top_k=top_k))
        retrieval_seconds = perf_counter() - started_at
        first_relevant_rank = _first_relevant_rank(
            results[:10], item.expected_url_keywords
        )
        details.append(
            QuestionEvaluation(
                question=item.question,
                expected_url_keywords=item.expected_url_keywords,
                results=results,
                top_1_hit=_has_url_hit(results[:1], item.expected_url_keywords),
                top_3_hit=_has_url_hit(results[:3], item.expected_url_keywords),
                top_5_hit=_has_url_hit(results[:5], item.expected_url_keywords),
                top_10_hit=_has_url_hit(results[:10], item.expected_url_keywords),
                first_relevant_rank=first_relevant_rank,
                reciprocal_rank=(
                    1.0 / first_relevant_rank
                    if first_relevant_rank is not None
                    else 0.0
                ),
                retrieval_seconds=retrieval_seconds,
                id=item.id,
                query_type=item.query_type,
                topic=item.topic,
            )
        )

    metrics = _ranking_metrics(details)
    query_type_metrics = {
        name: _ranking_metrics([item for item in details if item.query_type == name])
        for name in sorted({item.query_type for item in details if item.query_type})
    }
    topic_metrics = {
        name: _topic_metrics([item for item in details if item.topic == name])
        for name in sorted({item.topic for item in details if item.topic})
    }
    timings = [item.retrieval_seconds for item in details]
    return RetrievalEvaluation(
        top_1_hit_rate=metrics.hit_at_1,
        top_3_hit_rate=metrics.hit_at_3,
        top_5_hit_rate=metrics.hit_at_5,
        top_10_hit_rate=metrics.hit_at_10,
        mrr_at_10=metrics.mrr_at_10,
        question_count=len(details),
        average_retrieval_seconds=sum(timings) / len(timings),
        median_retrieval_seconds=median(timings),
        query_type_metrics=query_type_metrics,
        topic_metrics=topic_metrics,
        unmatched_top_10=tuple(
            item.id or item.question for item in details if not item.top_10_hit
        ),
        questions=tuple(details),
    )


def save_retrieval_evaluation(
    report: RetrievalEvaluation,
    path: Path,
    *,
    environment: Mapping[str, Any] | None = None,
    settings: Mapping[str, Any] | None = None,
) -> None:
    """Save a retrieval report as readable UTF-8 JSON."""
    payload = retrieval_evaluation_to_dict(report)
    if environment is not None:
        payload["environment"] = dict(environment)
    if settings is not None:
        payload["settings"] = dict(settings)
    destination = path.expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def retrieval_evaluation_to_dict(report: RetrievalEvaluation) -> dict[str, Any]:
    """Convert an evaluation report to a stable JSON-compatible structure."""
    return {
        "summary": {
            "question_count": report.question_count,
            "hit_at_1": report.top_1_hit_rate,
            "hit_at_3": report.top_3_hit_rate,
            "hit_at_5": report.top_5_hit_rate,
            "hit_at_10": report.top_10_hit_rate,
            "mrr_at_10": report.mrr_at_10,
            "average_retrieval_seconds": report.average_retrieval_seconds,
            "median_retrieval_seconds": report.median_retrieval_seconds,
        },
        "query_types": {
            name: metrics.to_dict()
            for name, metrics in report.query_type_metrics.items()
        },
        "topics": {
            name: metrics.to_dict() for name, metrics in report.topic_metrics.items()
        },
        "unmatched_top_10": list(report.unmatched_top_10),
        "questions": [
            {
                "id": item.id,
                "question": item.question,
                "query_type": item.query_type,
                "topic": item.topic,
                "expected_url_keywords": list(item.expected_url_keywords),
                "first_relevant_rank": item.first_relevant_rank,
                "reciprocal_rank": item.reciprocal_rank,
                "hit_at_1": item.top_1_hit,
                "hit_at_3": item.top_3_hit,
                "hit_at_5": item.top_5_hit,
                "hit_at_10": item.top_10_hit,
                "retrieval_seconds": item.retrieval_seconds,
                "results": [
                    {
                        "rank": result.rank,
                        "score": result.score,
                        "page_title": result.page_title,
                        "section_title": result.section_title,
                        "source_url": result.source_url,
                    }
                    for result in item.results[:10]
                ],
            }
            for item in report.questions
        ],
    }


def _ranking_metrics(details: Sequence[QuestionEvaluation]) -> RankingMetrics:
    count = len(details)
    if count == 0:
        return RankingMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return RankingMetrics(
        question_count=count,
        hit_at_1=sum(item.top_1_hit for item in details) / count,
        hit_at_3=sum(item.top_3_hit for item in details) / count,
        hit_at_5=sum(item.top_5_hit for item in details) / count,
        hit_at_10=sum(item.top_10_hit for item in details) / count,
        mrr_at_10=sum(item.reciprocal_rank for item in details) / count,
    )


def _topic_metrics(details: Sequence[QuestionEvaluation]) -> TopicMetrics:
    count = len(details)
    if count == 0:
        return TopicMetrics(0, 0.0)
    return TopicMetrics(
        question_count=count,
        hit_at_5=sum(item.top_5_hit for item in details) / count,
    )


def _first_relevant_rank(
    results: Sequence[SearchResult],
    expected_keywords: Sequence[str],
) -> int | None:
    for rank, result in enumerate(results, start=1):
        if any(keyword in result.source_url for keyword in expected_keywords):
            return rank
    return None


def _has_url_hit(
    results: Sequence[SearchResult],
    expected_keywords: Sequence[str],
) -> bool:
    return any(
        keyword in result.source_url
        for result in results
        for keyword in expected_keywords
    )


def _questions_by_stable_id(
    questions: Sequence[QuestionEvaluation],
) -> dict[str, QuestionEvaluation]:
    by_id: dict[str, QuestionEvaluation] = {}
    for item in questions:
        if item.id is None:
            raise ValueError("retrieval comparison requires question IDs")
        if item.id in by_id:
            raise ValueError(f"duplicate retrieval comparison ID: {item.id}")
        by_id[item.id] = item
    return by_id
