"""Offline portability-smoke records for one reusable local RAG service."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from python_doc_rag.models import AbstainedAnswer, CitedAnswer

_QUERY_TYPES = frozenset({"exact_identifier", "conceptual", "operational"})
_URL_PATTERN = re.compile(r"(?i)(?:https?://|www\.)\S+")


@dataclass(frozen=True, slots=True)
class PortabilityQuestion:
    """One bounded-corpus smoke question and its source-grounded expectation."""

    id: str
    question: str
    answerable: bool
    query_type: str
    topic: str
    expected_answer: str | None
    required_facts: tuple[str, ...]
    expected_url_keywords: tuple[str, ...]
    reason: str | None


@dataclass(frozen=True, slots=True)
class PortabilityCaseRecord:
    """Deterministic contract, citation, boundary, and timing observations."""

    id: str
    question: str
    answerable: bool
    query_type: str
    generation_succeeded: bool
    contract_failure: bool
    abstained: bool
    false_answer: bool
    false_abstention: bool
    answer_text: str
    displayed_source_urls: tuple[str, ...]
    retrieved_source_urls: tuple[str, ...]
    correct_source_top5: bool
    correct_source_cited: bool
    citation_format_valid: bool
    answer_contains_url: bool
    abstain_text_leakage: bool
    abstain_source_leakage: bool
    source_boundary_valid: bool
    retrieval_seconds: float
    generation_seconds: float
    total_seconds: float
    input_tokens: int
    generated_tokens: int
    generation_calls: int
    error: str | None


class PortabilityService(Protocol):
    """Small reusable answer service boundary implemented by the CLI runtime."""

    def answer(self, question: str) -> Any:
        """Return one independent answer execution."""
        ...


def load_portability_questions(path: Path) -> list[PortabilityQuestion]:
    """Load strict unique smoke rows without accepting contradictory fields."""
    rows: list[PortabilityQuestion] = []
    seen: set[str] = set()
    with path.expanduser().open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"question line {line_number} must be an object")
            question_id = _string(value, "id", line_number)
            if question_id in seen:
                raise ValueError(f"duplicate portability question id: {question_id}")
            seen.add(question_id)
            answerable = value.get("answerable")
            if not isinstance(answerable, bool):
                raise ValueError(f"line {line_number}: answerable must be boolean")
            query_type = _string(value, "query_type", line_number)
            if query_type not in _QUERY_TYPES:
                raise ValueError(f"line {line_number}: invalid query_type")
            expected_urls = _string_tuple(
                value.get("expected_url_keywords", []),
                "expected_url_keywords",
                line_number,
            )
            expected_answer = value.get("expected_answer")
            required_facts = _string_tuple(
                value.get("required_facts", []),
                "required_facts",
                line_number,
            )
            reason = value.get("reason")
            if answerable:
                if not isinstance(expected_answer, str) or not expected_answer.strip():
                    raise ValueError(f"line {line_number}: expected_answer is required")
                if not expected_urls or not required_facts or reason is not None:
                    raise ValueError(f"line {line_number}: invalid answerable fields")
            else:
                if expected_answer is not None or required_facts or expected_urls:
                    raise ValueError(
                        f"line {line_number}: unanswerable fields conflict"
                    )
                if not isinstance(reason, str) or not reason.strip():
                    raise ValueError(f"line {line_number}: reason is required")
            rows.append(
                PortabilityQuestion(
                    id=question_id,
                    question=_string(value, "question", line_number),
                    answerable=answerable,
                    query_type=query_type,
                    topic=_string(value, "topic", line_number),
                    expected_answer=(
                        expected_answer.strip()
                        if isinstance(expected_answer, str)
                        else None
                    ),
                    required_facts=required_facts,
                    expected_url_keywords=expected_urls,
                    reason=reason.strip() if isinstance(reason, str) else None,
                )
            )
    if not rows:
        raise ValueError("portability question set must not be empty")
    return rows


def evaluate_portability_questions(
    service: PortabilityService,
    questions: Sequence[PortabilityQuestion],
    *,
    allowed_source_prefixes: Sequence[str],
    on_case: Callable[[Sequence[PortabilityCaseRecord]], None] | None = None,
) -> list[PortabilityCaseRecord]:
    """Evaluate all cases independently and continue after one local failure."""
    prefixes = tuple(allowed_source_prefixes)
    if not prefixes:
        raise ValueError("allowed_source_prefixes must not be empty")
    records: list[PortabilityCaseRecord] = []
    for question in questions:
        try:
            execution = service.answer(question.question)
            answer = execution.answer
            abstained = isinstance(answer, AbstainedAnswer)
            answer_text = answer.answer_text if isinstance(answer, CitedAnswer) else ""
            displayed_urls = (
                tuple(source.url for source in answer.sources)
                if isinstance(answer, CitedAnswer)
                else ()
            )
            retrieved_urls = tuple(
                chunk.source_url for chunk in answer.retrieved_chunks[:5]
            )
            record = PortabilityCaseRecord(
                id=question.id,
                question=question.question,
                answerable=question.answerable,
                query_type=question.query_type,
                generation_succeeded=True,
                contract_failure=False,
                abstained=abstained,
                false_answer=not question.answerable and not abstained,
                false_abstention=question.answerable and abstained,
                answer_text=answer_text,
                displayed_source_urls=displayed_urls,
                retrieved_source_urls=retrieved_urls,
                correct_source_top5=_matches_expected(
                    retrieved_urls, question.expected_url_keywords
                ),
                correct_source_cited=_matches_expected(
                    displayed_urls, question.expected_url_keywords
                ),
                citation_format_valid=(
                    abstained
                    or bool(displayed_urls)
                    and all(
                        f"[{source.label}]" in answer_text for source in answer.sources
                    )
                ),
                answer_contains_url=bool(_URL_PATTERN.search(answer_text)),
                abstain_text_leakage=abstained and bool(answer_text),
                abstain_source_leakage=abstained and bool(displayed_urls),
                source_boundary_valid=all(
                    any(url.startswith(prefix) for prefix in prefixes)
                    for url in displayed_urls
                ),
                retrieval_seconds=float(execution.retrieval_seconds),
                generation_seconds=float(execution.generation_seconds),
                total_seconds=float(execution.total_seconds),
                input_tokens=int(getattr(execution, "input_tokens", 0)),
                generated_tokens=int(getattr(execution, "generated_tokens", 0)),
                generation_calls=int(getattr(execution, "generation_calls", 0)),
                error=None,
            )
        except Exception as error:  # noqa: BLE001 - one case must not stop smoke
            record = PortabilityCaseRecord(
                id=question.id,
                question=question.question,
                answerable=question.answerable,
                query_type=question.query_type,
                generation_succeeded=False,
                contract_failure=True,
                abstained=False,
                false_answer=False,
                false_abstention=False,
                answer_text="",
                displayed_source_urls=(),
                retrieved_source_urls=(),
                correct_source_top5=False,
                correct_source_cited=False,
                citation_format_valid=False,
                answer_contains_url=False,
                abstain_text_leakage=False,
                abstain_source_leakage=False,
                source_boundary_valid=True,
                retrieval_seconds=0.0,
                generation_seconds=0.0,
                total_seconds=0.0,
                input_tokens=0,
                generated_tokens=0,
                generation_calls=0,
                error=f"{type(error).__name__}: {error}",
            )
        records.append(record)
        if on_case is not None:
            on_case(tuple(records))
    return records


def summarize_portability(records: Sequence[PortabilityCaseRecord]) -> dict[str, Any]:
    """Return transparent aggregate contract and citation counts."""
    answerable = [record for record in records if record.answerable]
    unanswerable = [record for record in records if not record.answerable]
    completed = [record for record in records if record.generation_succeeded]
    return {
        "question_count": len(records),
        "answerable_count": len(answerable),
        "unanswerable_count": len(unanswerable),
        "valid_answer_count": sum(not r.abstained for r in completed),
        "valid_abstain_count": sum(r.abstained for r in completed),
        "false_answer_count": sum(r.false_answer for r in records),
        "false_abstention_count": sum(r.false_abstention for r in records),
        "contract_failure_count": sum(r.contract_failure for r in records),
        "correct_source_top5_count": sum(r.correct_source_top5 for r in answerable),
        "correct_source_cited_count": sum(r.correct_source_cited for r in answerable),
        "citation_format_failure_count": sum(
            not r.citation_format_valid for r in completed
        ),
        "answer_url_count": sum(r.answer_contains_url for r in records),
        "abstain_text_leakage_count": sum(r.abstain_text_leakage for r in records),
        "abstain_source_leakage_count": sum(r.abstain_source_leakage for r in records),
        "source_boundary_failure_count": sum(
            not r.source_boundary_valid for r in records
        ),
        "average_retrieval_seconds": _average(r.retrieval_seconds for r in completed),
        "average_generation_seconds": _average(r.generation_seconds for r in completed),
        "average_total_seconds": _average(r.total_seconds for r in completed),
        "average_input_tokens": _average(float(r.input_tokens) for r in completed),
        "generation_call_count": sum(r.generation_calls for r in records),
    }


def save_portability_evaluation_atomic(
    records: Sequence[PortabilityCaseRecord],
    path: Path,
    *,
    settings: dict[str, Any],
    environment: dict[str, Any],
) -> None:
    """Atomically persist partial or complete smoke results."""
    payload = {
        "summary": summarize_portability(records),
        "settings": settings,
        "environment": environment,
        "questions": [asdict(record) for record in records],
    }
    destination = path.expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
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
        os.replace(name, destination)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise


def _matches_expected(urls: Sequence[str], keywords: Sequence[str]) -> bool:
    return bool(keywords) and any(
        keyword in url for keyword in keywords for url in urls
    )


def _string(value: dict[str, Any], key: str, line_number: int) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise ValueError(f"line {line_number}: {key} must be a non-empty string")
    return item.strip()


def _string_tuple(value: object, label: str, line_number: int) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError(f"line {line_number}: {label} must be a string list")
    return tuple(item.strip() for item in value)


def _average(values: Any) -> float:
    collected = list(values)
    return sum(collected) / len(collected) if collected else 0.0
