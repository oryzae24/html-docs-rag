import json
from pathlib import Path

import pytest

from python_doc_rag.cli import AnswerExecution
from python_doc_rag.models import (
    AbstainedAnswer,
    CitationSource,
    CitedAnswer,
    SearchChunk,
)
from python_doc_rag.portability_evaluation import (
    PortabilityQuestion,
    evaluate_portability_questions,
    load_portability_questions,
    save_portability_evaluation_atomic,
    summarize_portability,
)


def _chunk(url: str = "https://docs.example.test/guide/#x") -> SearchChunk:
    return SearchChunk(
        text="fixture evidence",
        page_title="Fixture",
        section_title="Section",
        source_url=url,
        category="fixture",
        chunk_index=0,
        start_index=0,
    )


def _question(*, answerable: bool = True) -> PortabilityQuestion:
    return PortabilityQuestion(
        id="fixture",
        question="question",
        answerable=answerable,
        query_type="operational",
        topic="fixture",
        expected_answer="answer" if answerable else None,
        required_facts=("fact",) if answerable else (),
        expected_url_keywords=("/guide/",) if answerable else (),
        reason=None if answerable else "outside scope",
    )


class FakeService:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = outcomes
        self.calls: list[str] = []

    def answer(self, question: str) -> AnswerExecution:
        self.calls.append(question)
        outcome = self.outcomes[len(self.calls) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return AnswerExecution(
            outcome,  # type: ignore[arg-type]
            0.1,
            0.2,
            0.3,
            input_tokens=100,
            generated_tokens=10,
            generation_calls=1,
        )


def test_portability_evaluation_records_contract_citation_and_boundary() -> None:
    chunk = _chunk()
    answer = CitedAnswer(
        answer_text="grounded answer [S1]",
        sources=(
            CitationSource(
                label="S1",
                page_title="Fixture",
                section_title="Section",
                url=chunk.source_url,
            ),
        ),
        retrieved_chunks=(chunk,),
        generation_attempts=1,
    )

    records = evaluate_portability_questions(
        FakeService([answer]),
        [_question()],
        allowed_source_prefixes=("https://docs.example.test/guide/",),
    )

    assert records[0].correct_source_top5
    assert records[0].correct_source_cited
    assert records[0].citation_format_valid
    assert records[0].source_boundary_valid
    assert not records[0].answer_contains_url
    assert records[0].input_tokens == 100


def test_portability_evaluation_abstain_has_no_leakage() -> None:
    abstain = AbstainedAnswer(
        reason_code="insufficient_evidence",
        retrieved_chunks=(_chunk(),),
        generation_attempts=1,
    )

    records = evaluate_portability_questions(
        FakeService([abstain]),
        [_question(answerable=False)],
        allowed_source_prefixes=("https://docs.example.test/guide/",),
    )

    record = records[0]
    assert record.abstained
    assert not record.false_answer
    assert not record.abstain_text_leakage
    assert not record.abstain_source_leakage


def test_portability_evaluation_continues_and_saves_each_case(
    tmp_path: Path,
) -> None:
    service = FakeService(
        [
            RuntimeError("case failed"),
            AbstainedAnswer(
                reason_code="insufficient_evidence",
                retrieved_chunks=(),
                generation_attempts=1,
            ),
        ]
    )
    questions = [_question(), _question(answerable=False)]
    partial_counts: list[int] = []

    records = evaluate_portability_questions(
        service,
        questions,
        allowed_source_prefixes=("https://docs.example.test/guide/",),
        on_case=lambda values: partial_counts.append(len(values)),
    )
    output = tmp_path / "result.json"
    save_portability_evaluation_atomic(
        records,
        output,
        settings={"openai_api_used": False},
        environment={},
    )

    assert partial_counts == [1, 2]
    assert records[0].contract_failure
    assert records[1].generation_succeeded
    assert json.loads(output.read_text())["summary"]["question_count"] == 2
    assert not list(tmp_path.glob(".result.json.*.tmp"))


def test_portability_summary_counts_false_answer_and_abstention() -> None:
    answered = CitedAnswer(
        answer_text="unsupported [S1]",
        sources=(CitationSource("S1", "Page", "Section", _chunk().source_url),),
        retrieved_chunks=(_chunk(),),
        generation_attempts=1,
    )
    abstained = AbstainedAnswer(
        reason_code="insufficient_evidence",
        retrieved_chunks=(_chunk(),),
        generation_attempts=1,
    )
    records = evaluate_portability_questions(
        FakeService([answered, abstained]),
        [_question(answerable=False), _question(answerable=True)],
        allowed_source_prefixes=("https://docs.example.test/guide/",),
    )

    summary = summarize_portability(records)

    assert summary["false_answer_count"] == 1
    assert summary["false_abstention_count"] == 1


def test_portability_question_loader_rejects_duplicate_and_conflict(
    tmp_path: Path,
) -> None:
    row = {
        "id": "duplicate",
        "question": "question",
        "answerable": False,
        "query_type": "conceptual",
        "topic": "scope",
        "expected_url_keywords": [],
        "required_facts": [],
        "reason": "outside scope",
    }
    path = tmp_path / "questions.jsonl"
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")

    with pytest.raises(ValueError, match="duplicate"):
        load_portability_questions(path)

    row["expected_answer"] = "conflict"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="conflict"):
        load_portability_questions(path)
