import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from python_doc_rag.generation import GenerationConfig
from python_doc_rag.models import CitationSource, CitedAnswer, SearchChunk
from python_doc_rag.rag_evaluation import (
    DERIVED_RESULT_REVISION,
    JUDGE_PROMPT_REVISION,
    JUDGE_SCHEMA_REVISION,
    AnswerabilityQuestion,
    FakeJudge,
    OpenAIResponsesJudge,
    RagDerivedResult,
    RagEvaluationCase,
    RagJudgeResult,
    RagJudgeUsage,
    RagQualityQuestion,
    RecordingRetriever,
    apply_judge_to_records,
    derive_rag_result,
    evaluate_rag_questions,
    judge_result_from_mapping,
    load_answerability_questions,
    load_holdout_questions,
    load_rag_quality_questions,
    load_rag_records,
    save_rag_evaluation,
    summarize_rag_records,
    validate_derived_semantics,
    validate_judge_semantics,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def _chunk(
    url: str = "https://docs.python.org/ja/3.13/library/example.html",
) -> SearchChunk:
    return SearchChunk(
        text="直接答える本文",
        page_title="ページ",
        section_title="節",
        source_url=url,
        category="library",
        chunk_index=0,
        start_index=0,
    )


def _rag_row(**changes: object) -> dict[str, object]:
    row: dict[str, object] = {
        "id": "rag-1",
        "question": "質問",
        "query_type": "conceptual",
        "topic": "topic",
        "answerable": True,
        "expected_answer": "模範回答",
        "required_facts": ["事実"],
        "expected_url_keywords": ["library/example.html"],
    }
    row.update(changes)
    return row


def _answerability_row(**changes: object) -> dict[str, object]:
    row: dict[str, object] = {
        "id": "answerable-1",
        "question": "質問",
        "answerable": True,
        "expected_answer": "模範回答",
        "expected_url_keywords": ["library/example.html"],
    }
    row.update(changes)
    return row


class FixedRetriever:
    def __init__(self, chunks: tuple[SearchChunk, ...]) -> None:
        self.chunks = chunks
        self.calls = 0

    def retrieve(self, question: str, *, limit: int) -> tuple[SearchChunk, ...]:
        del question
        self.calls += 1
        return self.chunks[:limit]


class SequencePipeline:
    def __init__(self, outputs: list[CitedAnswer | Exception]) -> None:
        self.outputs = outputs
        self.calls = 0

    def answer(self, question: str) -> CitedAnswer:
        del question
        output = self.outputs[self.calls]
        self.calls += 1
        if isinstance(output, Exception):
            raise output
        return output


def _quality_question(identifier: str = "q1") -> RagQualityQuestion:
    return RagQualityQuestion(
        id=identifier,
        question=f"質問{identifier}",
        query_type="conceptual",
        topic="topic",
        answerable=True,
        expected_answer="模範回答",
        required_facts=("事実",),
        expected_url_keywords=("library/example.html",),
    )


def _judge_result() -> RagJudgeResult:
    return RagJudgeResult(
        answer_relevance=4,
        faithfulness=4,
        citation_support=4,
        completeness=4,
        concise_reason="資料が回答を支持する",
        unsupported_claims=(),
        missing_required_facts=(),
    )


def _judge_case(
    *,
    answerable: bool = True,
    answer_text: str = "回答[S1]",
    required_facts: tuple[str, ...] = ("事実",),
) -> RagEvaluationCase:
    return RagEvaluationCase(
        id="q1",
        question="質問",
        answerable=answerable,
        expected_answer="模範回答",
        required_facts=required_facts,
        answer_text=answer_text,
        retrieved_chunks=(_chunk(),),
        cited_chunks=(_chunk(),),
    )


def test_holdout_requires_metadata_and_rejects_development_id_overlap(
    tmp_path: Path,
) -> None:
    holdout = tmp_path / "holdout.jsonl"
    development = tmp_path / "development.jsonl"
    _write_jsonl(holdout, [_rag_row(id="shared")])
    _write_jsonl(
        development,
        [
            {
                "id": "shared",
                "question": "既存",
                "query_type": "operational",
                "topic": "old",
                "expected_url_keywords": ["old"],
            }
        ],
    )

    with pytest.raises(ValueError, match="overlap development"):
        load_holdout_questions(holdout, development_path=development)

    _write_jsonl(
        holdout,
        [{"question": "メタデータなし", "expected_url_keywords": ["url"]}],
    )
    with pytest.raises(ValueError, match="require id"):
        load_holdout_questions(holdout)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"query_type": "invalid"}, "unsupported query_type"),
        ({"expected_url_keywords": []}, "expected_url_keywords"),
        ({"required_facts": []}, "required_facts"),
        ({"answerable": False}, "answerable must be true"),
    ],
)
def test_rag_quality_loader_rejects_invalid_fields(
    changes: dict[str, object],
    message: str,
    tmp_path: Path,
) -> None:
    path = tmp_path / "rag.jsonl"
    _write_jsonl(path, [_rag_row(**changes)])

    with pytest.raises(ValueError, match=message):
        load_rag_quality_questions(path)


def test_loaders_reject_duplicate_ids_and_preserve_utf8(tmp_path: Path) -> None:
    path = tmp_path / "rag.jsonl"
    _write_jsonl(path, [_rag_row(), _rag_row(question="別の質問")])
    assert "質問" in path.read_text(encoding="utf-8")
    assert b"\\u8cea" not in path.read_bytes()

    with pytest.raises(ValueError, match="duplicate"):
        load_rag_quality_questions(path)


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (
            [_answerability_row(expected_url_keywords=[])],
            "answerable cases require",
        ),
        (
            [
                _answerability_row(
                    answerable=False,
                    expected_answer=None,
                    reason="対象外",
                    expected_behavior="abstain",
                )
            ],
            "empty expected_url_keywords",
        ),
        (
            [
                _answerability_row(
                    answerable=False,
                    expected_answer=None,
                    expected_url_keywords=[],
                    reason="対象外",
                    expected_behavior="answer",
                )
            ],
            "expected_behavior must be abstain",
        ),
    ],
)
def test_answerability_loader_enforces_conditional_fields(
    rows: list[dict[str, object]],
    message: str,
    tmp_path: Path,
) -> None:
    path = tmp_path / "answerability.jsonl"
    _write_jsonl(path, rows)

    with pytest.raises(ValueError, match=message):
        load_answerability_questions(path)


def test_evaluation_continues_after_case_failure_and_calculates_metrics() -> None:
    chunk = _chunk()
    source = CitationSource("S1", "ページ", "節", chunk.source_url)
    success = CitedAnswer("根拠のある回答[S1]", (source,), (chunk,), 1)
    pipeline = SequencePipeline([RuntimeError("failed"), success])
    trace = RecordingRetriever(FixedRetriever((chunk,)))

    records = evaluate_rag_questions(
        pipeline,
        trace,
        [_quality_question("q1"), _quality_question("q2")],
    )
    summary = summarize_rag_records(records)

    assert pipeline.calls == 2
    assert records[0].fail_closed is True
    assert records[1].generation_succeeded is True
    assert summary["generation_success_rate"] == pytest.approx(0.5)
    assert summary["citation_format_success_rate"] == pytest.approx(0.5)
    assert summary["expected_source_top_5_rate"] == pytest.approx(0.5)
    assert summary["expected_source_cited_rate"] == pytest.approx(0.5)
    assert summary["irrelevant_source_only_rate"] == 0.0


def test_answerability_metrics_cover_false_answer_and_false_abstention() -> None:
    chunk = _chunk()
    source = CitationSource("S1", "ページ", "節", chunk.source_url)
    answer = CitedAnswer("回答[S1]", (source,), (chunk,), 1)
    empty = CitedAnswer(
        "関連するPython 3.13日本語公式ドキュメントが見つかりませんでした。",
        (),
        (),
        0,
    )
    answerable = AnswerabilityQuestion(
        "a1",
        "回答可能",
        True,
        ("library/example.html",),
        expected_answer="模範",
    )
    unanswerable = AnswerabilityQuestion(
        "u1",
        "回答不能",
        False,
        (),
        reason="対象外",
        expected_behavior="abstain",
    )
    trace = RecordingRetriever(FixedRetriever((chunk,)))

    false_records = evaluate_rag_questions(
        SequencePipeline([empty, answer]),
        trace,
        [answerable, unanswerable],
    )
    false_summary = summarize_rag_records(false_records)
    assert false_summary["false_abstain_rate"] == 1.0
    assert false_summary["false_answer_rate"] == 1.0
    assert false_summary["unanswerable_source_display_rate"] == 1.0

    correct_records = evaluate_rag_questions(
        SequencePipeline([answer, empty]),
        trace,
        [answerable, unanswerable],
    )
    correct_summary = summarize_rag_records(correct_records)
    assert correct_summary["answerable_answer_success_rate"] == 1.0
    assert correct_summary["unanswerable_abstention_rate"] == 1.0


def test_empty_metrics_are_zero() -> None:
    summary = summarize_rag_records([])

    assert summary["question_count"] == 0
    assert summary["citation_format_success_rate"] == 0.0
    assert summary["false_answer_rate"] == 0.0
    assert summary["false_abstain_rate"] == 0.0


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"faithfulness": 5}, "faithfulness"),
        ({"unsupported_claims": "not-an-array"}, "string array"),
        ({"missing_required_facts": [""]}, "non-empty strings"),
    ],
)
def test_judge_result_rejects_invalid_raw_structure(
    change: dict[str, object],
    message: str,
) -> None:
    data = _judge_result().to_dict()
    data.update(change)

    with pytest.raises(ValueError, match=message):
        judge_result_from_mapping(data)


def _changed_judge_result(**changes: object) -> RagJudgeResult:
    data = _judge_result().to_dict()
    data.update(changes)
    return judge_result_from_mapping(data)


def test_complete_answer_is_derived_locally() -> None:
    result = _judge_result()
    case = _judge_case()

    derived = derive_rag_result(result, case, abstained=False)

    assert derived == RagDerivedResult(
        revision=DERIVED_RESULT_REVISION,
        grounded=True,
        coverage_label="complete",
        answerability_label="supported_answer",
    )
    assert validate_derived_semantics(derived, result, case, abstained=False) == ()


def test_supported_partial_answer_stays_grounded_and_supported() -> None:
    result = _changed_judge_result(
        faithfulness=4,
        citation_support=4,
        completeness=2,
        missing_required_facts=["事実2"],
    )
    case = _judge_case(required_facts=("事実1", "事実2"))

    derived = derive_rag_result(result, case, abstained=False)

    assert derived.grounded is True
    assert derived.coverage_label == "partial"
    assert derived.answerability_label == "supported_answer"
    assert validate_derived_semantics(derived, result, case, abstained=False) == ()


def test_unsupported_answer_is_false_answer() -> None:
    result = _changed_judge_result(
        faithfulness=0,
        citation_support=0,
        completeness=0,
        unsupported_claims=["未支持主張"],
        missing_required_facts=["事実"],
    )

    derived = derive_rag_result(result, _judge_case(), abstained=False)

    assert derived.grounded is False
    assert derived.coverage_label == "insufficient"
    assert derived.answerability_label == "false_answer"


def test_complete_but_unsupported_answer_is_valid() -> None:
    result = _changed_judge_result(
        answer_relevance=1,
        faithfulness=0,
        citation_support=0,
        completeness=4,
        unsupported_claims=["明確な誤回答"],
        missing_required_facts=[],
    )
    case = _judge_case()

    derived = derive_rag_result(result, case, abstained=False)

    assert derived.grounded is False
    assert derived.coverage_label == "complete"
    assert derived.answerability_label == "false_answer"
    assert validate_derived_semantics(derived, result, case, abstained=False) == ()


def test_insufficient_but_supported_answer_is_valid() -> None:
    result = _changed_judge_result(
        faithfulness=4,
        citation_support=4,
        completeness=1,
        unsupported_claims=[],
        missing_required_facts=["事実2"],
    )
    case = _judge_case(required_facts=("事実1", "事実2"))

    derived = derive_rag_result(result, case, abstained=False)

    assert derived.grounded is True
    assert derived.coverage_label == "insufficient"
    assert derived.answerability_label == "supported_answer"
    assert validate_derived_semantics(derived, result, case, abstained=False) == ()


@pytest.mark.parametrize(
    ("answerable", "abstained", "grounded", "expected"),
    [
        (True, False, True, "supported_answer"),
        (True, False, False, "false_answer"),
        (True, True, True, "false_abstention"),
        (False, True, True, "correct_abstention"),
        (False, False, True, "false_answer"),
    ],
)
def test_answerability_label_is_derived_from_local_inputs(
    answerable: bool,
    abstained: bool,
    grounded: bool,
    expected: str,
) -> None:
    result = _judge_result() if grounded else _changed_judge_result(faithfulness=0)
    answer_text = GenerationConfig().empty_result_message if abstained else "回答[S1]"
    case = _judge_case(answerable=answerable, answer_text=answer_text)

    derived = derive_rag_result(result, case, abstained=abstained)

    assert derived.answerability_label == expected


@pytest.mark.parametrize(
    ("completeness", "missing", "required", "expected"),
    [
        (4, [], (), "complete"),
        (2, [], (), "partial"),
        (0, [], (), "insufficient"),
        (2, ["事実2"], ("事実1", "事実2"), "partial"),
        (2, ["事実1", "事実2"], ("事実1", "事実2"), "insufficient"),
    ],
)
def test_coverage_boundaries_are_deterministic(
    completeness: int,
    missing: list[str],
    required: tuple[str, ...],
    expected: str,
) -> None:
    result = _changed_judge_result(
        completeness=completeness,
        missing_required_facts=missing,
    )

    derived = derive_rag_result(
        result,
        _judge_case(required_facts=required),
        abstained=False,
    )

    assert derived.coverage_label == expected


def test_completeness_and_missing_fact_contradiction_is_derived_error() -> None:
    result = _changed_judge_result(
        completeness=4,
        missing_required_facts=["事実"],
    )
    case = _judge_case()
    derived = derive_rag_result(result, case, abstained=False)

    issues = validate_derived_semantics(derived, result, case, abstained=False)

    assert "insufficient coverage conflicts with completeness=4" in issues
    assert "completeness=4 conflicts with missing required facts" in issues


@pytest.mark.parametrize("answerable", [True, False])
def test_abstention_requires_the_exact_non_answer_body(answerable: bool) -> None:
    result = _judge_result()
    valid_case = _judge_case(
        answerable=answerable,
        answer_text=GenerationConfig().empty_result_message,
    )
    valid = derive_rag_result(result, valid_case, abstained=True)
    assert validate_derived_semantics(valid, result, valid_case, abstained=True) == ()

    invalid_case = _judge_case(answerable=answerable, answer_text="実質的な回答")
    invalid = derive_rag_result(result, invalid_case, abstained=True)
    assert "abstention label conflicts with a substantive answer body" in (
        validate_derived_semantics(invalid, result, invalid_case, abstained=True)
    )


def test_raw_revision_mismatch_is_rejected() -> None:
    issues = validate_judge_semantics(
        _judge_result(),
        _judge_case(),
        prompt_revision="rag-grounding-v2",
        schema_revision="rag-judge-result-v2",
    )

    assert f"prompt revision must be {JUDGE_PROMPT_REVISION}" in issues
    assert f"schema revision must be {JUDGE_SCHEMA_REVISION}" in issues


def test_token_usage_requires_consistent_total() -> None:
    with pytest.raises(ValueError, match="total_tokens"):
        RagJudgeUsage(
            input_tokens=10,
            output_tokens=5,
            total_tokens=16,
            requested_model="model",
            actual_response_model="model",
        )


def test_fake_judge_and_saved_result_rejudging(tmp_path: Path) -> None:
    chunk = _chunk()
    source = CitationSource("S1", "ページ", "節", chunk.source_url)
    records = evaluate_rag_questions(
        SequencePipeline([CitedAnswer("回答[S1]", (source,), (chunk,), 1)]),
        RecordingRetriever(FixedRetriever((chunk,))),
        [_quality_question()],
    )
    judge = FakeJudge(_judge_result())
    output = tmp_path / "result.json"

    judged = apply_judge_to_records(records, judge)
    save_rag_evaluation(judged, output)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert len(judge.calls) == 1
    assert payload["summary"]["grounded_rate"] == 1.0
    assert "grounded" not in payload["questions"][0]["judge_result"]
    assert payload["questions"][0]["derived_result"] == {
        "revision": DERIVED_RESULT_REVISION,
        "grounded": True,
        "coverage_label": "complete",
        "answerability_label": "supported_answer",
    }
    assert payload["questions"][0]["judge_prompt_revision"] == JUDGE_PROMPT_REVISION
    assert payload["questions"][0]["judge_schema_revision"] == JUDGE_SCHEMA_REVISION
    assert "OPENAI_API_KEY" not in output.read_text(encoding="utf-8")


class FakeResponses:
    def __init__(self, output: dict[str, object] | Exception) -> None:
        self.output = output
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        if isinstance(self.output, Exception):
            raise self.output
        return SimpleNamespace(
            output_text=json.dumps(self.output),
            model="actual-test-model",
            usage=SimpleNamespace(
                input_tokens=101,
                output_tokens=17,
                total_tokens=118,
            ),
        )


def test_openai_judge_uses_v3_raw_schema_and_store_false() -> None:
    responses = FakeResponses(_judge_result().to_dict())
    judge = OpenAIResponsesJudge(
        model_name="test-model",
        client=SimpleNamespace(responses=responses),
    )

    result = judge.evaluate(_judge_case())

    assert result.faithfulness == 4
    assert responses.calls[0]["store"] is False
    assert responses.calls[0]["model"] == "test-model"
    assert responses.calls[0]["text"]["format"]["strict"] is True
    assert "tools" not in responses.calls[0]
    system_prompt = responses.calls[0]["input"][0]["content"]
    schema = responses.calls[0]["text"]["format"]
    assert JUDGE_PROMPT_REVISION == "rag-grounding-v3"
    assert JUDGE_SCHEMA_REVISION == "rag-judge-evidence-v3"
    assert DERIVED_RESULT_REVISION == "rag-derived-evaluation-v1"
    assert "0が最低品質、4が最高品質" in system_prompt
    assert "grounded、coverage、answerability labelは" in system_prompt
    for score in range(5):
        assert f"- {score}:" in system_prompt
    assert schema["name"] == "rag_judge_evidence_v3"
    properties = schema["schema"]["properties"]
    assert set(properties) == {
        "answer_relevance",
        "faithfulness",
        "citation_support",
        "completeness",
        "concise_reason",
        "unsupported_claims",
        "missing_required_facts",
    }
    assert "grounded" not in properties
    assert "answerability_label" not in properties
    assert judge.last_usage is not None
    assert judge.last_usage.input_tokens == 101
    assert judge.last_usage.output_tokens == 17
    assert judge.last_usage.total_tokens == 118
    assert judge.last_usage.actual_response_model == "actual-test-model"


def test_judge_failure_continues_with_the_next_question() -> None:
    chunk = _chunk()
    source = CitationSource("S1", "ページ", "節", chunk.source_url)
    answer = CitedAnswer("回答[S1]", (source,), (chunk,), 1)

    def result_or_error(case: RagEvaluationCase) -> RagJudgeResult:
        if case.id == "q1":
            raise RuntimeError("judge down")
        return _judge_result()

    judge = FakeJudge(result_or_error)
    records = evaluate_rag_questions(
        SequencePipeline([answer, answer]),
        RecordingRetriever(FixedRetriever((chunk,))),
        [_quality_question("q1"), _quality_question("q2")],
        judge=judge,
    )
    summary = summarize_rag_records(records)

    assert len(judge.calls) == 2
    assert records[0].judge_error == "RuntimeError: judge down"
    assert records[1].derived_result is not None
    assert summary["judge_completed_count"] == 1
    assert summary["raw_judge_error_count"] == 1
    assert summary["derived_semantic_error_count"] == 0


def test_raw_success_derived_failure_is_excluded_and_processing_continues() -> None:
    chunk = _chunk()
    source = CitationSource("S1", "ページ", "節", chunk.source_url)
    answer = CitedAnswer("回答[S1]", (source,), (chunk,), 1)
    contradiction = _changed_judge_result(
        completeness=4,
        missing_required_facts=["事実"],
    )
    judge = FakeJudge(
        lambda case: contradiction if case.id == "q1" else _judge_result()
    )

    records = evaluate_rag_questions(
        SequencePipeline([answer, answer]),
        RecordingRetriever(FixedRetriever((chunk,))),
        [_quality_question("q1"), _quality_question("q2")],
        judge=judge,
    )
    summary = summarize_rag_records(records)

    assert len(judge.calls) == 2
    assert records[0].judge_result == contradiction
    assert records[0].derived_result is None
    assert records[0].judge_error is None
    assert records[0].derived_error is not None
    assert records[1].derived_result is not None
    assert summary["judge_completed_count"] == 1
    assert summary["raw_judge_error_count"] == 0
    assert summary["derived_semantic_error_count"] == 1
    assert summary["average_completeness"] == 4.0


def test_openai_dependency_is_imported_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fail_import(name: str) -> object:
        calls.append(name)
        raise ImportError(name)

    monkeypatch.setattr(
        "python_doc_rag.rag_evaluation.importlib.import_module",
        fail_import,
    )

    assert calls == []
    with pytest.raises(RuntimeError, match="evaluation-openai"):
        OpenAIResponsesJudge(model_name="test-model")
    assert calls == ["openai"]


def test_v2_input_is_not_overwritten_when_v3_result_is_saved(tmp_path: Path) -> None:
    chunk = _chunk()
    source = CitationSource("S1", "ページ", "節", chunk.source_url)
    source_records = evaluate_rag_questions(
        SequencePipeline([CitedAnswer("回答[S1]", (source,), (chunk,), 1)]),
        RecordingRetriever(FixedRetriever((chunk,))),
        [_quality_question()],
    )
    v2_input = tmp_path / "rag_dense_openai_v2.json"
    v3_output = tmp_path / "rag_dense_openai_v3.json"
    save_rag_evaluation(source_records, v2_input)
    before = v2_input.read_bytes()

    loaded = load_rag_records(v2_input)
    judged = apply_judge_to_records(loaded, FakeJudge(_judge_result()))
    save_rag_evaluation(judged, v3_output)

    assert v2_input.read_bytes() == before
    assert v3_output != v2_input
    assert v3_output.exists()


def test_token_usage_and_models_are_saved_without_api_key(tmp_path: Path) -> None:
    chunk = _chunk()
    source = CitationSource("S1", "ページ", "節", chunk.source_url)
    judge = OpenAIResponsesJudge(
        model_name="requested-test-model",
        client=SimpleNamespace(responses=FakeResponses(_judge_result().to_dict())),
    )
    records = evaluate_rag_questions(
        SequencePipeline([CitedAnswer("回答[S1]", (source,), (chunk,), 1)]),
        RecordingRetriever(FixedRetriever((chunk,))),
        [_quality_question()],
        judge=judge,
    )
    output = tmp_path / "result.json"

    save_rag_evaluation(records, output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    usage = payload["questions"][0]["judge_usage"]

    assert usage == {
        "input_tokens": 101,
        "output_tokens": 17,
        "total_tokens": 118,
        "requested_model": "requested-test-model",
        "actual_response_model": "actual-test-model",
    }
    assert payload["summary"]["judge_input_tokens"] == 101
    assert payload["summary"]["judge_output_tokens"] == 17
    assert payload["summary"]["judge_total_tokens"] == 118
    serialized = output.read_text(encoding="utf-8")
    assert "OPENAI_API_KEY" not in serialized
    assert "sk-" not in serialized


def test_explicit_abstention_records_contract_fields_without_sources() -> None:
    from python_doc_rag.models import AbstainedAnswer

    chunk = _chunk()
    outcome = AbstainedAnswer("insufficient_evidence", (chunk,), 1)
    trace = RecordingRetriever(FixedRetriever((chunk,)))
    records = evaluate_rag_questions(
        SequencePipeline([outcome]),
        trace,
        [
            AnswerabilityQuestion(
                "u1",
                "回答不能",
                False,
                (),
                reason="対象外",
                expected_behavior="abstain",
            )
        ],
    )
    record = records[0]
    payload = record.to_dict()
    summary = summarize_rag_records(records)
    assert payload["status"] == "abstain"
    assert payload["reason_code"] == "insufficient_evidence"
    assert payload["answer_text"] == ""
    assert payload["citation_numbers"] == []
    assert payload["displayed_source_urls"] == []
    assert payload["first_attempt_contract_success"] is True
    assert payload["retry_used"] is False
    assert payload["contract_failed"] is False
    assert summary["unanswerable_abstention_rate"] == 1.0
    assert summary["abstain_source_display_count"] == 0
    assert summary["abstain_answer_text_count"] == 0


def test_explicit_abstention_is_false_abstention_for_answerable_question() -> None:
    from python_doc_rag.models import AbstainedAnswer

    chunk = _chunk()
    records = evaluate_rag_questions(
        SequencePipeline([AbstainedAnswer("insufficient_evidence", (chunk,), 2)]),
        RecordingRetriever(FixedRetriever((chunk,))),
        [
            AnswerabilityQuestion(
                "a1",
                "回答可能",
                True,
                ("library/example.html",),
                expected_answer="模範",
            )
        ],
    )
    summary = summarize_rag_records(records)
    assert summary["false_abstain_rate"] == 1.0
    assert summary["retry_used_count"] == 1
    assert summary["retry_used_rate"] == 1.0
