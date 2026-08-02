import pytest

from python_doc_rag.answer_contract import (
    AnswerContractError,
    AnswerOrAbstainGenerationContract,
    generation_contract_for_mode,
    parse_answer_contract,
)
from python_doc_rag.models import AbstainedAnswer, CitedAnswer, SearchChunk
from python_doc_rag.profiles import runtime_profile


def _chunk(number: int = 1) -> SearchChunk:
    return SearchChunk(
        text="根拠本文",
        page_title="ページ",
        section_title="節",
        source_url=f"https://trusted.invalid/{number}",
        category="library",
        chunk_index=number,
        start_index=0,
    )


def _finalize(raw: str, chunks: tuple[SearchChunk, ...] | None = None):
    return AnswerOrAbstainGenerationContract().finalize(
        raw,
        chunks or (_chunk(),),
        generation_attempts=1,
    )


@pytest.mark.parametrize(
    ("raw", "status"),
    [
        ('{"status":"answer","answer_text":"回答[S1]","reason":null}', "answer"),
        (
            '{"status":"abstain","answer_text":"",'
            '"reason":"insufficient_evidence"}',
            "abstain",
        ),
        (
            ' \n {"status":"answer","answer_text":"回答[S1]","reason":null}\t ',
            "answer",
        ),
    ],
)
def test_parser_accepts_valid_contract(raw: str, status: str) -> None:
    assert parse_answer_contract(raw).status == status


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        ("not-json", "invalid_json"),
        (
            '```json\n{"status":"answer","answer_text":"回答[S1]","reason":null}\n```',
            "markdown_code_fence",
        ),
        (
            'before {"status":"answer","answer_text":"回答[S1]","reason":null}',
            "invalid_json",
        ),
        (
            '{"status":"answer","answer_text":"回答[S1]","reason":null} after',
            "invalid_json",
        ),
        (
            '{"status":"answer","status":"abstain","answer_text":"",'
            '"reason":"insufficient_evidence"}',
            "duplicate_key",
        ),
        ('{"status":"answer","answer_text":"回答[S1]"}', "missing_key"),
        (
            '{"status":"answer","answer_text":"回答[S1]","reason":null,"x":1}',
            "extra_key",
        ),
        ('{"status":1,"answer_text":"回答[S1]","reason":null}', "status_wrong_type"),
        ('{"status":"maybe","answer_text":"回答[S1]","reason":null}', "unknown_status"),
        (
            '{"status":"abstain","answer_text":"","reason":"other"}',
            "unknown_reason",
        ),
        ('{"status":"answer","answer_text":"回答[S1]","reason":NaN}',
         "non_standard_json_value"),
        ('["answer", "回答[S1]"]', "top_level_not_object"),
    ],
)
def test_parser_rejects_invalid_contract(raw: str, reason: str) -> None:
    with pytest.raises(AnswerContractError) as error:
        parse_answer_contract(raw)
    assert reason in error.value.reasons


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        ('{"status":"answer","answer_text":"","reason":null}', "empty_answer"),
        (
            '{"status":"answer","answer_text":"回答[S1]",'
            '"reason":"insufficient_evidence"}',
            "answer_reason_must_be_null",
        ),
        (
            '{"status":"answer","answer_text":"引用なし","reason":null}',
            "citation_required",
        ),
        (
            '{"status":"answer","answer_text":"回答[s1]","reason":null}',
            "malformed_citation",
        ),
        (
            '{"status":"answer","answer_text":"回答[S2]","reason":null}',
            "citation_out_of_range",
        ),
        (
            '{"status":"answer","answer_text":"https://evil.invalid [S1]",'
            '"reason":null}',
            "url_detected",
        ),
        (
            '{"status":"answer","answer_text":"[資料](path) 回答[S1]",'
            '"reason":null}',
            "markdown_link_detected",
        ),
    ],
)
def test_answer_branch_reuses_citation_contract(raw: str, reason: str) -> None:
    with pytest.raises(AnswerContractError) as error:
        _finalize(raw)
    assert reason in error.value.reasons


def test_answer_sources_are_metadata_derived_and_preserve_selected_tuple() -> None:
    chunks = (_chunk(),)
    outcome = _finalize(
        '{"status":"answer","answer_text":"回答[S1]","reason":null}',
        chunks,
    )
    assert isinstance(outcome, CitedAnswer)
    assert outcome.retrieved_chunks is chunks
    assert outcome.sources[0].url == chunks[0].source_url


@pytest.mark.parametrize(
    ("answer_text", "reason"),
    [
        ("回答", "abstain_answer_text_must_be_empty"),
        ("回答[S1]", "abstain_citation_detected"),
        ("https://evil.invalid", "abstain_url_detected"),
        ("[資料](path)", "abstain_markdown_link_detected"),
    ],
)
def test_abstain_rejects_every_nonempty_body(answer_text: str, reason: str) -> None:
    raw = (
        '{"status":"abstain","answer_text":"'
        + answer_text
        + '","reason":"insufficient_evidence"}'
    )
    with pytest.raises(AnswerContractError) as error:
        _finalize(raw)
    assert reason in error.value.reasons


def test_valid_abstain_is_normal_outcome_with_no_sources_and_keeps_chunks() -> None:
    chunks = (_chunk(),)
    outcome = _finalize(
        '{"status":"abstain","answer_text":"",'
        '"reason":"insufficient_evidence"}',
        chunks,
    )
    assert isinstance(outcome, AbstainedAnswer)
    assert outcome.reason_code == "insufficient_evidence"
    assert outcome.retrieved_chunks is chunks
    assert outcome.generation_attempts == 1
    assert not hasattr(outcome, "sources")


@pytest.mark.parametrize(
    "profile_name",
    ("default", "recommended-v1", "recommended-v2", "recommended"),
)
def test_profile_contract_uses_sanitized_knowledge_base_scope(
    profile_name: str,
) -> None:
    profile = runtime_profile(profile_name)
    contract = generation_contract_for_mode(
        profile.answer_mode,
        document_scope=(
            "uv Documentation\nhttps://internal.invalid/secret "
            "/workspace/private/index.faiss"
        ),
    )

    prompt = contract.build_initial_prompt("質問", (_chunk(),))

    assert (
        "あなたはuv Documentation [URL除去済み] [パス除去済み]に基づいて"
        "回答します。"
    ) in prompt
    assert "Python 3.13日本語公式ドキュメント" not in prompt
    assert "internal.invalid" not in prompt
    assert "/workspace/private" not in prompt


@pytest.mark.parametrize("answer_mode", ("legacy", "answer-or-abstain"))
def test_unscoped_contract_preserves_legacy_python_prompt(answer_mode: str) -> None:
    prompt = generation_contract_for_mode(answer_mode).build_initial_prompt(
        "質問",
        (_chunk(),),
    )

    assert "あなたはPython 3.13日本語公式ドキュメントに基づいて回答します。" in prompt
