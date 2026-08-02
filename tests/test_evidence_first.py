from __future__ import annotations

import json

import pytest

from python_doc_rag.evidence_first import (
    EVIDENCE_FIRST_ANSWER_MODE,
    EVIDENCE_FIRST_REVISION,
    EvidenceContractError,
    EvidenceFirstPipeline,
    EvidenceItem,
    build_evidence_answer_prompt,
    parse_evidence_contract,
)
from python_doc_rag.models import AbstainedAnswer, SearchChunk
from python_doc_rag.pipeline import AnswerGenerationFailedError


class FakeRetriever:
    def __init__(self, chunks: tuple[SearchChunk, ...]) -> None:
        self.chunks = chunks

    def retrieve(self, question: str, *, limit: int) -> tuple[SearchChunk, ...]:
        return self.chunks[:limit]


class FakeGenerator:
    def __init__(self, outputs: tuple[str, ...]) -> None:
        self.outputs = outputs
        self.calls: list[tuple[str, int]] = []

    def generate(self, prompt: str, *, max_new_tokens: int) -> str:
        self.calls.append((prompt, max_new_tokens))
        return self.outputs[len(self.calls) - 1]


class FakeTokenizer:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        truncation: bool,
    ) -> list[int]:
        assert not add_special_tokens
        assert not truncation
        self.prompts.append(text)
        return list(range(len(text)))


def make_chunk(
    name: str = "one",
    *,
    text: str = "aclose()を await して終了処理を行います。",
) -> SearchChunk:
    return SearchChunk(
        text=text,
        page_title=f"ページ{name}",
        section_title=f"節{name}",
        source_url=f"https://trusted.invalid/{name}",
        category="library",
        chunk_index=0,
        start_index=0,
        extra_metadata={"source_path": f"/workspace/private/{name}.html"},
    )


def evidence_json(*facts: str, source: str = "S1") -> str:
    return json.dumps(
        {"evidence": [{"source": source, "facts": list(facts)}]},
        ensure_ascii=False,
    )


def make_pipeline(
    outputs: tuple[str, ...],
    *,
    chunks: tuple[SearchChunk, ...] | None = None,
) -> tuple[EvidenceFirstPipeline, FakeGenerator]:
    from python_doc_rag.generation import GenerationConfig

    generator = FakeGenerator(outputs)
    pipeline = EvidenceFirstPipeline(
        retriever=FakeRetriever(chunks if chunks is not None else (make_chunk(),)),
        generator=generator,
        tokenizer=FakeTokenizer(),
        config=GenerationConfig(
            retrieval_limit=5,
            max_prompt_tokens=100_000,
            max_new_tokens=111,
        ),
    )
    return pipeline, generator


def test_evidence_parser_accepts_only_extracts_from_matching_source() -> None:
    first = make_chunk(text="前半  根拠となる文です。 後半")
    second = make_chunk("two", text="別の根拠です。")

    evidence = parse_evidence_contract(
        '{"evidence":[{"source":"S1","facts":["根拠となる文です。"]},'
        '{"source":"S2","facts":["別の根拠です。"]}]}',
        (first, second),
    )

    assert evidence == (
        EvidenceItem("S1", ("根拠となる文です。",)),
        EvidenceItem("S2", ("別の根拠です。",)),
    )


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        ('{"evidence":[{"source":"S2","facts":["根拠"]}]}', "source"),
        ('{"evidence":[{"source":"S1","facts":["資料外"]}]}', "unsupported"),
        ('{"evidence":[{"source":"S1","facts":[]}]}', "facts"),
        ('{"evidence":[],"extra":1}', "schema"),
        ('```json\n{"evidence":[]}\n```', "json"),
    ],
)
def test_evidence_parser_rejects_invalid_source_schema_or_support(
    raw: str,
    reason: str,
) -> None:
    with pytest.raises(EvidenceContractError) as error:
        parse_evidence_contract(raw, (make_chunk(text="根拠"),))

    assert any(reason in item for item in error.value.reasons)


def test_empty_evidence_abstains_without_answer_generation_or_leakage() -> None:
    pipeline, generator = make_pipeline(('{"evidence":[]}',))

    result = pipeline.answer("資料外質問")

    assert isinstance(result, AbstainedAnswer)
    assert result.reason_code == "insufficient_evidence"
    assert result.generation_attempts == 1
    assert len(generator.calls) == 1
    assert pipeline.last_evidence == ()
    assert pipeline.answer_mode == EVIDENCE_FIRST_ANSWER_MODE
    assert pipeline.contract_revision == EVIDENCE_FIRST_REVISION


def test_answer_prompt_receives_only_validated_evidence_not_full_context() -> None:
    chunk = make_chunk(text="必要な根拠です。送ってはいけない残りの本文。")
    pipeline, generator = make_pipeline(
        (evidence_json("必要な根拠です。"), "回答。[S1]"),
        chunks=(chunk,),
    )

    result = pipeline.answer("質問")

    assert result.answer_text == "回答。[S1]"
    answer_prompt = generator.calls[1][0]
    assert "必要な根拠です。" in answer_prompt
    assert "送ってはいけない残りの本文" not in answer_prompt
    assert chunk.source_url not in answer_prompt
    assert result.sources[0].url == chunk.source_url
    assert pipeline.last_evidence == (EvidenceItem("S1", ("必要な根拠です。",)),)


def test_invalid_evidence_retries_without_exposing_rejected_output() -> None:
    pipeline, generator = make_pipeline(
        (
            evidence_json("資料外の捏造"),
            evidence_json("aclose()を await して終了処理を行います。"),
            "回答。[S1]",
        )
    )

    result = pipeline.answer("質問")

    assert result.answer_text == "回答。[S1]"
    assert pipeline.last_evidence_attempts == 2
    assert "資料外の捏造" not in generator.calls[1][0]
    assert len(generator.calls) == 3


def test_invalid_evidence_twice_fails_closed_and_continuation_is_possible() -> None:
    pipeline, generator = make_pipeline(
        (evidence_json("捏造1"), evidence_json("捏造2"))
    )

    with pytest.raises(AnswerGenerationFailedError) as error:
        pipeline.answer("質問")

    assert error.value.first_reasons == ("unsupported_evidence_fact",)
    assert error.value.second_reasons == ("unsupported_evidence_fact",)
    assert len(generator.calls) == 2


def test_answer_retry_uses_same_evidence_and_hides_invalid_answer() -> None:
    pipeline, generator = make_pipeline(
        (
            evidence_json("aclose()を await して終了処理を行います。"),
            "URL https://attacker.invalid [S1]",
            "修正回答。[S1]",
        )
    )

    result = pipeline.answer("質問")

    assert result.generation_attempts == 2
    assert "attacker.invalid" not in generator.calls[2][0]
    assert "aclose()を await" in generator.calls[2][0]


def test_answer_prompt_rejects_no_metadata_and_uses_available_labels() -> None:
    prompt = build_evidence_answer_prompt(
        "質問 /workspace/private/question.txt",
        (EvidenceItem("S2", ("検証済み根拠",)),),
    )

    assert "[S2]" in prompt
    assert "検証済み根拠" in prompt
    assert "/workspace/private" not in prompt
    assert "[パス除去済み]" in prompt
