from __future__ import annotations

import pytest

from python_doc_rag.models import AbstainedAnswer, SearchChunk
from python_doc_rag.pipeline import AnswerGenerationFailedError
from python_doc_rag.two_stage_answerability import (
    TWO_STAGE_ANSWER_MODE,
    TWO_STAGE_CONTRACT_REVISION,
    TwoStageAnswerabilityPipeline,
    build_answerability_choice_prompt,
)


class FakeRetriever:
    def __init__(self, chunks: tuple[SearchChunk, ...]) -> None:
        self.chunks = chunks
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, question: str, *, limit: int) -> tuple[SearchChunk, ...]:
        self.calls.append((question, limit))
        return self.chunks[:limit]


class FakeGenerator:
    def __init__(
        self,
        status: str,
        outputs: tuple[str, ...] = (),
    ) -> None:
        self.status = status
        self.outputs = outputs
        self.choice_calls: list[tuple[str, tuple[str, ...]]] = []
        self.generate_calls: list[tuple[str, int]] = []

    def choose_exact(self, prompt: str, *, choices: tuple[str, ...]) -> str:
        self.choice_calls.append((prompt, choices))
        return self.status

    def generate(self, prompt: str, *, max_new_tokens: int) -> str:
        self.generate_calls.append((prompt, max_new_tokens))
        return self.outputs[len(self.generate_calls) - 1]


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


def make_chunk(name: str, *, text: str = "根拠本文") -> SearchChunk:
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


def make_pipeline(
    generator: FakeGenerator,
    chunks: tuple[SearchChunk, ...] | None = None,
    *,
    budget: int = 100_000,
) -> TwoStageAnswerabilityPipeline:
    from python_doc_rag.generation import GenerationConfig

    return TwoStageAnswerabilityPipeline(
        retriever=FakeRetriever(chunks if chunks is not None else (make_chunk("one"),)),
        generator=generator,
        tokenizer=FakeTokenizer(),
        config=GenerationConfig(
            retrieval_limit=5,
            max_prompt_tokens=budget,
            max_new_tokens=123,
        ),
    )


def test_exact_abstain_skips_stage_two_and_leaks_no_answer_or_source() -> None:
    generator = FakeGenerator("abstain")
    pipeline = make_pipeline(generator)

    result = pipeline.answer("資料外の質問")

    assert isinstance(result, AbstainedAnswer)
    assert result.reason_code == "insufficient_evidence"
    assert result.generation_attempts == 1
    assert not generator.generate_calls
    assert generator.choice_calls[0][1] == ("answer", "abstain")
    assert pipeline.last_status_choice == "abstain"
    assert pipeline.answer_mode == TWO_STAGE_ANSWER_MODE
    assert pipeline.contract_revision == TWO_STAGE_CONTRACT_REVISION


def test_answer_stage_reuses_citation_finalization_and_metadata() -> None:
    chunk = make_chunk("answer")
    generator = FakeGenerator("answer", ("根拠に基づく回答。[S1]",))
    pipeline = make_pipeline(generator, (chunk,))

    result = pipeline.answer("回答できる質問")

    assert result.answer_text == "根拠に基づく回答。[S1]"
    assert result.sources[0].url == chunk.source_url
    assert result.retrieved_chunks == (chunk,)
    assert result.generation_attempts == 1
    assert len(generator.choice_calls) == 1
    assert len(generator.generate_calls) == 1
    assert generator.generate_calls[0][1] == 123


def test_answer_stage_retries_once_without_reusing_invalid_output() -> None:
    chunk = make_chunk("retry")
    generator = FakeGenerator(
        "answer",
        (
            "不正 https://attacker.invalid/ [S1]",
            "修正済み回答。[S1]",
        ),
    )
    pipeline = make_pipeline(generator, (chunk,))

    result = pipeline.answer("質問")

    assert result.generation_attempts == 2
    assert len(generator.generate_calls) == 2
    assert "attacker.invalid" not in generator.generate_calls[1][0]
    assert "不正" not in generator.generate_calls[1][0]
    assert pipeline.last_selected_chunks == (chunk,)


def test_answer_stage_fails_closed_after_one_retry() -> None:
    generator = FakeGenerator("answer", ("引用なし", "まだ引用なし"))
    pipeline = make_pipeline(generator)

    with pytest.raises(AnswerGenerationFailedError) as error:
        pipeline.answer("質問")

    assert error.value.first_reasons == ("citation_required",)
    assert error.value.second_reasons == ("citation_required",)
    assert len(generator.generate_calls) == 2


def test_choice_and_answer_prompts_hide_urls_paths_and_internal_metadata() -> None:
    chunk = make_chunk(
        "unsafe",
        text=(
            "本文 https://attacker.invalid/fake "
            "/workspace/private/source.html parent_id=hidden"
        ),
    )
    generator = FakeGenerator("answer", ("安全な回答。[S1]",))
    pipeline = make_pipeline(generator, (chunk,))

    pipeline.answer("質問 /workspace/private/question.txt")

    prompts = [generator.choice_calls[0][0], generator.generate_calls[0][0]]
    for prompt in prompts:
        assert "https://" not in prompt
        assert "/workspace/" not in prompt
        assert chunk.source_url not in prompt
        assert chunk.extra_metadata["source_path"] not in prompt
    assert "[URL除去済み]" in prompts[0]
    assert "[パス除去済み]" in prompts[0]


def test_empty_retrieval_skips_both_stages() -> None:
    generator = FakeGenerator("answer", ("unused[S1]",))
    result = make_pipeline(generator, ()).answer("質問")

    assert isinstance(result, AbstainedAnswer)
    assert result.reason_code == "no_retrieval_results"
    assert result.generation_attempts == 0
    assert not generator.choice_calls
    assert not generator.generate_calls


def test_choice_prompt_has_only_labels_and_sanitized_context() -> None:
    chunk = make_chunk("prompt", text="根拠 https://fake.invalid/item")

    prompt = build_answerability_choice_prompt("質問", (chunk,))

    assert "answerまたはabstain" in prompt
    assert "[S1]" in prompt
    assert "https://" not in prompt
    assert chunk.source_url not in prompt
