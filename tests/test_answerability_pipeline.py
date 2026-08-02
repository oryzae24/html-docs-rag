import pytest

from python_doc_rag.answer_contract import AnswerOrAbstainGenerationContract
from python_doc_rag.generation import GenerationConfig
from python_doc_rag.models import AbstainedAnswer, CitedAnswer, SearchChunk
from python_doc_rag.pipeline import AnswerGenerationFailedError, RagPipeline


class Retriever:
    def __init__(self, chunks: tuple[SearchChunk, ...]) -> None:
        self.chunks = chunks

    def retrieve(self, question: str, *, limit: int) -> tuple[SearchChunk, ...]:
        return self.chunks[:limit]


class Generator:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.calls: list[str] = []

    def generate(self, prompt: str, *, max_new_tokens: int) -> str:
        self.calls.append(prompt)
        return self.outputs[len(self.calls) - 1]


class Tokenizer:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def encode(
        self, text: str, *, add_special_tokens: bool, truncation: bool
    ) -> list[int]:
        assert not add_special_tokens
        assert not truncation
        self.calls.append(text)
        return list(range(len(text)))


def chunk() -> SearchChunk:
    return SearchChunk(
        text="根拠 /workspace/private/file https://evil.invalid",
        page_title="ページ",
        section_title="節",
        source_url="https://trusted.invalid/source",
        category="library",
        chunk_index=0,
        start_index=0,
        extra_metadata={"source_path": "/content/gdrive/private.json"},
    )


def pipeline(outputs: list[str], chunks: tuple[SearchChunk, ...] | None = None):
    generator = Generator(outputs)
    tokenizer = Tokenizer()
    result = RagPipeline(
        retriever=Retriever(chunks if chunks is not None else (chunk(),)),
        generator=generator,
        tokenizer=tokenizer,
        config=GenerationConfig(max_prompt_tokens=100_000),
        generation_contract=AnswerOrAbstainGenerationContract(),
    )
    return result, generator, tokenizer


ANSWER = '{"status":"answer","answer_text":"回答[S1]","reason":null}'
ABSTAIN = '{"status":"abstain","answer_text":"","reason":"insufficient_evidence"}'


def test_pipeline_returns_valid_answer() -> None:
    service, generator, _ = pipeline([ANSWER])
    outcome = service.answer("質問")
    assert isinstance(outcome, CitedAnswer)
    assert outcome.generation_attempts == 1
    assert len(generator.calls) == 1


def test_pipeline_returns_valid_abstention_as_normal_outcome() -> None:
    service, _, _ = pipeline([ABSTAIN])
    outcome = service.answer("質問")
    assert isinstance(outcome, AbstainedAnswer)
    assert outcome.reason_code == "insufficient_evidence"
    assert outcome.generation_attempts == 1


@pytest.mark.parametrize("retry", [ANSWER, ABSTAIN])
def test_invalid_first_output_retries_once_without_echoing_it(retry: str) -> None:
    invalid = "INVALID https://attacker.invalid /workspace/secret"
    service, generator, tokenizer = pipeline([invalid, retry])
    outcome = service.answer("質問")
    assert outcome.generation_attempts == 2
    assert len(generator.calls) == 2
    assert invalid not in generator.calls[1]
    assert "attacker.invalid" not in generator.calls[1]
    assert tokenizer.calls[-1] == generator.calls[1]


def test_two_invalid_outputs_fail_closed() -> None:
    service, generator, _ = pipeline(["invalid", "still invalid"])
    with pytest.raises(AnswerGenerationFailedError) as error:
        service.answer("質問")
    assert error.value.first_reasons == ("invalid_json",)
    assert error.value.second_reasons == ("invalid_json",)
    assert len(generator.calls) == 2


def test_initial_and_retry_use_same_selected_chunks_without_urls_or_paths() -> None:
    service, generator, _ = pipeline(["invalid", ANSWER])
    outcome = service.answer("質問 /workspace/question")
    assert outcome.retrieved_chunks == (chunk(),)
    for prompt in generator.calls:
        assert "https://" not in prompt
        assert "/workspace/" not in prompt
        assert "/content/gdrive" not in prompt
        assert "[S1]" in prompt


def test_empty_retrieval_skips_generator_and_has_zero_generation_time() -> None:
    service, generator, _ = pipeline([ANSWER], ())
    outcome = service.answer("質問")
    assert isinstance(outcome, AbstainedAnswer)
    assert outcome.reason_code == "no_retrieval_results"
    assert outcome.generation_attempts == 0
    assert service.last_generation_seconds == 0.0
    assert not generator.calls
