import pytest

from python_doc_rag.generation import (
    ChatTemplatePromptSerializer,
    GenerationConfig,
    PromptSerializer,
    build_regeneration_prompt,
)
from python_doc_rag.models import AbstainedAnswer, SearchChunk
from python_doc_rag.pipeline import (
    AnswerGenerationFailedError,
    RagPipeline,
)


class FakeRetriever:
    """Return a fixed ranked tuple and record retrieval calls."""

    def __init__(self, chunks: tuple[SearchChunk, ...]) -> None:
        self.chunks = chunks
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, question: str, *, limit: int) -> tuple[SearchChunk, ...]:
        self.calls.append((question, limit))
        return self.chunks[:limit]


class CapturingGenerator:
    """Return configured outputs while recording its complete call boundary."""

    def __init__(self, initial_answer: str, retry_answer: str = "") -> None:
        self.answers = [initial_answer, retry_answer]
        self.calls: list[tuple[str, int]] = []

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
    ) -> str:
        self.calls.append((prompt, max_new_tokens))
        return self.answers[len(self.calls) - 1]


class FakeTokenizer:
    """Count and capture exact prompt strings without truncation."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.chat_calls: list[tuple[list[dict[str, str]], bool, bool]] = []
        self.chat_options: list[dict[str, object]] = []

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        truncation: bool,
    ) -> list[int]:
        assert not add_special_tokens
        assert not truncation
        self.calls.append(text)
        return list(range(len(text)))

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        **kwargs: object,
    ) -> str:
        self.chat_calls.append(
            (conversation, tokenize, add_generation_prompt)
        )
        self.chat_options.append(kwargs)
        return f"<user>{conversation[0]['content']}</user><assistant>"


def make_chunk(
    name: str,
    *,
    text: str = "検索本文",
    url: str | None = None,
    source_path: str | None = None,
) -> SearchChunk:
    return SearchChunk(
        extra_metadata={"source_path": source_path} if source_path else {},
        text=text,
        page_title=f"ページ{name}",
        section_title=f"節{name}",
        source_url=url or f"https://trusted.invalid/{name}",
        category="tutorial",
        chunk_index=0,
        start_index=0,
    )


def make_pipeline(
    chunks: tuple[SearchChunk, ...],
    generator: CapturingGenerator,
    *,
    budget: int = 100_000,
    tokenizer: FakeTokenizer | None = None,
    prompt_serializer: PromptSerializer | None = None,
) -> RagPipeline:
    return RagPipeline(
        retriever=FakeRetriever(chunks),
        generator=generator,
        tokenizer=tokenizer or FakeTokenizer(),
        prompt_serializer=prompt_serializer,
        config=GenerationConfig(
            retrieval_limit=5,
            max_prompt_tokens=budget,
            max_new_tokens=321,
        ),
    )


def test_model_independent_e2e_preserves_rank_to_citation_mapping() -> None:
    first = make_chunk("first")
    second = make_chunk("second")
    generator = CapturingGenerator("第2位の根拠[S2]。第1位の根拠[S1]。")

    result = make_pipeline((first, second), generator).answer("質問")

    assert result.retrieved_chunks == (first, second)
    assert [source.label for source in result.sources] == ["S2", "S1"]
    assert [source.url for source in result.sources] == [
        second.source_url,
        first.source_url,
    ]
    assert result.generation_attempts == 1
    assert len(generator.calls) == 1
    assert generator.calls[0][1] == 321


def test_pipeline_succeeds_after_exactly_one_regeneration() -> None:
    chunk = make_chunk("one")
    generator = CapturingGenerator(
        "不正URL https://untrusted.invalid [S1]",
        "修正された回答[S1]",
    )

    result = make_pipeline((chunk,), generator).answer("質問")

    assert result.generation_attempts == 2
    assert len(generator.calls) == 2
    assert "回答:" in generator.calls[0][0]
    assert "修正回答:" in generator.calls[1][0]
    assert "不正URL" not in generator.calls[1][0]
    assert "https://untrusted.invalid" not in generator.calls[1][0]


def test_pipeline_fails_when_both_generation_attempts_are_invalid() -> None:
    generator = CapturingGenerator("引用なし", "まだ引用なし")
    pipeline = make_pipeline((make_chunk("one"),), generator)

    with pytest.raises(AnswerGenerationFailedError) as error:
        pipeline.answer("質問")

    assert error.value.first_reasons == ("citation_required",)
    assert error.value.second_reasons == ("citation_required",)
    assert len(generator.calls) == 2


def test_empty_retrieval_does_not_call_generator() -> None:
    generator = CapturingGenerator("呼ばれてはいけない")
    result = make_pipeline((), generator).answer("質問")

    assert isinstance(result, AbstainedAnswer)
    assert result.reason_code == "no_retrieval_results"
    assert result.retrieved_chunks == ()
    assert result.generation_attempts == 0
    assert not generator.calls


def test_generator_receives_only_sanitized_measured_prompt() -> None:
    source_url = "https://trusted.invalid/source"
    source_path = "/workspace/private/secret.html"
    chunk = make_chunk(
        "unsafe",
        text=(
            "本文 https://example.invalid/fake "
            "/workspace/private/secret.html "
            "/content/gdrive/private/data.json"
        ),
        url=source_url,
        source_path=source_path,
    )
    generator = CapturingGenerator("安全な回答[S1]")
    tokenizer = FakeTokenizer()

    result = make_pipeline(
        (chunk,),
        generator,
        tokenizer=tokenizer,
    ).answer("質問 /content/gdrive/private/data.json")

    prompt, max_new_tokens = generator.calls[0]
    assert isinstance(prompt, str)
    assert max_new_tokens == 321
    assert tokenizer.calls[-1] == prompt
    assert source_url not in prompt
    assert source_path not in prompt
    assert "https://example.invalid/fake" not in prompt
    assert "/content/gdrive/private/data.json" not in prompt
    assert "[URL除去済み]" in prompt
    assert "[パス除去済み]" in prompt
    assert result.retrieved_chunks == (chunk,)


def test_regeneration_uses_same_sanitized_measured_boundary() -> None:
    chunk = make_chunk(
        "retry",
        text=(
            "本文 https://example.invalid/fake "
            "/workspace/private/secret.html "
            "/content/gdrive/private/data.json"
        ),
        source_path="/workspace/private/secret.html",
    )
    generator = CapturingGenerator(
        "不正 https://attacker.invalid/leak [S1]",
        "修正回答[S1]",
    )
    tokenizer = FakeTokenizer()

    make_pipeline((chunk,), generator, tokenizer=tokenizer).answer("質問")

    assert len(generator.calls) == 2
    retry_prompt = generator.calls[1][0]
    assert tokenizer.calls[-1] == retry_prompt
    assert "https://" not in retry_prompt
    assert "/workspace/private/secret.html" not in retry_prompt
    assert "/content/gdrive/private/data.json" not in retry_prompt
    assert "attacker.invalid" not in retry_prompt
    assert "不正 " not in retry_prompt


def test_budget_excluded_chunk_is_not_sent_to_generator() -> None:
    excluded_text = "除外対象" + ("大" * 10_000)
    retained_text = "採用対象"
    retained = make_chunk("retained", text=retained_text)
    retry_prompt_length = len(
        build_regeneration_prompt(
            "質問",
            (retained,),
            available_citations=("[S1]",),
        )
    )
    generator = CapturingGenerator("採用された根拠[S1]")

    result = make_pipeline(
        (
            make_chunk("excluded", text=excluded_text),
            retained,
        ),
        generator,
        budget=retry_prompt_length,
    ).answer("質問")

    prompt = generator.calls[0][0]
    assert excluded_text not in prompt
    assert "除外対象" not in prompt
    assert retained_text in prompt
    assert "[S1]" in prompt
    assert "[S2]" not in prompt
    assert result.retrieved_chunks == (retained,)


def test_chat_template_final_string_is_counted_and_generated_unchanged() -> None:
    tokenizer = FakeTokenizer()
    serializer = ChatTemplatePromptSerializer(tokenizer)
    generator = CapturingGenerator("テンプレート済み回答[S1]")
    chunk = make_chunk(
        "chat",
        text=(
            "本文 https://example.invalid/fake "
            "/workspace/private/secret.html"
        ),
        source_path="/content/gdrive/private/data.json",
    )

    make_pipeline(
        (chunk,),
        generator,
        tokenizer=tokenizer,
        prompt_serializer=serializer,
    ).answer("質問")

    final_prompt = generator.calls[0][0]
    assert tokenizer.calls[-1] == final_prompt
    assert final_prompt.startswith("<user>")
    assert final_prompt.endswith("</user><assistant>")
    assert "https://example.invalid/fake" not in final_prompt
    assert "/workspace/private/secret.html" not in final_prompt
    assert "/content/gdrive/private/data.json" not in final_prompt
    assert tokenizer.chat_calls
    assert all(not tokenize for _, tokenize, _ in tokenizer.chat_calls)
    assert all(add_prompt for _, _, add_prompt in tokenizer.chat_calls)


def test_chat_template_can_disable_thinking_without_changing_prompt_content() -> None:
    tokenizer = FakeTokenizer()
    serializer = ChatTemplatePromptSerializer(
        tokenizer, template_kwargs={"enable_thinking": False}
    )

    serialized = serializer.serialize("prepared prompt")

    assert serialized == "<user>prepared prompt</user><assistant>"
    assert tokenizer.chat_options == [{"enable_thinking": False}]


def test_pipeline_uses_custom_context_selector_once_for_both_attempts() -> None:
    first = make_chunk("first")
    second = make_chunk("second")
    generator = CapturingGenerator("引用なし", "修正回答[S1]")
    calls: list[tuple[str, tuple[SearchChunk, ...]]] = []

    def select(question, candidates, **kwargs):
        del kwargs
        calls.append((question, tuple(candidates)))
        return (candidates[1],)

    pipeline = RagPipeline(
        retriever=FakeRetriever((first, second)),
        generator=generator,
        tokenizer=FakeTokenizer(),
        config=GenerationConfig(
            retrieval_limit=5,
            max_prompt_tokens=100_000,
            max_new_tokens=321,
        ),
        context_selector=select,
    )

    result = pipeline.answer("質問")

    assert calls == [("質問", (first, second))]
    assert pipeline.last_selected_chunks == (second,)
    assert result.retrieved_chunks == (second,)
    assert all("ページsecond" in prompt for prompt, _ in generator.calls)
    assert all("ページfirst" not in prompt for prompt, _ in generator.calls)
