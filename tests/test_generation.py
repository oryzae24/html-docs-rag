import builtins
import importlib
import sys

import pytest

from python_doc_rag.generation import (
    ContextBudgetExceededError,
    build_generation_prompt,
    build_regeneration_prompt,
    select_contexts_within_budget,
)
from python_doc_rag.models import SearchChunk


class FakeTokenizer:
    """Count characters while recording explicit truncation settings."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, bool, bool]] = []

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        truncation: bool,
    ) -> list[int]:
        self.calls.append((text, add_special_tokens, truncation))
        return list(range(len(text)))


def make_chunk(
    text: str,
    *,
    title: str = "ページ",
    section: str = "節",
    url: str = "https://docs.python.org/ja/3.13/example.html",
) -> SearchChunk:
    return SearchChunk(
        text=text,
        page_title=title,
        section_title=section,
        source_url=url,
        category="tutorial",
        chunk_index=0,
        start_index=0,
    )


def test_prompts_never_contain_urls_from_any_input_field() -> None:
    chunk = make_chunk(
        (
            "本文https://example.invalid/fake "
            "/workspace/private/secret.html "
            "/content/gdrive/private/data.json"
        ),
        title="ページ www.title.invalid",
        section="節 ftp://section.invalid/file",
        url="https://trusted.invalid/source",
    )
    question = "質問 https://question.invalid/path"
    initial = build_generation_prompt(question, (chunk,))
    retry = build_regeneration_prompt(
        question,
        (chunk,),
        available_citations=("[S1]",),
    )

    for prompt in (initial, retry):
        assert "http" not in prompt.lower()
        assert "www." not in prompt.lower()
        assert chunk.source_url not in prompt
        assert "利用可能な引用番号: [S1]" in prompt
        assert "/workspace/private/secret.html" not in prompt
        assert "/content/gdrive/private/data.json" not in prompt
        assert "[URL除去済み]" in prompt
        assert "[パス除去済み]" in prompt


def test_budget_drops_whole_chunks_then_renumbers_remaining_context() -> None:
    tokenizer = FakeTokenizer()
    oversized = make_chunk("大" * 10_000)
    retained = make_chunk("短い本文")
    budget = len(
        build_regeneration_prompt(
            "質問",
            (retained,),
            available_citations=("[S1]",),
        )
    )

    selected = select_contexts_within_budget(
        "質問",
        (oversized, retained),
        tokenizer=tokenizer,
        max_prompt_tokens=budget,
    )

    assert selected == (retained,)
    assert selected[0].text == "短い本文"
    prompt = build_generation_prompt("質問", selected)
    assert "[S1]" in prompt
    assert "[S2]" not in prompt
    assert tokenizer.calls
    assert all(not add_special for _, add_special, _ in tokenizer.calls)
    assert all(not truncation for _, _, truncation in tokenizer.calls)


def test_budget_rejects_when_no_complete_chunk_fits() -> None:
    tokenizer = FakeTokenizer()
    with pytest.raises(ContextBudgetExceededError):
        select_contexts_within_budget(
            "質問",
            (make_chunk("本文"),),
            tokenizer=tokenizer,
            max_prompt_tokens=1,
        )


def test_generation_import_does_not_require_transformers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if name.split(".", maxsplit=1)[0] == "transformers":
            raise ModuleNotFoundError(name)
        return original_import(name, *args, **kwargs)

    sys.modules.pop("python_doc_rag.generation", None)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    module = importlib.import_module("python_doc_rag.generation")

    assert module.GenerationConfig().retrieval_limit == 5
