from __future__ import annotations

import pytest

from python_doc_rag.constrained_generation import (
    ExactChoiceGenerationError,
    ExactChoiceTokenTrie,
    exact_choice_settings,
    token_ids,
)


class FakeTokenizer:
    eos_token_id = 99

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return {
            "answer": [10],
            "abstain": [20, 21],
            "duplicate": [10],
        }[text]


class TensorLike:
    def tolist(self) -> list[int]:
        return [4, 5]


def test_exact_choice_trie_allows_only_valid_next_tokens() -> None:
    trie = ExactChoiceTokenTrie.from_tokenizer(
        FakeTokenizer(),
        ("answer", "abstain"),
    )

    assert trie.allowed_tokens(()) == (10, 20)
    assert trie.allowed_tokens((20,)) == (21,)
    assert trie.allowed_tokens((10,)) == (99,)
    assert trie.resolve((10, 99)) == "answer"
    assert trie.resolve((20, 21)) == "abstain"
    assert trie.max_sequence_length == 2


def test_exact_choice_trie_rejects_invalid_or_ambiguous_tokens() -> None:
    tokenizer = FakeTokenizer()
    trie = ExactChoiceTokenTrie.from_tokenizer(
        tokenizer,
        ("answer", "abstain"),
    )

    with pytest.raises(ExactChoiceGenerationError, match="invalid choice prefix"):
        trie.allowed_tokens((55,))
    with pytest.raises(ExactChoiceGenerationError, match="exact choice"):
        trie.resolve((20, 99))
    with pytest.raises(ExactChoiceGenerationError, match="same token sequence"):
        ExactChoiceTokenTrie.from_tokenizer(
            tokenizer,
            ("answer", "duplicate"),
        )


def test_token_container_conversion_and_provenance_are_model_independent() -> None:
    assert token_ids(TensorLike()) == (4, 5)
    assert exact_choice_settings(("answer", "abstain")) == {
        "revision": "exact-choice-token-trie-v1",
        "choices": ["answer", "abstain"],
        "decoding": "greedy-prefix-allowed-tokens",
    }


@pytest.mark.parametrize("choices", [(), ("answer",), ("answer", "answer")])
def test_invalid_choice_sets_are_rejected(choices: tuple[str, ...]) -> None:
    with pytest.raises((TypeError, ValueError, ExactChoiceGenerationError)):
        ExactChoiceTokenTrie.from_tokenizer(FakeTokenizer(), choices)
