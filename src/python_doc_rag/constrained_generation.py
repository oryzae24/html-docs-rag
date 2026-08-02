"""Small model-independent token constraints for exact local choices."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


class ExactChoiceGenerationError(ValueError):
    """Report tokenizer or generation output that cannot encode one exact choice."""


@dataclass(frozen=True, slots=True)
class ExactChoiceTokenTrie:
    """Allow only complete token sequences for a fixed set of string choices."""

    sequences: tuple[tuple[str, tuple[int, ...]], ...]
    eos_token_id: int

    @classmethod
    def from_tokenizer(
        cls,
        tokenizer: Any,
        choices: Sequence[str],
    ) -> ExactChoiceTokenTrie:
        """Tokenize unique non-empty choices without adding model special tokens."""
        normalized = _validate_choices(choices)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if not isinstance(eos_token_id, int) or isinstance(eos_token_id, bool):
            raise ExactChoiceGenerationError("tokenizer requires one integer EOS token")
        sequences: list[tuple[str, tuple[int, ...]]] = []
        seen_tokens: set[tuple[int, ...]] = set()
        for choice in normalized:
            tokens = tuple(
                int(token)
                for token in tokenizer.encode(choice, add_special_tokens=False)
            )
            if not tokens:
                raise ExactChoiceGenerationError("choice tokenization must not be empty")
            if tokens in seen_tokens:
                raise ExactChoiceGenerationError(
                    "distinct choices produced the same token sequence"
                )
            seen_tokens.add(tokens)
            sequences.append((choice, tokens))
        return cls(tuple(sequences), eos_token_id)

    @property
    def max_sequence_length(self) -> int:
        """Return the longest exact choice before the terminal EOS token."""
        return max(len(tokens) for _, tokens in self.sequences)

    def allowed_tokens(self, generated: Sequence[int]) -> tuple[int, ...]:
        """Return deterministic next tokens or reject an invalid prefix."""
        prefix = tuple(int(token) for token in generated)
        allowed: set[int] = set()
        for _, tokens in self.sequences:
            if len(prefix) < len(tokens) and tokens[: len(prefix)] == prefix:
                allowed.add(tokens[len(prefix)])
            elif prefix == tokens:
                allowed.add(self.eos_token_id)
        if not allowed:
            raise ExactChoiceGenerationError("model produced an invalid choice prefix")
        return tuple(sorted(allowed))

    def resolve(self, generated: Sequence[int]) -> str:
        """Map an exact generated sequence, with optional final EOS, to its choice."""
        tokens = tuple(int(token) for token in generated)
        if tokens and tokens[-1] == self.eos_token_id:
            tokens = tokens[:-1]
        for choice, candidate in self.sequences:
            if tokens == candidate:
                return choice
        raise ExactChoiceGenerationError("generation did not end at an exact choice")


def token_ids(value: Any) -> tuple[int, ...]:
    """Convert a one-dimensional token container without depending on torch."""
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        converted = tolist()
    elif hasattr(value, "values"):
        converted = value.values
    else:
        converted = value
    if not isinstance(converted, (list, tuple)):
        raise TypeError("token IDs must be a one-dimensional sequence")
    return tuple(int(token) for token in converted)


def exact_choice_settings(choices: Sequence[str]) -> Mapping[str, object]:
    """Return stable provenance for the constrained choice boundary."""
    return {
        "revision": "exact-choice-token-trie-v1",
        "choices": list(_validate_choices(choices)),
        "decoding": "greedy-prefix-allowed-tokens",
    }


def _validate_choices(choices: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(choices)
    if len(normalized) < 2:
        raise ValueError("exact choice requires at least two choices")
    if any(not isinstance(choice, str) or not choice for choice in normalized):
        raise TypeError("choices must be non-empty strings")
    if len(set(normalized)) != len(normalized):
        raise ValueError("choices must be unique")
    return normalized
