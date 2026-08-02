"""Pinned local generator candidates for the bounded quality tournament."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class GeneratorModelSpec:
    """One reproducible local generator and its official decoding settings."""

    key: str
    model_name: str
    revision: str
    license: str
    model_card_url: str
    language_evidence: str
    parameter_billions: float
    dtype: str
    template_kwargs: tuple[tuple[str, object], ...]
    generation_kwargs: tuple[tuple[str, Any], ...]
    sampling_seed: int | None
    trust_remote_code: bool = False

    def template_options(self) -> dict[str, object]:
        """Return chat-template options without exposing mutable global state."""
        return dict(self.template_kwargs)

    def generation_options(self) -> dict[str, Any]:
        """Return decoding options without exposing mutable global state."""
        return dict(self.generation_kwargs)


GENERATOR_MODEL_SPECS = (
    GeneratorModelSpec(
        key="baseline-qwen3-4b",
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        revision="cdbee75f17c01a7cc42f958dc650907174af0554",
        license="Apache-2.0",
        model_card_url="https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507",
        language_evidence="Existing fixed multilingual recommended-v1 baseline.",
        parameter_billions=4.0,
        dtype="bfloat16",
        template_kwargs=(),
        generation_kwargs=(("do_sample", False),),
        sampling_seed=None,
    ),
    GeneratorModelSpec(
        key="qwen3-8b",
        model_name="Qwen/Qwen3-8B",
        revision="b968826d9c46dd6066d109eabc6255188de91218",
        license="Apache-2.0",
        model_card_url="https://huggingface.co/Qwen/Qwen3-8B",
        language_evidence=(
            "The official card states support for over 100 languages and "
            "multilingual instruction following."
        ),
        parameter_billions=8.2,
        dtype="bfloat16",
        template_kwargs=(("enable_thinking", False),),
        generation_kwargs=(
            ("do_sample", True),
            ("temperature", 0.7),
            ("top_p", 0.8),
            ("top_k", 20),
            ("min_p", 0.0),
        ),
        sampling_seed=20260731,
    ),
    GeneratorModelSpec(
        key="qwen2.5-7b-instruct",
        model_name="Qwen/Qwen2.5-7B-Instruct",
        revision="a09a35458c702b33eeacc393d103063234e8bc28",
        license="Apache-2.0",
        model_card_url="https://huggingface.co/Qwen/Qwen2.5-7B-Instruct",
        language_evidence=(
            "The official card explicitly includes Japanese among more than "
            "29 supported languages and highlights structured-output following."
        ),
        parameter_billions=7.61,
        dtype="bfloat16",
        template_kwargs=(),
        generation_kwargs=(
            ("do_sample", True),
            ("temperature", 0.7),
            ("top_p", 0.8),
            ("top_k", 20),
            ("repetition_penalty", 1.05),
        ),
        sampling_seed=20260731,
    ),
)


def generator_model_spec(key: str) -> GeneratorModelSpec:
    """Resolve the baseline or one of exactly two additional candidates."""
    for spec in GENERATOR_MODEL_SPECS:
        if spec.key == key:
            return spec
    raise ValueError(f"unsupported generator model key: {key}")
