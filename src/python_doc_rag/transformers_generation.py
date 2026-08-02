"""Lazy Hugging Face Transformers integration for prepared RAG prompts."""

from __future__ import annotations

import importlib
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from python_doc_rag.constrained_generation import (
    ExactChoiceTokenTrie,
    token_ids,
)


class InputTokenLimitExceededError(ValueError):
    """Raised when a serialized prompt exceeds the configured model input limit."""


@dataclass(frozen=True, slots=True)
class GenerationMetrics:
    """Measurements for one model.generate call."""

    input_tokens: int
    generated_tokens: int
    elapsed_seconds: float


class TransformersPromptEncoder:
    """Cache the exact tensor inputs measured immediately before generation."""

    def __init__(self, tokenizer: Any, *, device: str) -> None:
        self._tokenizer = tokenizer
        self._device = device
        self._cached_key: tuple[str, bool, bool] | None = None
        self._cached_inputs: dict[str, Any] | None = None

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        truncation: bool,
    ) -> Any:
        """Prepare and cache model inputs, returning the measured token row."""
        inputs = self._prepare(
            text,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
        )
        return inputs["input_ids"][0]

    def model_inputs(self, prompt: str) -> dict[str, Any]:
        """Return the cached safe inputs or prepare them once for direct use."""
        expected_key = (prompt, False, False)
        if self._cached_key != expected_key or self._cached_inputs is None:
            return self._prepare(
                prompt,
                add_special_tokens=False,
                truncation=False,
            )
        return self._cached_inputs

    def _prepare(
        self,
        prompt: str,
        *,
        add_special_tokens: bool,
        truncation: bool,
    ) -> dict[str, Any]:
        encoded = self._tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=add_special_tokens,
            truncation=truncation,
        )
        inputs = {
            name: tensor.to(self._device)
            for name, tensor in encoded.items()
        }
        self._cached_key = (prompt, add_special_tokens, truncation)
        self._cached_inputs = inputs
        return inputs


class TransformersAnswerGenerator:
    """Generate from a final prompt string without rebuilding RAG context."""

    def __init__(
        self,
        tokenizer: Any,
        model: Any,
        *,
        model_name: str,
        revision: str | None = None,
        device: str = "auto",
        dtype: str | Any = "auto",
        max_input_tokens: int = 8192,
        trust_remote_code: bool = False,
        model_load_seconds: float = 0.0,
        generation_kwargs: Mapping[str, Any] | None = None,
        sampling_seed: int | None = None,
    ) -> None:
        """Reuse injected tokenizer/model objects and prepare them for inference."""
        _validate_model_settings(model_name, max_input_tokens)
        self._torch = _import_torch()
        self._device = resolve_device(device, torch_module=self._torch)
        self._dtype = resolve_torch_dtype(
            dtype,
            self._device,
            torch_module=self._torch,
        )
        self._tokenizer = tokenizer
        self._prompt_tokenizer = TransformersPromptEncoder(
            tokenizer,
            device=self._device,
        )
        self._model = model.to(device=self._device, dtype=self._dtype)
        self._model.eval()
        self._model_name = model_name
        self._requested_revision = revision
        self._max_input_tokens = max_input_tokens
        self._trust_remote_code = trust_remote_code
        self._model_load_seconds = model_load_seconds
        self._generation_kwargs = _validated_generation_kwargs(generation_kwargs)
        self._sampling_seed = sampling_seed
        if sampling_seed is not None:
            manual_seed = getattr(self._torch, "manual_seed", None)
            if callable(manual_seed):
                manual_seed(sampling_seed)
        self._history: list[GenerationMetrics] = []
        self._history_base = 0
        self._generation_count = 0
        self._history_limit: int | None = None

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        *,
        revision: str | None = None,
        device: str = "auto",
        dtype: str = "auto",
        max_input_tokens: int = 8192,
        trust_remote_code: bool = False,
        generation_kwargs: Mapping[str, Any] | None = None,
        sampling_seed: int | None = None,
    ) -> TransformersAnswerGenerator:
        """Load one tokenizer/model pair with explicit security and dtype settings."""
        _validate_model_settings(model_name, max_input_tokens)
        torch = _import_torch()
        transformers = _import_transformers()
        resolved_device = resolve_device(device, torch_module=torch)
        resolved_dtype = resolve_torch_dtype(
            dtype,
            resolved_device,
            torch_module=torch,
        )
        common_options: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
        }
        if revision is not None:
            common_options["revision"] = revision

        started_at = time.perf_counter()
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name,
            **common_options,
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=resolved_dtype,
            **common_options,
        )
        load_seconds = time.perf_counter() - started_at
        return cls(
            tokenizer,
            model,
            model_name=model_name,
            revision=revision,
            device=resolved_device,
            dtype=resolved_dtype,
            max_input_tokens=max_input_tokens,
            trust_remote_code=trust_remote_code,
            model_load_seconds=load_seconds,
            generation_kwargs=generation_kwargs,
            sampling_seed=sampling_seed,
        )

    @property
    def tokenizer(self) -> Any:
        """Return the reused tokenizer for chat template serialization."""
        return self._tokenizer

    @property
    def prompt_tokenizer(self) -> TransformersPromptEncoder:
        """Return the shared encoder used for Pipeline counting and generation."""
        return self._prompt_tokenizer

    @property
    def model_name(self) -> str:
        """Return the configured Hugging Face model identifier."""
        return self._model_name

    @property
    def model_revision(self) -> str:
        """Return the resolved model revision when Transformers exposes it."""
        config = getattr(self._model, "config", None)
        resolved = getattr(config, "_commit_hash", None)
        return str(resolved or self._requested_revision or "main")

    @property
    def device(self) -> str:
        """Return the selected inference device."""
        return self._device

    @property
    def dtype_name(self) -> str:
        """Return the selected torch dtype as a stable short name."""
        return str(self._dtype).removeprefix("torch.")

    @property
    def max_input_tokens(self) -> int:
        """Return the hard model input limit."""
        return self._max_input_tokens

    @property
    def trust_remote_code(self) -> bool:
        """Return whether remote model code loading was allowed."""
        return self._trust_remote_code

    @property
    def model_load_seconds(self) -> float:
        """Return tokenizer and model loading wall time."""
        return self._model_load_seconds

    @property
    def generation_history(self) -> tuple[GenerationMetrics, ...]:
        """Return immutable measurements for completed generation calls."""
        return tuple(self._history)

    @property
    def generation_cursor(self) -> int:
        """Return a monotonic cursor without copying retained measurements."""
        return self._generation_count

    def generation_metrics_since(self, cursor: int) -> tuple[GenerationMetrics, ...]:
        """Return retained metrics after a previously observed cursor."""
        if (
            isinstance(cursor, bool)
            or not isinstance(cursor, int)
            or cursor < self._history_base
            or cursor > self._generation_count
        ):
            raise ValueError("generation cursor is outside retained history")
        return tuple(self._history[cursor - self._history_base :])

    def set_generation_history_limit(self, limit: int) -> None:
        """Bound retained metrics while preserving a monotonic call cursor."""
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("generation history limit must be positive")
        self._history_limit = limit
        self._trim_generation_history()

    @property
    def generation_kwargs(self) -> dict[str, Any]:
        """Return a copy of the explicit decoding configuration."""
        return dict(self._generation_kwargs)

    @property
    def sampling_seed(self) -> int | None:
        """Return the one-time seed used for a sampling run."""
        return self._sampling_seed

    def generate(self, prompt: str, *, max_new_tokens: int) -> str:
        """Use measured tensors unchanged and decode only newly generated tokens."""
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        if (
            not isinstance(max_new_tokens, int)
            or isinstance(max_new_tokens, bool)
            or max_new_tokens <= 0
        ):
            raise ValueError("max_new_tokens must be a positive integer")

        model_inputs = self._prompt_tokenizer.model_inputs(prompt)
        input_ids = model_inputs["input_ids"]
        input_tokens = int(input_ids.shape[-1])
        if input_tokens > self._max_input_tokens:
            raise InputTokenLimitExceededError(
                "serialized prompt exceeds max_input_tokens: "
                f"{input_tokens} > {self._max_input_tokens}"
            )

        _synchronize_cuda(self._torch, self._device)
        started_at = time.perf_counter()
        with self._torch.inference_mode():
            output_ids = self._model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                **self._generation_kwargs,
            )
        _synchronize_cuda(self._torch, self._device)
        elapsed_seconds = time.perf_counter() - started_at
        new_token_ids = output_ids[0][input_tokens:]
        generated_tokens = int(new_token_ids.shape[-1])
        self._record_generation_metrics(
            GenerationMetrics(
                input_tokens=input_tokens,
                generated_tokens=generated_tokens,
                elapsed_seconds=elapsed_seconds,
            )
        )
        return self._tokenizer.decode(
            new_token_ids,
            skip_special_tokens=True,
        ).strip()

    def choose_exact(self, prompt: str, *, choices: tuple[str, ...]) -> str:
        """Greedily generate only one token-trie-constrained exact choice."""
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        trie = ExactChoiceTokenTrie.from_tokenizer(self._tokenizer, choices)
        model_inputs = self._prompt_tokenizer.model_inputs(prompt)
        input_ids = model_inputs["input_ids"]
        input_tokens = int(input_ids.shape[-1])
        if input_tokens > self._max_input_tokens:
            raise InputTokenLimitExceededError(
                "serialized prompt exceeds max_input_tokens: "
                f"{input_tokens} > {self._max_input_tokens}"
            )

        def allowed_tokens(_batch_id: int, current_ids: Any) -> list[int]:
            current = token_ids(current_ids)
            generated = current[input_tokens:]
            return list(trie.allowed_tokens(generated))

        _synchronize_cuda(self._torch, self._device)
        started_at = time.perf_counter()
        with self._torch.inference_mode():
            output_ids = self._model.generate(
                **model_inputs,
                max_new_tokens=trie.max_sequence_length + 1,
                do_sample=False,
                prefix_allowed_tokens_fn=allowed_tokens,
                eos_token_id=trie.eos_token_id,
                pad_token_id=trie.eos_token_id,
            )
        _synchronize_cuda(self._torch, self._device)
        elapsed_seconds = time.perf_counter() - started_at
        generated_ids = token_ids(output_ids[0][input_tokens:])
        self._record_generation_metrics(
            GenerationMetrics(
                input_tokens=input_tokens,
                generated_tokens=len(generated_ids),
                elapsed_seconds=elapsed_seconds,
            )
        )
        return trie.resolve(generated_ids)

    def _record_generation_metrics(self, metrics: GenerationMetrics) -> None:
        self._history.append(metrics)
        self._generation_count += 1
        self._trim_generation_history()

    def _trim_generation_history(self) -> None:
        if self._history_limit is None:
            return
        overflow = len(self._history) - self._history_limit
        if overflow > 0:
            del self._history[:overflow]
            self._history_base += overflow


def resolve_device(device: str, *, torch_module: Any | None = None) -> str:
    """Resolve auto to one CUDA device when available, otherwise CPU."""
    if not isinstance(device, str) or not device.strip():
        raise ValueError("device must be a non-empty string")
    normalized = device.strip().lower()
    if normalized != "auto":
        return normalized
    torch = torch_module or _import_torch()
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_torch_dtype(
    dtype: str | Any,
    device: str,
    *,
    torch_module: Any | None = None,
) -> Any:
    """Resolve an explicit or hardware-aware dtype without importing at module load."""
    torch = torch_module or _import_torch()
    if not isinstance(dtype, str):
        return dtype
    normalized = dtype.strip().lower()
    if normalized == "auto":
        if device.startswith("cuda"):
            is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", None)
            if callable(is_bf16_supported) and is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        return torch.float32
    choices = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    try:
        return choices[normalized]
    except KeyError as error:
        raise ValueError(
            "dtype must be one of: auto, bfloat16, float16, float32"
        ) from error


def _validate_model_settings(model_name: str, max_input_tokens: int) -> None:
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must not be blank")
    if (
        not isinstance(max_input_tokens, int)
        or isinstance(max_input_tokens, bool)
        or max_input_tokens <= 0
    ):
        raise ValueError("max_input_tokens must be a positive integer")


def _validated_generation_kwargs(
    values: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Reject options that could replace already measured prompt tensors."""
    kwargs = dict(values or {"do_sample": False})
    forbidden = {
        "input_ids",
        "inputs",
        "attention_mask",
        "max_new_tokens",
        "max_length",
    }
    overlap = sorted(forbidden.intersection(kwargs))
    if overlap:
        raise ValueError(
            "generation kwargs may not replace measured inputs or limits: "
            + ", ".join(overlap)
        )
    if not isinstance(kwargs.get("do_sample"), bool):
        raise ValueError("generation kwargs require a boolean do_sample")
    return kwargs


def _synchronize_cuda(torch: Any, device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _import_torch() -> Any:
    return importlib.import_module("torch")


def _import_transformers() -> Any:
    return importlib.import_module("transformers")
