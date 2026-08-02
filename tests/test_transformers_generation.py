import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest

import python_doc_rag.transformers_generation as generation_module
from python_doc_rag.constrained_generation import ExactChoiceGenerationError
from python_doc_rag.transformers_generation import (
    InputTokenLimitExceededError,
    TransformersAnswerGenerator,
    resolve_device,
    resolve_torch_dtype,
)


class FakeVector:
    """One-dimensional tensor-like test value."""

    def __init__(self, values: list[int]) -> None:
        self.values = values

    @property
    def shape(self) -> tuple[int]:
        return (len(self.values),)

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, item: slice) -> "FakeVector":
        return FakeVector(self.values[item])


class FakeTensor:
    """Two-dimensional tensor-like test value that records device movement."""

    def __init__(self, rows: list[list[int]]) -> None:
        self.rows = rows
        self.devices: list[str] = []

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.rows), len(self.rows[0]))

    def to(self, device: str) -> "FakeTensor":
        self.devices.append(device)
        return self

    def __getitem__(self, item: int) -> FakeVector:
        return FakeVector(self.rows[item])


class FakeTokenizer:
    """Capture exact tokenization and decoding calls."""

    def __init__(self, input_tokens: list[int] | None = None) -> None:
        self.input_ids = FakeTensor([input_tokens or [10, 11, 12]])
        self.attention_mask = FakeTensor([[1] * self.input_ids.shape[-1]])
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.decode_calls: list[tuple[list[int], bool]] = []
        self.eos_token_id = 99

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return {"answer": [90], "abstain": [91, 92]}[text]

    def __call__(self, prompt: str, **kwargs: Any) -> dict[str, FakeTensor]:
        self.calls.append((prompt, kwargs))
        return {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
        }

    def decode(self, token_ids: FakeVector, *, skip_special_tokens: bool) -> str:
        self.decode_calls.append((token_ids.values, skip_special_tokens))
        return "  生成回答[S1]  "


class FakeModel:
    """Append two tokens and capture the exact generate inputs."""

    def __init__(self) -> None:
        self.to_calls: list[dict[str, Any]] = []
        self.generate_calls: list[dict[str, Any]] = []
        self.eval_called = False
        self.config = SimpleNamespace(_commit_hash="resolved-revision")

    def to(self, **kwargs: Any) -> "FakeModel":
        self.to_calls.append(kwargs)
        return self

    def eval(self) -> None:
        self.eval_called = True

    def generate(self, **kwargs: Any) -> FakeTensor:
        self.generate_calls.append(kwargs)
        input_ids = kwargs["input_ids"]
        return FakeTensor([[*input_ids.rows[0], 90, 91]])


class ExactChoiceModel(FakeModel):
    """Follow the prefix callback to choose the longer abstain sequence."""

    def generate(self, **kwargs: Any) -> FakeTensor:
        self.generate_calls.append(kwargs)
        input_ids = kwargs["input_ids"]
        prefix = kwargs["prefix_allowed_tokens_fn"]
        current = list(input_ids.rows[0])
        assert prefix(0, FakeVector(current)) == [90, 91]
        current.append(91)
        assert prefix(0, FakeVector(current)) == [92]
        current.append(92)
        assert prefix(0, FakeVector(current)) == [99]
        current.append(99)
        return FakeTensor([current])


class InvalidChoiceModel(FakeModel):
    """Simulate a backend violating the prefix callback contract."""

    def generate(self, **kwargs: Any) -> FakeTensor:
        self.generate_calls.append(kwargs)
        input_ids = kwargs["input_ids"]
        return FakeTensor([[*input_ids.rows[0], 55]])


class FakeInferenceMode:
    """Record inference-mode entry and exit."""

    def __init__(self, torch: "FakeTorch") -> None:
        self._torch = torch

    def __enter__(self) -> None:
        self._torch.inference_entries += 1

    def __exit__(self, *args: object) -> None:
        self._torch.inference_exits += 1


class FakeCuda:
    """Configurable CUDA capability surface."""

    def __init__(self, *, available: bool = True, bf16: bool = True) -> None:
        self._available = available
        self._bf16 = bf16
        self.synchronize_calls = 0

    def is_available(self) -> bool:
        return self._available

    def is_bf16_supported(self) -> bool:
        return self._bf16

    def synchronize(self) -> None:
        self.synchronize_calls += 1


class FakeTorch:
    """Minimal torch surface used by the concrete generator."""

    bfloat16 = "torch.bfloat16"
    float16 = "torch.float16"
    float32 = "torch.float32"

    def __init__(self, *, available: bool = True, bf16: bool = True) -> None:
        self.cuda = FakeCuda(available=available, bf16=bf16)
        self.inference_entries = 0
        self.inference_exits = 0
        self.manual_seeds: list[int] = []

    def inference_mode(self) -> FakeInferenceMode:
        return FakeInferenceMode(self)

    def manual_seed(self, seed: int) -> None:
        self.manual_seeds.append(seed)


def make_generator(
    monkeypatch: pytest.MonkeyPatch,
    *,
    tokenizer: FakeTokenizer | None = None,
    model: FakeModel | None = None,
    max_input_tokens: int = 32,
) -> tuple[TransformersAnswerGenerator, FakeTokenizer, FakeModel, FakeTorch]:
    fake_torch = FakeTorch()
    fake_tokenizer = tokenizer or FakeTokenizer()
    fake_model = model or FakeModel()
    monkeypatch.setattr(generation_module, "_import_torch", lambda: fake_torch)
    generator = TransformersAnswerGenerator(
        fake_tokenizer,
        fake_model,
        model_name="test/model",
        revision="requested-revision",
        device="cuda",
        dtype="auto",
        max_input_tokens=max_input_tokens,
    )
    return generator, fake_tokenizer, fake_model, fake_torch


def test_generate_uses_measured_ids_and_decodes_only_new_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator, tokenizer, model, torch = make_generator(monkeypatch)
    measured_ids = generator.prompt_tokenizer.encode(
        "<chat>prepared prompt",
        add_special_tokens=False,
        truncation=False,
    )

    answer = generator.generate("<chat>prepared prompt", max_new_tokens=17)

    assert answer == "生成回答[S1]"
    assert len(measured_ids) == 3
    assert tokenizer.calls == [
        (
            "<chat>prepared prompt",
            {
                "return_tensors": "pt",
                "add_special_tokens": False,
                "truncation": False,
            },
        )
    ]
    generate_call = model.generate_calls[0]
    assert generate_call["input_ids"] is tokenizer.input_ids
    assert generate_call["attention_mask"] is tokenizer.attention_mask
    assert generate_call["max_new_tokens"] == 17
    assert generate_call["do_sample"] is False
    assert tokenizer.decode_calls == [([90, 91], True)]
    assert model.eval_called
    assert model.to_calls == [{"device": "cuda", "dtype": FakeTorch.bfloat16}]
    assert torch.inference_entries == 1
    assert torch.inference_exits == 1
    assert generator.generation_history[0].input_tokens == 3
    assert generator.generation_history[0].generated_tokens == 2
    assert generator.model_revision == "resolved-revision"
    assert generator.dtype_name == "bfloat16"


def test_input_limit_fails_before_model_generate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator, _, model, _ = make_generator(
        monkeypatch,
        tokenizer=FakeTokenizer([1, 2, 3]),
        max_input_tokens=2,
    )

    with pytest.raises(InputTokenLimitExceededError, match="3 > 2"):
        generator.generate("prepared", max_new_tokens=5)

    assert not model.generate_calls
    assert not generator.generation_history


def test_exact_choice_uses_greedy_prefix_constraint_and_records_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator, _, model, _ = make_generator(
        monkeypatch,
        model=ExactChoiceModel(),
    )

    choice = generator.choose_exact(
        "prepared choice prompt",
        choices=("answer", "abstain"),
    )

    assert choice == "abstain"
    call = model.generate_calls[0]
    assert call["do_sample"] is False
    assert call["max_new_tokens"] == 3
    assert call["eos_token_id"] == 99
    assert call["pad_token_id"] == 99
    assert callable(call["prefix_allowed_tokens_fn"])
    assert generator.generation_history[0].generated_tokens == 3


def test_generation_cursor_supports_bounded_server_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator, _, _, _ = make_generator(monkeypatch)
    generator.set_generation_history_limit(2)

    first_cursor = generator.generation_cursor
    generator.generate("first", max_new_tokens=2)
    first_metrics = generator.generation_metrics_since(first_cursor)
    generator.generate("second", max_new_tokens=2)
    generator.generate("third", max_new_tokens=2)

    assert len(first_metrics) == 1
    assert generator.generation_cursor == 3
    assert len(generator.generation_history) == 2
    assert len(generator.generation_metrics_since(1)) == 2
    with pytest.raises(ValueError, match="outside retained history"):
        generator.generation_metrics_since(0)


def test_exact_choice_rejects_backend_output_outside_allowed_sequences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator, _, _, _ = make_generator(
        monkeypatch,
        model=InvalidChoiceModel(),
    )

    with pytest.raises(ExactChoiceGenerationError, match="exact choice"):
        generator.choose_exact(
            "prepared choice prompt",
            choices=("answer", "abstain"),
        )


def test_explicit_sampling_settings_and_seed_reach_generate_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = FakeTorch()
    tokenizer = FakeTokenizer()
    model = FakeModel()
    monkeypatch.setattr(generation_module, "_import_torch", lambda: fake_torch)
    generator = TransformersAnswerGenerator(
        tokenizer,
        model,
        model_name="test/sampling-model",
        device="cuda",
        dtype="bfloat16",
        generation_kwargs={
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.8,
        },
        sampling_seed=42,
    )

    generator.generate("prepared", max_new_tokens=9)

    assert fake_torch.manual_seeds == [42]
    assert generator.sampling_seed == 42
    assert generator.generation_kwargs == {
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.8,
    }
    assert model.generate_calls[0]["do_sample"] is True
    assert model.generate_calls[0]["temperature"] == 0.7
    assert model.generate_calls[0]["top_p"] == 0.8


@pytest.mark.parametrize(
    "option", ["input_ids", "attention_mask", "max_new_tokens", "max_length"]
)
def test_generation_settings_cannot_replace_safe_inputs_or_limits(
    option: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = FakeTorch()
    monkeypatch.setattr(generation_module, "_import_torch", lambda: fake_torch)

    with pytest.raises(ValueError, match="may not replace"):
        TransformersAnswerGenerator(
            FakeTokenizer(),
            FakeModel(),
            model_name="test/model",
            device="cpu",
            dtype="float32",
            generation_kwargs={"do_sample": False, option: 1},
        )


def test_injected_objects_do_not_import_transformers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = FakeTorch()
    monkeypatch.setattr(generation_module, "_import_torch", lambda: fake_torch)
    monkeypatch.setattr(
        generation_module,
        "_import_transformers",
        lambda: pytest.fail("injected construction must not import transformers"),
    )

    generator = TransformersAnswerGenerator(
        FakeTokenizer(),
        FakeModel(),
        model_name="test/model",
        device="cpu",
        dtype="float32",
    )

    assert generator.device == "cpu"


@pytest.mark.parametrize(
    ("device", "available", "bf16", "requested", "expected"),
    [
        ("cuda", True, True, "auto", FakeTorch.bfloat16),
        ("cuda", True, False, "auto", FakeTorch.float16),
        ("cpu", False, False, "auto", FakeTorch.float32),
        ("cuda", True, True, "float16", FakeTorch.float16),
        ("cpu", False, False, "bfloat16", FakeTorch.bfloat16),
    ],
)
def test_dtype_selection(
    device: str,
    available: bool,
    bf16: bool,
    requested: str,
    expected: str,
) -> None:
    torch = FakeTorch(available=available, bf16=bf16)

    assert (
        resolve_torch_dtype(
            requested,
            device,
            torch_module=torch,
        )
        == expected
    )


def test_auto_device_selection() -> None:
    assert resolve_device("auto", torch_module=FakeTorch()) == "cuda"
    assert (
        resolve_device(
            "auto",
            torch_module=FakeTorch(available=False),
        )
        == "cpu"
    )


def test_package_import_does_not_require_torch_or_transformers() -> None:
    code = """
import builtins

original_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name.split('.', maxsplit=1)[0] in {'torch', 'transformers'}:
        raise ModuleNotFoundError(name)
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
import python_doc_rag
assert hasattr(python_doc_rag, 'TransformersAnswerGenerator')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
