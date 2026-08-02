import pytest

from python_doc_rag.generator_models import (
    GENERATOR_MODEL_SPECS,
    generator_model_spec,
)


def test_generator_tournament_has_baseline_and_exactly_two_candidates() -> None:
    assert [spec.key for spec in GENERATOR_MODEL_SPECS] == [
        "baseline-qwen3-4b",
        "qwen3-8b",
        "qwen2.5-7b-instruct",
    ]
    for spec in GENERATOR_MODEL_SPECS:
        assert len(spec.revision) == 40
        assert spec.license == "Apache-2.0"
        assert spec.dtype == "bfloat16"
        assert spec.trust_remote_code is False


def test_qwen3_8b_uses_official_non_thinking_switch_and_sampling() -> None:
    spec = generator_model_spec("qwen3-8b")

    assert spec.parameter_billions == 8.2
    assert spec.template_options() == {"enable_thinking": False}
    assert spec.generation_options() == {
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
    }
    assert spec.sampling_seed == 20260731


def test_qwen25_candidate_keeps_structured_output_sampling_config() -> None:
    spec = generator_model_spec("qwen2.5-7b-instruct")

    assert spec.generation_options()["repetition_penalty"] == 1.05
    assert spec.template_options() == {}
    assert "Japanese" in spec.language_evidence


def test_baseline_generator_semantics_remain_greedy_and_pinned() -> None:
    spec = generator_model_spec("baseline-qwen3-4b")

    assert spec.model_name == "Qwen/Qwen3-4B-Instruct-2507"
    assert spec.revision == "cdbee75f17c01a7cc42f958dc650907174af0554"
    assert spec.generation_options() == {"do_sample": False}
    assert spec.template_options() == {}


def test_unknown_generator_candidate_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        generator_model_spec("remote-model")
