import importlib.util
from argparse import Namespace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def _load_runner() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts/run_rag_quality_evaluation.py"
    )
    spec = importlib.util.spec_from_file_location("rag_evaluation_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args(tmp_path: Path, *, retriever: str = "dense") -> Namespace:
    return Namespace(
        data_root=tmp_path,
        questions=tmp_path / "questions.jsonl",
        dataset_type="rag-quality",
        input_results=None,
        output_path=tmp_path / "output.json",
        index_path=None,
        metadata_path=None,
        index_manifest_path=None,
        embedding_model=None,
        model_name="fake-qwen",
        revision=None,
        retriever=retriever,
        device="cpu",
        dtype="float32",
        top_k=5,
        max_input_tokens=1024,
        max_new_tokens=32,
        judge="none",
        judge_model=None,
        judge_timeout=30.0,
        judge_max_retries=2,
    )


def test_main_builds_runtime_once_and_reuses_pipeline(
    monkeypatch,
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    args = _args(tmp_path)
    pipeline = object()
    trace = object()
    build_calls: list[tuple[Namespace, Path]] = []
    evaluate_calls: list[tuple[object, object]] = []

    monkeypatch.setattr(runner, "parse_args", lambda: args)
    monkeypatch.setattr(
        runner,
        "load_rag_quality_questions",
        lambda path: [SimpleNamespace(id="q1")],
    )

    def build_runtime(received: Namespace, root: Path):
        build_calls.append((received, root))
        return pipeline, trace, {"runtime": "fake"}

    def evaluate(received_pipeline, received_trace, questions, **kwargs):
        del questions, kwargs
        evaluate_calls.append((received_pipeline, received_trace))
        return []

    monkeypatch.setattr(runner, "build_runtime", build_runtime)
    monkeypatch.setattr(runner, "evaluate_rag_questions", evaluate)
    monkeypatch.setattr(runner, "save_rag_evaluation", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "_print_summary", lambda path: None)

    assert runner.main() == 0
    assert build_calls == [(args, tmp_path)]
    assert evaluate_calls == [(pipeline, trace)]


def test_hybrid_settings_are_fixed_and_dense_has_no_rrf(tmp_path: Path) -> None:
    runner = _load_runner()

    dense = runner._settings(_args(tmp_path), judge_model=None)
    hybrid = runner._settings(
        _args(tmp_path, retriever="hybrid"),
        judge_model=None,
    )

    assert "rrf_k" not in dense
    assert dense["judge_prompt_revision"] == "rag-grounding-v3"
    assert dense["judge_schema_revision"] == "rag-judge-evidence-v3"
    assert dense["derived_result_revision"] == "rag-derived-evaluation-v1"
    assert hybrid["japanese_ngram_sizes"] == [2]
    assert hybrid["rrf_k"] == 10
    assert hybrid["candidate_k"] == 30
    assert hybrid["retriever_weights"] == [1.0, 1.0]


def test_saved_result_judging_skips_qwen_runtime(
    monkeypatch,
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    args = _args(tmp_path)
    args.input_results = tmp_path / "saved.json"
    args.judge = "openai"
    args.judge_model = "fake-judge"
    fake_judge = SimpleNamespace(model_name="fake-judge")

    monkeypatch.setattr(runner, "parse_args", lambda: args)
    monkeypatch.setattr(runner, "_build_judge", lambda received: fake_judge)
    monkeypatch.setattr(runner, "load_rag_records", lambda path: [])
    monkeypatch.setattr(runner, "apply_judge_to_records", lambda *args, **kwargs: [])
    monkeypatch.setattr(runner, "save_rag_evaluation", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "_print_summary", lambda path: None)

    def fail_build(*args, **kwargs):
        raise AssertionError("Qwen runtime must not be built for saved-result judging")

    monkeypatch.setattr(runner, "build_runtime", fail_build)

    assert runner.main() == 0


def test_saved_result_cannot_overwrite_its_input(
    monkeypatch,
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    args = _args(tmp_path)
    args.input_results = args.output_path
    args.judge = "openai"
    args.judge_model = "fake-judge"
    monkeypatch.setattr(runner, "parse_args", lambda: args)

    def fail_build(*args, **kwargs):
        raise AssertionError("judge client must not be built for an unsafe path")

    monkeypatch.setattr(runner, "_build_judge", fail_build)

    with pytest.raises(ValueError, match="must differ"):
        runner.main()


def test_answer_mode_is_saved_with_contract_revision(tmp_path: Path) -> None:
    runner = _load_runner()
    args = _args(tmp_path)
    args.answer_mode = "answer-or-abstain"
    settings = runner._settings(args, judge_model=None)
    assert settings["answer_mode"] == "answer-or-abstain"
    assert settings["contract_revision"] == "answer-or-abstain-v1"


def test_reranker_settings_are_opt_in_and_pinned(tmp_path: Path) -> None:
    runner = _load_runner()
    args = _args(tmp_path)
    baseline = runner._settings(args, judge_model=None)
    args.reranker_model_key = "mmarco-minilm"
    args.reranker_candidate_k = 30
    args.reranker_batch_size = 16
    args.reranker_max_length = 512

    selected = runner._settings(args, judge_model=None)

    assert baseline["reranking_revision"] is None
    assert selected["reranking_revision"] == "local-cross-encoder-rerank-v1"
    assert selected["reranker_model_key"] == "mmarco-minilm"
    assert selected["reranker_candidate_k"] == 30


def test_section_parent_settings_are_research_only_and_explicit(tmp_path: Path) -> None:
    runner = _load_runner()
    args = _args(tmp_path)
    args.context_mode = "section-parent"
    args.child_candidate_k = 30

    settings = runner._settings(args, judge_model=None)

    assert settings["context_mode"] == "section-parent"
    assert settings["section_parent_revision"] == "section-parent-v1"
    assert settings["child_candidate_k"] == 30
