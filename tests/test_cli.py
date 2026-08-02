import builtins
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import python_doc_rag.cli as cli
from python_doc_rag.application import AnswerServiceError, ArtifactValidationError
from python_doc_rag.cli import (
    AnswerExecution,
    CliError,
    _RagService,
    _TimingRetriever,
    build_parser,
    inspect_data_root,
    main,
    resolve_data_root,
)
from python_doc_rag.models import CitationSource, CitedAnswer, SearchChunk


def _chunk(number: int) -> SearchChunk:
    return SearchChunk(
        text=f"本文{number}",
        page_title=f"ページ{number}",
        section_title=f"節{number}",
        source_url=f"https://docs.python.org/ja/3.13/tutorial/{number}.html",
        category="tutorial",
        chunk_index=number,
        start_index=number * 10,
    )


def _answer(question: str) -> CitedAnswer:
    chunk = _chunk(1)
    return CitedAnswer(
        answer_text=f"{question}への回答[S1]",
        sources=(
            CitationSource(
                label="S1",
                page_title=chunk.page_title,
                section_title=chunk.section_title,
                url=chunk.source_url,
            ),
        ),
        retrieved_chunks=(chunk,),
        generation_attempts=1,
    )


class FakeService:
    """Return deterministic cited answers without loading models."""

    def __init__(self) -> None:
        self.questions: list[str] = []

    def answer(self, question: str) -> AnswerExecution:
        self.questions.append(question)
        return AnswerExecution(
            answer=_answer(question),
            retrieval_seconds=0.125,
            generation_seconds=0.5,
            total_seconds=0.75,
        )


class FakeRetriever:
    """Record lightweight retrieval calls made by a fake pipeline."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, question: str, *, limit: int) -> tuple[SearchChunk, ...]:
        self.calls.append((question, limit))
        return (_chunk(1),)


class FakeGenerator:
    """Expose the generation-history surface consumed by the CLI service."""

    def __init__(self) -> None:
        self.generation_history: list[SimpleNamespace] = []


class FakePipeline:
    """Exercise a fake retriever and generator without a model."""

    def __init__(
        self,
        retriever: _TimingRetriever,
        generator: FakeGenerator,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.questions: list[str] = []

    def answer(self, question: str) -> CitedAnswer:
        self.questions.append(question)
        self.retriever.retrieve(question, limit=5)
        self.generator.generation_history.append(
            SimpleNamespace(elapsed_seconds=0.25)
        )
        return _answer(question)


class FakeCuda:
    """Represent a host without CUDA for preflight tests."""

    @staticmethod
    def is_available() -> bool:
        return False


class FakeTorch:
    """Minimal torch module needed before model imports."""

    cuda = FakeCuda()


class FakeFaiss:
    """Read a synthetic FAISS header without requiring faiss-cpu."""

    def __init__(self, count: int, dimension: int) -> None:
        self._count = count
        self._dimension = dimension

    def read_index(self, path: str) -> SimpleNamespace:
        assert Path(path).is_file()
        return SimpleNamespace(ntotal=self._count, d=self._dimension)


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    processed = tmp_path / "data/processed/python_3_13_ja_chunks.jsonl"
    index = tmp_path / "indexes/python_3_13_ja.faiss"
    metadata = tmp_path / "indexes/python_3_13_ja_metadata.jsonl"
    manifest = tmp_path / "indexes/python_3_13_ja_index_manifest.json"
    processed.parent.mkdir(parents=True)
    index.parent.mkdir(parents=True)
    lines = "".join(
        f"{json.dumps(_chunk(number).to_dict(), ensure_ascii=False)}\n"
        for number in (1, 2)
    )
    processed.write_text(lines, encoding="utf-8")
    metadata.write_text(lines, encoding="utf-8")
    index.write_bytes(b"fake-faiss-index")
    manifest.write_text(
        json.dumps(
            {
                "model_name": "fixture-embedding-model",
                "embedding_dimension": 384,
                "chunk_count": 2,
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


@pytest.mark.parametrize(
    "command",
    [None, "ask", "chat", "check", "profile", "prepare", "serve"],
)
def test_help_commands(
    command: str | None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    arguments = ["--help"] if command is None else [command, "--help"]

    with pytest.raises(SystemExit) as error:
        main(arguments)

    assert error.value.code == 0
    output = capsys.readouterr().out
    assert "usage:" in output
    if command is None:
        assert "{ask,chat,check,profile,prepare,serve}" in output
    else:
        assert command in output


def test_prepare_source_mode_flags_have_safe_defaults() -> None:
    args = build_parser().parse_args(
        ["prepare", "--site-config", "configs/sites/python-docs.toml"]
    )

    assert args.source_root is None
    assert not args.offline
    assert not args.refresh
    assert not args.rebuild
    assert not args.resume


def test_prepare_cli_forwards_source_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import python_doc_rag.preparation as preparation

    captured: dict[str, Any] = {}

    def fake_prepare(
        site_config: Path,
        selected_data_root: Path,
        **kwargs: Any,
    ) -> SimpleNamespace:
        captured.update(
            {
                "site_config": site_config,
                "data_root": selected_data_root,
                **kwargs,
            }
        )
        return SimpleNamespace(
            dataset_manifest=SimpleNamespace(dataset_name="Fixture docs"),
            data_root=selected_data_root,
            source_page_count=1,
            parsed_page_count=1,
            failed_page_count=0,
            section_count=1,
            chunk_count=1,
            corpus_sha256="a" * 64,
            reused_dataset=False,
            source_cache_reused=True,
            source_snapshot_sha256="b" * 64,
            dense_index_result=None,
            profile_result=None,
        )

    monkeypatch.setattr(preparation, "prepare_dataset", fake_prepare)
    config = tmp_path / "site.toml"
    source = tmp_path / "source"
    selected_data_root = tmp_path / "data-root"

    assert (
        main(
            [
                "prepare",
                "--site-config",
                str(config),
                "--data-root",
                str(selected_data_root),
                "--source-root",
                str(source),
                "--offline",
                "--rebuild",
                "--resume",
                "--until",
                "corpus",
                "--device",
                "cpu",
            ]
        )
        == 0
    )

    assert captured == {
        "site_config": config,
        "data_root": selected_data_root.resolve(),
        "source_root": source,
        "offline": True,
        "refresh": False,
        "rebuild": True,
        "resume": True,
        "until": "corpus",
        "device": "cpu",
    }
    assert "dataset準備: 成功" in capsys.readouterr().out


@pytest.mark.parametrize("command", ["ask", "chat"])
def test_generation_commands_default_to_dense_retrieval(command: str) -> None:
    args = build_parser().parse_args([command])

    assert args.retriever == "dense"


@pytest.mark.parametrize("command", ["ask", "chat"])
def test_generation_commands_accept_hybrid_retrieval(command: str) -> None:
    args = build_parser().parse_args([command, "--retriever", "hybrid"])

    assert args.retriever == "hybrid"


@pytest.mark.parametrize("command", ["ask", "chat"])
def test_versioned_recommended_profiles_preserve_default(command: str) -> None:
    default_args = build_parser().parse_args([command])
    recommended_v1 = build_parser().parse_args(
        [command, "--profile", "recommended-v1"]
    )
    recommended_v2 = build_parser().parse_args(
        [command, "--profile", "recommended-v2"]
    )
    recommended = build_parser().parse_args([command, "--profile", "recommended"])

    cli._apply_runtime_profile(default_args)
    cli._apply_runtime_profile(recommended_v1)
    cli._apply_runtime_profile(recommended_v2)
    cli._apply_runtime_profile(recommended)

    assert default_args.retriever == "dense"
    assert default_args.answer_mode == "legacy"
    assert default_args.reranker_model_key is None
    assert recommended_v1.retriever == "hybrid"
    assert recommended_v1.model_revision == (
        "cdbee75f17c01a7cc42f958dc650907174af0554"
    )
    assert recommended_v2.retriever == "technical-field"
    assert recommended_v2.embedding_model_key == "bge-m3"
    assert recommended_v2.field_candidate_k == 30
    assert recommended.answer_mode == "answer-or-abstain"
    assert recommended.reranker_model_key == "mmarco-minilm"
    assert recommended.reranker_candidate_k == 30
    assert recommended.model_revision == (
        "b968826d9c46dd6066d109eabc6255188de91218"
    )
    assert recommended.generator_model_key == "qwen3-8b"


def test_profile_command_prints_pinned_recommended_contents(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["profile", "recommended"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["alias_of"] == "recommended-v2"
    assert payload["revision"] == "recommended-v2"
    assert payload["retriever"] == "technical-field"
    assert payload["answer_mode"] == "answer-or-abstain"
    assert payload["embedding_model"] == "BAAI/bge-m3"
    assert payload["embedding_model_revision"] == (
        "5617a9f61b028005a4858fdac845db406aefb181"
    )
    assert payload["generation_model"] == "Qwen/Qwen3-8B"
    assert payload["generation_template_options"] == {"enable_thinking": False}
    assert payload["reranker_model"] == (
        "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    )
    assert payload["reranker_model_revision"] == (
        "1427fd652930e4ba29e8149678df786c240d8825"
    )
    assert payload["reranker_trust_remote_code"] is False


def test_data_root_argument_has_priority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment_root = tmp_path / "environment"
    argument_root = tmp_path / "argument"
    monkeypatch.setenv("PYTHON_DOC_RAG_DATA_ROOT", str(environment_root))

    assert resolve_data_root(argument_root) == argument_root


def test_data_root_uses_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTHON_DOC_RAG_DATA_ROOT", str(tmp_path))

    assert resolve_data_root(None) == tmp_path


def test_data_root_missing_is_an_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PYTHON_DOC_RAG_DATA_ROOT", raising=False)

    with pytest.raises(CliError, match="--data-root"):
        resolve_data_root(None)


def test_ask_accepts_question_argument_and_prints_answer(
    data_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    service = FakeService()
    monkeypatch.setattr(cli, "_build_service", lambda args, root: service)

    exit_code = main(
        [
            "ask",
            "--data-root",
            str(data_root),
            "--question",
            "list.sort()とは？",
        ]
    )

    assert exit_code == 0
    assert service.questions == ["list.sort()とは？"]
    output = capsys.readouterr().out
    assert "回答\n----" in output
    assert "list.sort()とは？への回答[S1]" in output
    assert "[S1] ページ1" in output
    assert "節1" in output
    assert "https://docs.python.org/ja/3.13/tutorial/1.html" in output
    assert "検索: 0.125秒" in output
    assert "生成: 0.500秒" in output
    assert "全体: 0.750秒" in output


def test_ask_accepts_standard_input(
    data_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = FakeService()
    monkeypatch.setattr(cli, "_build_service", lambda args, root: service)
    monkeypatch.setattr(builtins, "input", lambda prompt: "標準入力の質問")

    exit_code = main(["ask", "--data-root", str(data_root)])

    assert exit_code == 0
    assert service.questions == ["標準入力の質問"]


def test_ask_rejects_blank_question_before_loading_models(
    data_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        cli,
        "_build_service",
        lambda args, root: pytest.fail("models must not load for a blank question"),
    )

    exit_code = main(
        ["ask", "--data-root", str(data_root), "--question", " \n "]
    )

    assert exit_code == 1
    assert "質問を空にすることはできません" in capsys.readouterr().err


def test_missing_data_root_returns_nonzero(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("PYTHON_DOC_RAG_DATA_ROOT", raising=False)

    exit_code = main(["ask", "--question", "質問"])

    assert exit_code == 1
    assert "PYTHON_DOC_RAG_DATA_ROOT" in capsys.readouterr().err


def test_chat_answers_multiple_questions_with_one_service(
    data_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = FakeService()
    factory_calls: list[Path] = []

    def fake_factory(args: Any, root: Path) -> FakeService:
        factory_calls.append(root)
        return service

    inputs = iter(["質問1", "", "質問2", "exit"])
    monkeypatch.setattr(cli, "_build_service", fake_factory)
    monkeypatch.setattr(builtins, "input", lambda prompt: next(inputs))

    exit_code = main(["chat", "--data-root", str(data_root)])

    assert exit_code == 0
    assert factory_calls == [data_root]
    assert service.questions == ["質問1", "質問2"]


def test_chat_recovers_from_cli_error_and_answers_next_question(
    data_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    service = FakeService()
    original_answer = service.answer
    factory_calls: list[Path] = []

    def fake_factory(args: Any, root: Path) -> FakeService:
        factory_calls.append(root)
        return service

    def answer(question: str) -> AnswerExecution:
        if not service.questions:
            service.questions.append(question)
            raise CliError("回答の引用検証に2回失敗しました。")
        return original_answer(question)

    inputs = iter(["失敗する質問", "成功する質問", "exit"])
    monkeypatch.setattr(service, "answer", answer)
    monkeypatch.setattr(cli, "_build_service", fake_factory)
    monkeypatch.setattr(builtins, "input", lambda prompt: next(inputs))

    exit_code = main(["chat", "--data-root", str(data_root)])

    assert exit_code == 0
    assert factory_calls == [data_root]
    assert service.questions == ["失敗する質問", "成功する質問"]
    captured = capsys.readouterr()
    assert "回答できませんでした" in captured.err
    assert "回答の引用検証に2回失敗しました。" in captured.err
    assert "質問を言い換えるか、別の質問を入力してください。" in captured.err
    assert "成功する質問への回答[S1]" in captured.out
    assert "失敗する質問への回答" not in captured.out


def test_ask_cli_error_remains_nonzero(
    data_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    service = FakeService()
    monkeypatch.setattr(
        service,
        "answer",
        lambda question: (_ for _ in ()).throw(CliError("回答生成失敗")),
    )
    monkeypatch.setattr(cli, "_build_service", lambda args, root: service)

    exit_code = main(
        [
            "ask",
            "--data-root",
            str(data_root),
            "--question",
            "失敗する質問",
        ]
    )

    assert exit_code == 1
    assert "エラー: 回答生成失敗" in capsys.readouterr().err


@pytest.mark.parametrize("ending", ["exit", "quit", "EOF"])
def test_chat_exit_quit_and_eof_are_normal(
    ending: str,
    data_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    service = FakeService()
    monkeypatch.setattr(cli, "_build_service", lambda args, root: service)
    if ending == "EOF":
        monkeypatch.setattr(
            builtins,
            "input",
            lambda prompt: (_ for _ in ()).throw(EOFError),
        )
    else:
        monkeypatch.setattr(builtins, "input", lambda prompt: ending)

    exit_code = main(["chat", "--data-root", str(data_root)])

    assert exit_code == 0
    assert not service.questions
    assert "終了します" in capsys.readouterr().out


@pytest.mark.parametrize("during_answer", [False, True])
def test_chat_ctrl_c_is_normal(
    during_answer: bool,
    data_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    service = FakeService()
    if during_answer:
        monkeypatch.setattr(
            service,
            "answer",
            lambda question: (_ for _ in ()).throw(KeyboardInterrupt),
        )
        monkeypatch.setattr(builtins, "input", lambda prompt: "質問")
    else:
        monkeypatch.setattr(
            builtins,
            "input",
            lambda prompt: (_ for _ in ()).throw(KeyboardInterrupt),
        )
    monkeypatch.setattr(cli, "_build_service", lambda args, root: service)

    exit_code = main(["chat", "--data-root", str(data_root)])

    assert exit_code == 0
    assert "終了します" in capsys.readouterr().out


def test_rag_service_uses_fake_pipeline_retriever_and_generator() -> None:
    retriever = FakeRetriever()
    timing_retriever = _TimingRetriever(retriever)
    generator = FakeGenerator()
    pipeline = FakePipeline(timing_retriever, generator)
    service = _RagService(
        pipeline=pipeline,
        timing_retriever=timing_retriever,
        generator=generator,
    )

    first = service.answer("質問1")
    second = service.answer("質問2")

    assert first.answer.answer_text == "質問1への回答[S1]"
    assert second.answer.answer_text == "質問2への回答[S1]"
    assert first.generation_seconds == 0.25
    assert second.generation_seconds == 0.25
    assert pipeline.questions == ["質問1", "質問2"]
    assert retriever.calls == [("質問1", 5), ("質問2", 5)]


def test_build_profile_service_preserves_public_wrapper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = FakeService()
    captured: dict[str, Any] = {}

    def build(
        data_root: Path,
        *,
        profile_name: str,
        device: str,
    ) -> FakeService:
        captured.update(
            data_root=data_root,
            profile_name=profile_name,
            device=device,
        )
        return service

    monkeypatch.setattr("python_doc_rag.application.build_profile_service", build)

    result = cli.build_profile_service(
        tmp_path,
        profile_name="recommended-v1",
        device="cpu",
    )

    assert result.answer("question") == service.answer("question")
    assert captured == {
        "data_root": tmp_path,
        "profile_name": "recommended-v1",
        "device": "cpu",
    }


def test_build_profile_service_preserves_cli_error_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = FakeService()
    monkeypatch.setattr(
        service,
        "answer",
        lambda question: (_ for _ in ()).throw(AnswerServiceError("failed")),
    )
    monkeypatch.setattr(
        "python_doc_rag.application.build_profile_service",
        lambda *args, **kwargs: service,
    )

    built = cli.build_profile_service(tmp_path)

    with pytest.raises(CliError, match="failed"):
        built.answer("question")


def test_build_profile_service_preserves_construction_error_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise ArtifactValidationError("invalid artifacts")

    monkeypatch.setattr("python_doc_rag.application.build_profile_service", fail)

    with pytest.raises(CliError, match="invalid artifacts"):
        cli.build_profile_service(tmp_path)


def test_check_success(
    data_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "python_doc_rag.vector_store._import_faiss",
        lambda: FakeFaiss(2, 384),
    )

    exit_code = main(["check", "--data-root", str(data_root)])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "データ検査: 成功" in output
    assert "チャンク数: 2" in output
    assert "index件数: 2" in output
    assert "次元: 384" in output
    assert "Embeddingモデル: fixture-embedding-model" in output
    assert "profile: default" in output


def test_check_recommended_v1_reuses_validated_baseline_artifacts(
    data_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "python_doc_rag.vector_store._import_faiss",
        lambda: FakeFaiss(2, 384),
    )

    exit_code = main(
        ["check", "--profile", "recommended-v1", "--data-root", str(data_root)]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "profile: recommended-v1" in output
    assert "baseline index/metadataを再利用" in output


def test_check_recommended_v2_reports_missing_sidecars(
    data_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "python_doc_rag.vector_store._import_faiss",
        lambda: FakeFaiss(2, 384),
    )

    exit_code = main(
        ["check", "--profile", "recommended-v2", "--data-root", str(data_root)]
    )

    assert exit_code == 1
    error = capsys.readouterr().err
    assert "profile artifactが見つかりません" in error
    assert "symbol_fields.jsonl" in error


def test_check_reports_each_missing_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tmp_path.mkdir(exist_ok=True)

    exit_code = main(["check", "--data-root", str(tmp_path)])

    assert exit_code == 1
    error = capsys.readouterr().err
    assert "processed JSONLが見つかりません" in error
    assert "FAISS indexが見つかりません" in error
    assert "metadata JSONLが見つかりません" in error
    assert "index manifestが見つかりません" in error


def test_check_reports_count_mismatch(
    data_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "python_doc_rag.vector_store._import_faiss",
        lambda: FakeFaiss(3, 384),
    )

    result = inspect_data_root(data_root)
    exit_code = main(["check", "--data-root", str(data_root)])

    assert not result.succeeded
    assert exit_code == 1
    error = capsys.readouterr().err
    assert "metadataとFAISS indexの件数が一致しません: 2 != 3" in error
    assert "manifestとFAISS indexの件数が一致しません: 2 != 3" in error


@pytest.mark.parametrize(
    ("device", "dtype", "message"),
    [
        ("cuda", "auto", "CUDAを利用できません"),
        ("cpu", "float16", "CPUでは--dtype float16を使用できません"),
        ("cpu", "bfloat16", "CPUでは--dtype bfloat16を使用できません"),
    ],
)
def test_device_dtype_errors_precede_model_loading(
    device: str,
    dtype: str,
    message: str,
    data_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = build_parser().parse_args(
        [
            "ask",
            "--data-root",
            str(data_root),
            "--question",
            "質問",
            "--device",
            device,
            "--dtype",
            dtype,
        ]
    )

    def fake_import(name: str, error_message: str) -> Any:
        if name == "torch":
            return FakeTorch()
        pytest.fail(f"model dependency must not load: {name}, {error_message}")

    monkeypatch.setattr(cli, "_import_runtime_module", fake_import)

    with pytest.raises(CliError, match=message):
        cli._build_service(args, data_root)


@pytest.mark.parametrize(
    "arguments",
    [
        ["--help"],
        ["ask", "--help"],
        ["chat", "--help"],
        ["check", "--help"],
        ["profile", "--help"],
        ["prepare", "--help"],
        ["serve", "--help"],
    ],
)
def test_help_does_not_import_heavy_dependencies(arguments: list[str]) -> None:
    code = f"""
import builtins
import runpy
import sys

original_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name.split('.', maxsplit=1)[0] in {{
        'fastapi',
        'torch',
        'transformers',
        'sentence_transformers',
        'uvicorn',
    }}:
        raise ModuleNotFoundError(name)
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
sys.argv = ['python -m python_doc_rag', *{arguments!r}]
runpy.run_module('python_doc_rag', run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_package_import_does_not_import_heavy_dependencies() -> None:
    code = """
import builtins

original_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name.split('.', maxsplit=1)[0] in {
        'torch',
        'transformers',
        'sentence_transformers',
    }:
        raise ModuleNotFoundError(name)
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
import python_doc_rag
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_serve_defaults_to_one_implicit_worker_and_safe_host() -> None:
    args = build_parser().parse_args(
        ["serve", "--service-config", "service.toml"]
    )

    assert args.host == "127.0.0.1"
    assert args.port == 8000
    assert not hasattr(args, "workers")
    assert not hasattr(args, "reload")


def test_serve_missing_api_dependency_is_actionable(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def missing(name: str, message: str) -> Any:
        assert name == "fastapi"
        raise CliError(message)

    monkeypatch.setattr(cli, "_import_runtime_module", missing)

    code = main(["serve", "--service-config", "service.toml"])

    assert code == 1
    error = capsys.readouterr().err
    assert "--extra inference --extra api" in error


@pytest.mark.parametrize(("started", "expected_code"), [(True, 0), (False, 1)])
def test_serve_forces_one_worker_and_reports_startup_failure(
    started: bool,
    expected_code: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeConfig:
        def __init__(self, app: Any, **options: Any) -> None:
            captured["app"] = app
            captured.update(options)

    class FakeServer:
        def __init__(self, config: FakeConfig) -> None:
            captured["config"] = config
            self.started = started

        def run(self) -> None:
            captured["ran"] = True

    fake_uvicorn = SimpleNamespace(Config=FakeConfig, Server=FakeServer)

    def fake_import(name: str, message: str) -> Any:
        assert message
        return object() if name == "fastapi" else fake_uvicorn

    app = object()
    monkeypatch.setattr(cli, "_import_runtime_module", fake_import)
    monkeypatch.setattr("python_doc_rag.api.create_app", lambda **kwargs: app)

    code = main(
        [
            "serve",
            "--service-config",
            "service.toml",
            "--host",
            "127.0.0.2",
            "--port",
            "8123",
        ]
    )

    assert code == expected_code
    assert captured["app"] is app
    assert captured["host"] == "127.0.0.2"
    assert captured["port"] == 8123
    assert captured["workers"] == 1
    assert captured["reload"] is False
    assert captured["ran"] is True


def test_serve_normalizes_uvicorn_startup_system_exit_to_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeConfig:
        def __init__(self, app: Any, **options: Any) -> None:
            del app, options

    class FakeServer:
        started = False

        def __init__(self, config: FakeConfig) -> None:
            del config

        def run(self) -> None:
            raise SystemExit(3)

    fake_uvicorn = SimpleNamespace(Config=FakeConfig, Server=FakeServer)
    monkeypatch.setattr(
        cli,
        "_import_runtime_module",
        lambda name, message: object() if name == "fastapi" else fake_uvicorn,
    )
    monkeypatch.setattr("python_doc_rag.api.create_app", lambda **kwargs: object())

    assert main(["serve", "--service-config", "missing.toml"]) == 1


def _abstained_answer(*, model_reason: str = "insufficient_evidence"):
    from python_doc_rag.models import AbstainedAnswer

    return AbstainedAnswer(
        reason_code=model_reason,
        retrieved_chunks=(_chunk(1),),
        generation_attempts=1,
    )


def test_answer_mode_defaults_to_legacy_and_is_available_only_for_generation() -> None:
    parser = build_parser()
    for command in ("ask", "chat"):
        args = parser.parse_args([command])
        assert args.answer_mode == "legacy"
        selected = parser.parse_args(
            [command, "--answer-mode", "answer-or-abstain"]
        )
        assert selected.answer_mode == "answer-or-abstain"
    check = parser.parse_args(["check"])
    assert not hasattr(check, "answer_mode")


def test_ask_abstention_is_success_with_fixed_message_and_no_sources(
    data_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class AbstainService:
        def answer(self, question: str) -> AnswerExecution:
            return AnswerExecution(
                answer=_abstained_answer(),
                retrieval_seconds=0.1,
                generation_seconds=0.2,
                total_seconds=0.3,
            )

    monkeypatch.setattr(cli, "_build_service", lambda args, root: AbstainService())
    code = main(
        [
            "ask",
            "--data-root",
            str(data_root),
            "--answer-mode",
            "answer-or-abstain",
            "--question",
            "回答不能",
        ]
    )
    output = capsys.readouterr().out
    assert code == 0
    assert "回答を控えます" in output
    assert "取得した資料だけでは" in output
    assert "出典" not in output
    assert "insufficient_evidence" not in output
    assert "trusted.invalid" not in output


def test_chat_continues_after_normal_abstention(
    data_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class MixedService:
        def answer(self, question: str) -> AnswerExecution:
            calls.append(question)
            outcome = _abstained_answer() if len(calls) == 1 else _answer(question)
            return AnswerExecution(outcome, 0.1, 0.2, 0.3)

    inputs = iter(["回答不能", "回答可能", "exit"])
    monkeypatch.setattr(cli, "_build_service", lambda args, root: MixedService())
    monkeypatch.setattr(builtins, "input", lambda prompt: next(inputs))
    assert main(["chat", "--data-root", str(data_root)]) == 0
    assert calls == ["回答不能", "回答可能"]


def test_chat_heading_uses_generic_dataset_name_without_changing_legacy(
    tmp_path: Path,
) -> None:
    from python_doc_rag.cli import _chat_heading
    from python_doc_rag.dataset_layout import (
        generic_dataset_manifest,
        write_dataset_manifest_atomic,
    )

    assert _chat_heading(tmp_path) == "Python公式ドキュメント質問応答"
    manifest = generic_dataset_manifest(
        dataset_name="Fixture docs",
        dataset_slug="fixture-docs",
        loader_type="bounded-http",
        parser_type="generic-html",
        site_config_sha256="a" * 64,
        created_at="2026-08-01T00:00:00+00:00",
        source_page_count=1,
        section_count=1,
        chunk_count=1,
    )
    write_dataset_manifest_atomic(manifest, tmp_path / "dataset_manifest.json")
    assert _chat_heading(tmp_path) == "Fixture docs 質問応答"
