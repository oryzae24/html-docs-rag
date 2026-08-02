"""Command-line interface for the Python documentation RAG PoC."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from python_doc_rag.answer_contract import ANSWER_MODE_LEGACY, ANSWER_MODES
from python_doc_rag.application import (
    AnswerExecution,
    AnswerServiceError,
    ApplicationError,
    RagService,
    ResolvedRuntimeSettings,
    TimingRetriever,
    build_resolved_service,
)
from python_doc_rag.config import DEFAULT_GENERATION_MODEL
from python_doc_rag.generation import GenerationConfig
from python_doc_rag.models import AbstainedAnswer
from python_doc_rag.profile_artifacts import (
    profile_artifact_errors,
    profile_artifact_paths,
    safe_profile_relative_path,
)
from python_doc_rag.profiles import (
    DEFAULT_PROFILE_NAME,
    PROFILE_NAMES,
    RuntimeProfile,
    runtime_profile,
)

_DATA_ROOT_ENVIRONMENT_VARIABLE = "PYTHON_DOC_RAG_DATA_ROOT"
_DEFAULT_GENERATION_CONFIG = GenerationConfig()
_PROCESSED_JSONL = Path("data/processed/python_3_13_ja_chunks.jsonl")
_INDEX_FILE = Path("indexes/python_3_13_ja.faiss")
_METADATA_JSONL = Path("indexes/python_3_13_ja_metadata.jsonl")
_INDEX_MANIFEST = Path("indexes/python_3_13_ja_index_manifest.json")
_DEFAULT_RETRIEVER = "dense"


class CliError(RuntimeError):
    """Represent an expected CLI failure with an actionable message."""


@dataclass(frozen=True, slots=True)
class ArtifactPaths:
    """Expected Python documentation artifacts below one data root."""

    data_root: Path
    processed_jsonl: Path
    index_path: Path
    metadata_path: Path
    manifest_path: Path


@dataclass(frozen=True, slots=True)
class DataInspection:
    """Results of inspecting saved corpus and vector-index artifacts."""

    data_root: Path
    chunk_count: int | None
    metadata_count: int | None
    index_count: int | None
    index_dimension: int | None
    embedding_model_name: str | None
    errors: tuple[str, ...]

    @property
    def succeeded(self) -> bool:
        """Return whether every required artifact and invariant was valid."""
        return not self.errors


_TimingRetriever = TimingRetriever
_RagService = RagService


class _CliRagServiceAdapter:
    """Preserve the public CLI factory's historical CliError boundary."""

    def __init__(self, service: RagService) -> None:
        self._service = service

    def answer(self, question: str) -> AnswerExecution:
        """Answer one question and translate application errors for CLI callers."""
        return _answer_with_cli_errors(self._service, question)


def build_parser() -> argparse.ArgumentParser:
    """Create the argparse command tree without importing heavy dependencies."""
    parser = argparse.ArgumentParser(
        prog="python -m python_doc_rag",
        description=(
            "準備済みのローカル文書datasetを検索し、出典付きで回答します。"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask_parser = subparsers.add_parser(
        "ask",
        help="1件の質問へ回答する",
        description="引数または標準入力から1件の質問を受け付けます。",
    )
    _add_generation_arguments(ask_parser)
    ask_parser.add_argument("--question", help="回答する質問")
    ask_parser.set_defaults(handler=_run_ask)

    chat_parser = subparsers.add_parser(
        "chat",
        help="同じモデルとindexで複数の質問へ回答する",
        description=(
            "モデルとindexを1回ロードし、独立した複数の質問を受け付けます。"
        ),
    )
    _add_generation_arguments(chat_parser)
    chat_parser.set_defaults(handler=_run_chat)

    check_parser = subparsers.add_parser(
        "check",
        help="モデルをロードせずローカル成果物を検査する",
        description=(
            "コーパス、FAISS index、metadata、manifestの存在と整合性を検査します。"
        ),
    )
    _add_data_root_argument(check_parser)
    check_parser.add_argument(
        "--debug",
        action="store_true",
        help="失敗時にtracebackを表示する",
    )
    _add_profile_argument(check_parser)
    check_parser.set_defaults(handler=_run_check)

    profile_parser = subparsers.add_parser(
        "profile",
        help="固定runtime profileの内容を表示する",
        description="モデルをロードせず、固定runtime profileの内容を表示します。",
    )
    profile_parser.add_argument("name", choices=PROFILE_NAMES)
    profile_parser.set_defaults(handler=_run_profile)

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="設定済み静的HTML datasetを取得・解析・index化する",
        description=(
            "SiteConfigに従ってLoader、Parser、Chunker、indexとprofile "
            "artifactを一つの再開可能な処理で準備します。"
        ),
    )
    prepare_parser.add_argument(
        "--site-config",
        type=Path,
        required=True,
        help="strict TOML site configuration",
    )
    _add_data_root_argument(prepare_parser)
    prepare_parser.add_argument(
        "--source-root",
        type=Path,
        help="local-html-tree loader用のHTML document root",
    )
    prepare_parser.add_argument(
        "--offline",
        action="store_true",
        help="HTTP取得を禁止し、完成済みraw cacheだけを再生する",
    )
    prepare_parser.add_argument(
        "--refresh",
        action="store_true",
        help="refresh対応sourceを明示的に再取得して全artifactを再構築する",
    )
    prepare_parser.add_argument(
        "--rebuild",
        action="store_true",
        help="networkを使わず固定snapshot/cacheから下流artifactだけ再構築する",
    )
    prepare_parser.add_argument(
        "--resume",
        action="store_true",
        help="検証可能なstagingとraw cacheから中断処理を再開する",
    )
    prepare_parser.add_argument(
        "--until",
        choices=("corpus", "index", "profile", "all"),
        default="all",
        help="準備を完了する段階（既定: all）",
    )
    prepare_parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda",
        help="index構築device（既定: cuda）",
    )
    prepare_parser.add_argument(
        "--debug",
        action="store_true",
        help="失敗時にtracebackを表示する",
    )
    prepare_parser.set_defaults(handler=_run_prepare)

    serve_parser = subparsers.add_parser(
        "serve",
        help="複数knowledge baseのread-only REST APIを起動する",
        description=(
            "strict ServiceConfigを読み込み、共有ローカルモデルとKB別検索"
            "artifactを1 workerで提供します。"
        ),
    )
    serve_parser.add_argument(
        "--service-config",
        type=Path,
        required=True,
        help="multi-kb-service-v1形式のstrict TOML設定",
    )
    serve_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="listen host（既定: 127.0.0.1）",
    )
    serve_parser.add_argument(
        "--port",
        type=_port_number,
        default=8000,
        help="listen port（既定: 8000）",
    )
    serve_parser.add_argument(
        "--log-level",
        choices=("critical", "error", "warning", "info", "debug"),
        default="info",
        help="Uvicorn log level（既定: info）",
    )
    serve_parser.set_defaults(handler=_run_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one CLI command and translate expected failures to exit codes."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except CliError as error:
        print(f"エラー: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n処理を中断しました。", file=sys.stderr)
        return 130
    except Exception as error:
        if getattr(args, "debug", False):
            raise
        if getattr(args, "handler", None) is _run_serve:
            detail_hint = "\n詳細は --log-level debug で確認してください。"
        elif hasattr(args, "debug"):
            detail_hint = (
                "\n詳細を確認するには --debug を付けて再実行してください。"
            )
        else:
            detail_hint = ""
        print(
            f"エラー: {_exception_message(error)}{detail_hint}",
            file=sys.stderr,
        )
        return 1


def resolve_data_root(argument: Path | None) -> Path:
    """Resolve CLI argument before the environment variable."""
    if argument is not None:
        return argument.expanduser()
    configured = os.environ.get(_DATA_ROOT_ENVIRONMENT_VARIABLE)
    if configured and configured.strip():
        return Path(configured).expanduser()
    raise CliError(
        "--data-rootを指定するか、環境変数 "
        f"{_DATA_ROOT_ENVIRONMENT_VARIABLE} を設定してください。"
    )


def build_profile_service(
    data_root: Path,
    *,
    profile_name: str = "recommended-v2",
    device: str = "auto",
) -> _CliRagServiceAdapter:
    """Build one reusable profile service for local evaluation and CLI smoke."""
    from python_doc_rag.application import build_profile_service as build_service

    try:
        service = build_service(
            data_root,
            profile_name=profile_name,
            device=device,
        )
    except ApplicationError as error:
        raise CliError(str(error)) from error
    return _CliRagServiceAdapter(service)


def inspect_data_root(data_root: Path) -> DataInspection:
    """Inspect saved artifacts without loading either ML model."""
    paths = _artifact_paths(data_root)
    errors: list[str] = []
    if not paths.data_root.exists():
        errors.append(f"data-rootが存在しません: {paths.data_root}")
    elif not paths.data_root.is_dir():
        errors.append(f"data-rootがディレクトリではありません: {paths.data_root}")

    expected_files = (
        ("processed JSONL", paths.processed_jsonl),
        ("FAISS index", paths.index_path),
        ("metadata JSONL", paths.metadata_path),
        ("index manifest", paths.manifest_path),
    )
    for label, path in expected_files:
        if not path.is_file():
            errors.append(f"{label}が見つかりません: {path}")

    chunk_count = _safe_chunk_count(
        paths.processed_jsonl,
        "processed JSONL",
        errors,
    )
    metadata_count = _safe_chunk_count(
        paths.metadata_path,
        "metadata JSONL",
        errors,
    )
    manifest = _safe_manifest(paths.manifest_path, errors)
    embedding_model_name = _manifest_string(manifest, "model_name", errors)
    manifest_count = _manifest_positive_integer(manifest, "chunk_count", errors)
    manifest_dimension = _manifest_positive_integer(
        manifest,
        "embedding_dimension",
        errors,
    )
    index_count, index_dimension = _safe_index_summary(paths.index_path, errors)

    _compare_counts("processed JSONL", chunk_count, "metadata", metadata_count, errors)
    _compare_counts("metadata", metadata_count, "FAISS index", index_count, errors)
    _compare_counts("manifest", manifest_count, "FAISS index", index_count, errors)
    _compare_counts(
        "manifest次元",
        manifest_dimension,
        "FAISS index次元",
        index_dimension,
        errors,
    )

    return DataInspection(
        data_root=paths.data_root,
        chunk_count=chunk_count,
        metadata_count=metadata_count,
        index_count=index_count,
        index_dimension=index_dimension,
        embedding_model_name=embedding_model_name,
        errors=tuple(errors),
    )


def _add_data_root_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-root",
        type=Path,
        help=(
            "data/とindexes/を含むルート。未指定時は"
            f"{_DATA_ROOT_ENVIRONMENT_VARIABLE}を使用"
        ),
    )


def _add_generation_arguments(parser: argparse.ArgumentParser) -> None:
    _add_data_root_argument(parser)
    _add_profile_argument(parser)
    parser.add_argument(
        "--model-name",
        default=DEFAULT_GENERATION_MODEL,
        help=f"回答生成モデル（既定: {DEFAULT_GENERATION_MODEL}）",
    )
    parser.add_argument(
        "--model-revision",
        help="回答生成モデルの固定revision（既定: model側の既定revision）",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="推論device（既定: auto）",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
        help="生成モデルのdtype（既定: auto）",
    )
    parser.add_argument(
        "--top-k",
        type=_positive_integer,
        default=_DEFAULT_GENERATION_CONFIG.retrieval_limit,
        help=(
            "検索する上位チャンク数"
            f"（既定: {_DEFAULT_GENERATION_CONFIG.retrieval_limit}）"
        ),
    )
    parser.add_argument(
        "--retriever",
        choices=("dense", "hybrid"),
        default=_DEFAULT_RETRIEVER,
        help=f"検索方式（既定: {_DEFAULT_RETRIEVER}）",
    )
    parser.add_argument(
        "--answer-mode",
        choices=ANSWER_MODES,
        default=ANSWER_MODE_LEGACY,
        help=f"回答契約（既定: {ANSWER_MODE_LEGACY}）",
    )
    parser.add_argument(
        "--max-input-tokens",
        type=_positive_integer,
        default=_DEFAULT_GENERATION_CONFIG.max_prompt_tokens,
        help=(
            "入力token上限"
            f"（既定: {_DEFAULT_GENERATION_CONFIG.max_prompt_tokens}）"
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=_positive_integer,
        default=_DEFAULT_GENERATION_CONFIG.max_new_tokens,
        help=(
            "生成token上限"
            f"（既定: {_DEFAULT_GENERATION_CONFIG.max_new_tokens}）"
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="失敗時にtracebackを表示する",
    )


def _run_ask(args: argparse.Namespace) -> int:
    _apply_runtime_profile(args)
    question = args.question
    if question is None:
        try:
            question = input("質問> ")
        except EOFError as error:
            raise CliError("質問が入力されませんでした。") from error
    question = _validate_question(question)
    data_root = resolve_data_root(args.data_root)
    service = _build_service(args, data_root)
    _print_answer(_answer_with_cli_errors(service, question))
    return 0


def _run_chat(args: argparse.Namespace) -> int:
    _apply_runtime_profile(args)
    data_root = resolve_data_root(args.data_root)
    service = _build_service(args, data_root)
    print(_chat_heading(data_root))
    print("終了するには exit、quit、またはCtrl+Dを入力してください。")
    while True:
        try:
            question = input("\n質問> ")
        except EOFError:
            print("\n終了します。")
            return 0
        except KeyboardInterrupt:
            print("\n終了します。")
            return 0

        normalized = question.strip()
        if not normalized:
            continue
        if normalized.lower() in {"exit", "quit"}:
            print("終了します。")
            return 0
        try:
            execution = _answer_with_cli_errors(service, normalized)
        except KeyboardInterrupt:
            print("\n終了します。")
            return 0
        except CliError as error:
            _print_chat_failure(error)
            continue
        _print_answer(execution)


def _run_check(args: argparse.Namespace) -> int:
    profile = runtime_profile(args.profile)
    data_root = resolve_data_root(args.data_root)
    result = inspect_data_root(data_root)
    profile_errors = profile_artifact_errors(profile, data_root)
    if not result.succeeded or profile_errors:
        print("データ検査: 失敗", file=sys.stderr)
        for error in (*result.errors, *profile_errors):
            print(f"- {error}", file=sys.stderr)
        return 1

    print("データ検査: 成功")
    print(f"チャンク数: {result.chunk_count}")
    print(f"index件数: {result.index_count}")
    print(f"次元: {result.index_dimension}")
    print(f"Embeddingモデル: {result.embedding_model_name}")
    print(f"data-root: {result.data_root}")
    print(f"profile: {profile.name}")
    if profile.retriever == "technical-field":
        print("追加data artifact:")
        resolved_profile = profile_artifact_paths(profile, data_root)
        for path in resolved_profile.required_paths:
            print(f"- {path.relative_to(resolved_profile.data_root).as_posix()}")
        print("モデルcacheは起動時に検査します（optional extra: inference）")
    elif profile.reranker_model_key is not None:
        print(
            "追加data artifact: なし（baseline index/metadataを再利用。"
            "local reranker modelは起動時に検証）"
        )
    return 0


def _run_profile(args: argparse.Namespace) -> int:
    profile = runtime_profile(args.name)
    payload = profile.to_dict()
    if profile.reranker_model_key is not None:
        from python_doc_rag.reranking import model_spec_for_key

        spec = model_spec_for_key(profile.reranker_model_key)
        payload.update(
            {
                "reranker_model": spec.model_name,
                "reranker_model_revision": spec.revision,
                "reranker_model_license": spec.license,
                "reranker_trust_remote_code": False,
            }
        )
    if profile.embedding_model_key is not None:
        from python_doc_rag.embedding_models import retrieval_embedding_spec

        spec = retrieval_embedding_spec(profile.embedding_model_key)
        payload.update(
            {
                "embedding_model": spec.model_name,
                "embedding_model_revision": spec.revision,
                "embedding_model_license": spec.license,
                "embedding_dimension": spec.embedding_dimension,
                "embedding_max_sequence_length": spec.max_sequence_length,
                "embedding_pooling": spec.pooling,
                "embedding_query_prefix": spec.query_prefix,
                "embedding_document_prefix": spec.document_prefix,
                "embedding_normalized": spec.normalize_embeddings,
                "embedding_trust_remote_code": spec.trust_remote_code,
            }
        )
    if profile.generation_model_key is not None:
        from python_doc_rag.generator_models import generator_model_spec

        spec = generator_model_spec(profile.generation_model_key)
        payload.update(
            {
                "generation_model_license": spec.license,
                "generation_template_options": spec.template_options(),
                "generation_options": spec.generation_options(),
                "generation_sampling_seed": spec.sampling_seed,
                "generation_trust_remote_code": spec.trust_remote_code,
            }
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _run_prepare(args: argparse.Namespace) -> int:
    from python_doc_rag.preparation import prepare_dataset

    result = prepare_dataset(
        args.site_config,
        resolve_data_root(args.data_root),
        source_root=args.source_root,
        offline=args.offline,
        refresh=args.refresh,
        rebuild=args.rebuild,
        resume=args.resume,
        until=args.until,
        device=args.device,
    )
    print("dataset準備: 成功")
    print(f"dataset: {result.dataset_manifest.dataset_name}")
    print(f"data-root: {result.data_root}")
    print(f"source page: {result.source_page_count}")
    print(f"parsed page: {result.parsed_page_count}")
    print(f"failed page: {result.failed_page_count}")
    print(f"section: {result.section_count}")
    print(f"chunk: {result.chunk_count}")
    print(f"processed SHA-256: {result.corpus_sha256}")
    print(f"既存dataset再利用: {result.reused_dataset}")
    print(f"source cache再利用: {result.source_cache_reused}")
    print(f"source snapshot SHA-256: {result.source_snapshot_sha256}")
    if result.dense_index_result is not None:
        print(f"dense index: {result.dense_index_result.index_path}")
    if result.profile_result is not None:
        print(
            "recommended-v2 artifact再利用: "
            f"{result.profile_result.reused_existing}"
        )
    return 0


def _run_serve(args: argparse.Namespace) -> int:
    _import_runtime_module(
        "fastapi",
        "REST APIにはapi依存関係が必要です。"
        "`uv sync --frozen --extra inference --extra api`を実行してください。",
    )
    uvicorn = _import_runtime_module(
        "uvicorn",
        "REST APIにはUvicornが必要です。"
        "`uv sync --frozen --extra inference --extra api`を実行してください。",
    )
    from python_doc_rag.api import create_app

    app = create_app(service_config=args.service_config)
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host=args.host,
            port=args.port,
            log_level=args.log_level,
            workers=1,
            reload=False,
        )
    )
    try:
        server.run()
    except KeyboardInterrupt:
        return 0
    except SystemExit as error:
        return 0 if error.code in {None, 0} else 1
    return 0 if server.started else 1


def _build_service(args: argparse.Namespace, data_root: Path) -> _RagService:
    profile = runtime_profile(args.profile)
    settings = ResolvedRuntimeSettings(
        profile=profile,
        requested_device=args.device,
        generation_model=args.model_name,
        generation_model_revision=args.model_revision,
        dtype=args.dtype,
        retriever=args.retriever,
        answer_mode=args.answer_mode,
        top_k=args.top_k,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        field_rrf_k=getattr(args, "field_rrf_k", profile.field_rrf_k),
        field_candidate_k=getattr(
            args,
            "field_candidate_k",
            profile.field_candidate_k,
        ),
        reranker_model_key=getattr(
            args,
            "reranker_model_key",
            profile.reranker_model_key,
        ),
        reranker_candidate_k=getattr(
            args,
            "reranker_candidate_k",
            profile.reranker_candidate_k,
        ),
        reranker_batch_size=getattr(
            args,
            "reranker_batch_size",
            profile.reranker_batch_size,
        ),
        reranker_max_length=getattr(
            args,
            "reranker_max_length",
            profile.reranker_max_length,
        ),
    )
    try:
        return build_resolved_service(
            settings,
            data_root,
            module_importer=_import_runtime_module,
        )
    except AnswerServiceError as error:
        raise CliError(str(error)) from error
    except Exception as error:
        from python_doc_rag.application import ApplicationError

        if isinstance(error, ApplicationError):
            raise CliError(str(error)) from error
        raise


def _artifact_paths(data_root: Path) -> ArtifactPaths:
    from python_doc_rag.dataset_layout import resolve_dataset_artifacts

    resolved = resolve_dataset_artifacts(data_root)
    return ArtifactPaths(
        data_root=resolved.data_root,
        processed_jsonl=resolved.processed_jsonl,
        index_path=resolved.index_path,
        metadata_path=resolved.metadata_path,
        manifest_path=resolved.index_manifest_path,
    )


def _chat_heading(data_root: Path) -> str:
    from python_doc_rag.dataset_layout import resolve_dataset_artifacts

    dataset = resolve_dataset_artifacts(data_root).dataset_manifest
    if dataset is None:
        return "Python公式ドキュメント質問応答"
    return f"{dataset.dataset_name} 質問応答"


def _add_profile_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--profile",
        choices=PROFILE_NAMES,
        default=DEFAULT_PROFILE_NAME,
        help=(
            "固定runtime構成。recommendedはtechnical field retrieval + "
            "local reranker + Qwen3-8B + answer-or-abstain（既定: default）"
        ),
    )


def _apply_runtime_profile(args: argparse.Namespace) -> None:
    profile = runtime_profile(args.profile)
    args.embedding_model_key = profile.embedding_model_key
    args.field_rrf_k = profile.field_rrf_k
    args.field_candidate_k = profile.field_candidate_k
    args.generator_model_key = profile.generation_model_key
    args.reranker_model_key = profile.reranker_model_key
    args.reranker_candidate_k = profile.reranker_candidate_k
    args.reranker_batch_size = profile.reranker_batch_size
    args.reranker_max_length = profile.reranker_max_length
    if profile.name == DEFAULT_PROFILE_NAME:
        return
    args.retriever = profile.retriever
    args.answer_mode = profile.answer_mode
    args.model_name = profile.generation_model
    args.model_revision = profile.generation_model_revision
    args.dtype = profile.dtype
    args.top_k = profile.top_k
    args.max_input_tokens = profile.max_input_tokens
    args.max_new_tokens = profile.max_new_tokens


def _require_runtime_artifacts(
    paths: ArtifactPaths,
    profile: RuntimeProfile,
) -> None:
    if not paths.data_root.is_dir():
        raise CliError(f"data-rootが存在しません: {paths.data_root}")
    required = (
        ("FAISS index", paths.index_path),
        ("metadata JSONL", paths.metadata_path),
        ("index manifest", paths.manifest_path),
    )
    missing = [f"{label}: {path}" for label, path in required if not path.is_file()]
    if missing:
        details = "\n".join(f"- {item}" for item in missing)
        raise CliError(
            "質問応答に必要なファイルが不足しています。checkコマンドを"
            f"実行してください。\n{details}"
        )
    profile_errors = profile_artifact_errors(profile, paths.data_root)
    if profile_errors:
        details = "\n".join(f"- {item}" for item in profile_errors)
        raise CliError(
            f"profile {profile.name}に必要なartifactが不正です。"
            f"check --profile {profile.name}を実行してください。\n{details}"
        )


def _required_profile_path(value: str | None, label: str) -> Path:
    try:
        return safe_profile_relative_path(value, label)
    except ValueError as error:
        raise CliError(str(error)) from error


def _required_resolved_path(value: Path | None, label: str) -> Path:
    if value is None:
        raise CliError(f"profileに{label} pathがありません。")
    return value


def _safe_chunk_count(path: Path, label: str, errors: list[str]) -> int | None:
    if not path.is_file():
        return None
    try:
        from python_doc_rag.vector_store import iter_chunks_jsonl

        count = sum(1 for _ in iter_chunks_jsonl(path))
    except Exception as error:
        errors.append(f"{label}を読み取れません: {_exception_message(error)}")
        return None
    if count < 1:
        errors.append(f"{label}にチャンクがありません: {path}")
    return count


def _safe_manifest(path: Path, errors: list[str]) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return _read_json_object(path)
    except (OSError, ValueError) as error:
        errors.append(f"index manifestを読み取れません: {_exception_message(error)}")
        return None


def _safe_index_summary(
    path: Path,
    errors: list[str],
) -> tuple[int | None, int | None]:
    if not path.is_file():
        return None, None
    try:
        from python_doc_rag.vector_store import _import_faiss

        faiss = _import_faiss()
        index = faiss.read_index(str(path))
        count = int(index.ntotal)
        dimension = int(index.d)
    except Exception as error:
        errors.append(
            "FAISS indexを読み取れません。faiss-cpuの導入とファイル内容を"
            f"確認してください: {_exception_message(error)}"
        )
        return None, None
    if count < 1:
        errors.append(f"FAISS indexにベクトルがありません: {path}")
    if dimension < 1:
        errors.append(f"FAISS indexの次元が不正です: {dimension}")
    return count, dimension


def _manifest_string(
    manifest: dict[str, Any] | None,
    key: str,
    errors: list[str],
) -> str | None:
    if manifest is None:
        return None
    value = manifest.get(key)
    if not isinstance(value, str) or not value.strip():
        errors.append(f"index manifestの{key}が空または不正です。")
        return None
    return value


def _manifest_positive_integer(
    manifest: dict[str, Any] | None,
    key: str,
    errors: list[str],
) -> int | None:
    if manifest is None:
        return None
    value = manifest.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        errors.append(f"index manifestの{key}が正の整数ではありません。")
        return None
    return value


def _compare_counts(
    first_label: str,
    first: int | None,
    second_label: str,
    second: int | None,
    errors: list[str],
) -> None:
    if first is not None and second is not None and first != second:
        errors.append(
            f"{first_label}と{second_label}の件数が一致しません: "
            f"{first} != {second}"
        )


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.expanduser().read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"{path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: JSONが不正です（{error.msg}）") from error
    if not isinstance(data, dict):
        raise ValueError(f"JSON objectではありません: {path}")
    return data


def _import_runtime_module(name: str, message: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as error:
        raise CliError(message) from error


def _validate_question(question: str) -> str:
    normalized = question.strip()
    if not normalized:
        raise CliError("質問を空にすることはできません。")
    return normalized


def _answer_with_cli_errors(service: Any, question: str) -> AnswerExecution:
    try:
        return service.answer(question)
    except AnswerServiceError as error:
        raise CliError(str(error)) from error


def _print_answer(execution: AnswerExecution) -> None:
    answer = execution.answer
    if isinstance(answer, AbstainedAnswer):
        print("\n回答を控えます")
        print("--------------")
        print("取得した資料だけでは、質問へ十分な根拠をもって回答できませんでした。")
    else:
        print("\n回答")
        print("----")
        print(answer.answer_text)
        print("\n出典")
        print("----")
        for source in answer.sources:
            print(f"[{source.label}] {source.page_title}")
            print(source.section_title)
            print(source.url)
    print("\n処理時間")
    print("--------")
    print(f"検索: {execution.retrieval_seconds:.3f}秒")
    print(f"生成: {execution.generation_seconds:.3f}秒")
    print(f"全体: {execution.total_seconds:.3f}秒")


def _print_chat_failure(error: CliError) -> None:
    """Report one recoverable chat failure without ending the session."""
    print("\n回答できませんでした", file=sys.stderr)
    print("--------------------", file=sys.stderr)
    print(error, file=sys.stderr)
    print(
        "質問を言い換えるか、別の質問を入力してください。",
        file=sys.stderr,
    )


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("正の整数を指定してください") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("1以上の整数を指定してください")
    return parsed


def _port_number(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("portは整数で指定してください") from error
    if not 1 <= parsed <= 65535:
        raise argparse.ArgumentTypeError("portは1から65535で指定してください")
    return parsed


def _exception_message(error: BaseException) -> str:
    message = str(error).strip()
    return message or error.__class__.__name__
