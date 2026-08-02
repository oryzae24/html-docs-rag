"""Public application services shared by the local CLI and HTTP API."""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

from python_doc_rag.answer_contract import generation_contract_for_mode
from python_doc_rag.generation import DEFAULT_DOCUMENT_SCOPE, GenerationConfig
from python_doc_rag.models import AnswerOutcome
from python_doc_rag.profile_artifacts import (
    ProfileArtifactPaths,
    profile_artifact_paths,
    validate_profile_artifacts,
)
from python_doc_rag.profiles import RuntimeProfile, runtime_profile

_HYBRID_NGRAM_SIZES = (2,)
_HYBRID_RRF_K = 10
_HYBRID_CANDIDATE_K = 30
LOGGER = logging.getLogger(__name__)
_SERVER_GENERATION_HISTORY_LIMIT = 64


class ApplicationError(RuntimeError):
    """Base class for expected application-layer failures."""


class ArtifactValidationError(ApplicationError):
    """Raised before model loading when a knowledge base is invalid."""


class InferenceLoadError(ApplicationError):
    """Raised when one shared local inference resource cannot be loaded."""


class AnswerServiceError(ApplicationError):
    """Raised when retrieval or generation cannot produce an outcome."""

    def __init__(self, message: str, *, contract_failure: bool = False) -> None:
        self.contract_failure = contract_failure
        super().__init__(message)


class KnowledgeBaseNotFoundError(ApplicationError):
    """Raised when a configured knowledge base ID is unknown."""


@dataclass(frozen=True, slots=True)
class ArtifactPaths:
    """Resolved baseline artifact paths below one data root."""

    data_root: Path
    processed_jsonl: Path
    index_path: Path
    metadata_path: Path
    manifest_path: Path


@dataclass(frozen=True, slots=True)
class DataInspection:
    """Model-free inspection of one dataset's baseline artifacts."""

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


@dataclass(frozen=True, slots=True)
class AnswerExecution:
    """One finalized answer with per-question timing and token usage."""

    answer: AnswerOutcome
    retrieval_seconds: float
    generation_seconds: float
    total_seconds: float
    input_tokens: int = 0
    generated_tokens: int = 0
    generation_calls: int = 0


@dataclass(frozen=True, slots=True)
class EmbeddingIdentity:
    """Properties that must match before one embedding model can be shared."""

    model_name: str
    model_revision: str | None
    embedding_dimension: int
    normalized_embeddings: bool
    query_prefix: str
    document_prefix: str
    trust_remote_code: bool


@dataclass(frozen=True, slots=True)
class ResolvedRuntimeSettings:
    """Complete model and pipeline settings after applying one profile."""

    profile: RuntimeProfile
    requested_device: str
    generation_model: str
    generation_model_revision: str | None
    dtype: str
    retriever: str
    answer_mode: str
    top_k: int
    max_input_tokens: int
    max_new_tokens: int
    field_rrf_k: int | None
    field_candidate_k: int | None
    reranker_model_key: str | None
    reranker_candidate_k: int | None
    reranker_batch_size: int | None
    reranker_max_length: int | None


@dataclass(frozen=True, slots=True)
class KnowledgeBasePlan:
    """Validated model-free plan for one configured knowledge base."""

    id: str
    display_name: str
    data_root: Path
    dataset_name: str
    embedding_identity: EmbeddingIdentity
    artifacts: Any = None
    profile_artifacts: Any = None


@dataclass(frozen=True, slots=True)
class SharedInferenceResources:
    """Models loaded once and reused by every knowledge base."""

    profile: RuntimeProfile
    resolved_device: str
    embedding_model: Any
    reranker_scorer: Any | None
    generator: Any
    prompt_serializer: Any = None
    settings: ResolvedRuntimeSettings | None = None


class KnowledgeBaseConfigProtocol(Protocol):
    """Configuration fields required by the runtime planner."""

    id: str
    display_name: str
    data_root: Path


class ServiceConfigProtocol(Protocol):
    """Service configuration fields required by the runtime builder."""

    profile: str | RuntimeProfile
    device: str
    knowledge_bases: tuple[KnowledgeBaseConfigProtocol, ...]


@dataclass(frozen=True, slots=True)
class RuntimeLoaders:
    """Injectable model and knowledge-base construction boundaries."""

    validate_knowledge_base: Callable[
        [KnowledgeBaseConfigProtocol, RuntimeProfile], KnowledgeBasePlan
    ]
    load_embedding: Callable[[EmbeddingIdentity, str], Any]
    load_reranker: Callable[[RuntimeProfile, str], Any | None]
    load_generator: Callable[[RuntimeProfile, str], Any]
    build_knowledge_base_service: Callable[
        [KnowledgeBasePlan, SharedInferenceResources], Any
    ]


class TimingRetriever:
    """Measure retrieval calls while preserving the Retriever interface."""

    def __init__(self, retriever: Any) -> None:
        self._retriever = retriever
        self.history: list[float] = []
        self._history_base = 0
        self._retrieval_count = 0
        self._history_limit: int | None = None

    @property
    def wrapped(self) -> Any:
        """Return the wrapped knowledge-base-specific retriever."""
        return self._retriever

    @property
    def retrieval_cursor(self) -> int:
        """Return a monotonic cursor without copying retained timings."""
        return self._retrieval_count

    def retrieval_metrics_since(self, cursor: int) -> tuple[float, ...]:
        """Return retained retrieval timings after an observed cursor."""
        if (
            isinstance(cursor, bool)
            or not isinstance(cursor, int)
            or cursor < self._history_base
            or cursor > self._retrieval_count
        ):
            raise ValueError("retrieval cursor is outside retained history")
        return tuple(self.history[cursor - self._history_base :])

    def set_history_limit(self, limit: int) -> None:
        """Bound retained timings for a long-lived server."""
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("retrieval history limit must be positive")
        self._history_limit = limit
        self._trim_history()

    def retrieve(self, question: str, *, limit: int) -> Any:
        """Retrieve and retain only elapsed time outside model-visible data."""
        started_at = time.perf_counter()
        result = self._retriever.retrieve(question, limit=limit)
        self.history.append(time.perf_counter() - started_at)
        self._retrieval_count += 1
        self._trim_history()
        return result

    def _trim_history(self) -> None:
        if self._history_limit is None:
            return
        overflow = len(self.history) - self._history_limit
        if overflow > 0:
            del self.history[:overflow]
            self._history_base += overflow


class RagService:
    """Reuse one loaded pipeline and expose independent answer measurements."""

    def __init__(
        self,
        *,
        pipeline: Any,
        timing_retriever: TimingRetriever,
        generator: Any,
    ) -> None:
        self._pipeline = pipeline
        self._timing_retriever = timing_retriever
        self._generator = generator

    @property
    def pipeline(self) -> Any:
        """Return the knowledge-base-specific pipeline for diagnostics."""
        return self._pipeline

    @property
    def retriever(self) -> Any:
        """Return the knowledge-base-specific retriever for diagnostics."""
        return self._timing_retriever.wrapped

    def set_retrieval_history_limit(self, limit: int) -> None:
        """Bound retained retrieval metrics for a long-lived runtime."""
        self._timing_retriever.set_history_limit(limit)

    def answer(self, question: str) -> AnswerExecution:
        """Generate one independent answer without conversation history."""
        generation_offset = _generation_cursor(self._generator)
        retrieval_offset = self._timing_retriever.retrieval_cursor
        started_at = time.perf_counter()
        try:
            answer = self._pipeline.answer(question)
        except Exception as error:
            from python_doc_rag.pipeline import AnswerGenerationFailedError

            if isinstance(error, AnswerGenerationFailedError):
                raise AnswerServiceError(
                    "回答出力契約の検証に2回失敗しました。質問または検索データを"
                    "確認してください。",
                    contract_failure=True,
                ) from error
            raise AnswerServiceError(
                f"回答生成に失敗しました: {_exception_message(error)}"
            ) from error
        total_seconds = time.perf_counter() - started_at

        retrieval_history = self._timing_retriever.retrieval_metrics_since(
            retrieval_offset
        )
        generation_history = _generation_metrics_since(
            self._generator,
            generation_offset,
        )
        return AnswerExecution(
            answer=answer,
            retrieval_seconds=sum(retrieval_history),
            generation_seconds=sum(
                float(metric.elapsed_seconds) for metric in generation_history
            ),
            total_seconds=total_seconds,
            input_tokens=sum(
                int(getattr(metric, "input_tokens", 0))
                for metric in generation_history
            ),
            generated_tokens=sum(
                int(getattr(metric, "generated_tokens", 0))
                for metric in generation_history
            ),
            generation_calls=len(generation_history),
        )


@dataclass(frozen=True, slots=True)
class KnowledgeBaseService:
    """One isolated artifact/retrieval graph using shared inference models."""

    id: str
    display_name: str
    dataset_name: str
    data_root: Path
    answer_service: RagService
    retriever: Any
    artifacts: ArtifactPaths
    profile_artifacts: ProfileArtifactPaths

    def answer(self, question: str) -> AnswerExecution:
        """Answer against only this knowledge base's retrieval artifacts."""
        return self.answer_service.answer(question)


class MultiKnowledgeBaseRuntime:
    """Immutable registry with globally serialized answer execution."""

    def __init__(
        self,
        *,
        profile: RuntimeProfile,
        shared_resources: SharedInferenceResources,
        knowledge_bases: Mapping[str, Any],
    ) -> None:
        ordered = dict(knowledge_bases)
        if not ordered:
            raise ValueError("knowledge_bases must not be empty")
        self._profile = profile
        self._shared_resources = shared_resources
        self._knowledge_bases = MappingProxyType(ordered)
        self._answer_semaphore = asyncio.Semaphore(1)

    @property
    def ready(self) -> bool:
        """Return true because runtimes are published only after eager loading."""
        return True

    @property
    def profile(self) -> RuntimeProfile:
        """Return the single runtime profile shared by all knowledge bases."""
        return self._profile

    @property
    def profile_name(self) -> str:
        """Return the configured profile name, preserving aliases."""
        return self._profile.name

    @property
    def shared_resources(self) -> SharedInferenceResources:
        """Return shared models for diagnostics without copying them."""
        return self._shared_resources

    @property
    def knowledge_bases(self) -> Mapping[str, Any]:
        """Return the immutable registry in ServiceConfig order."""
        return self._knowledge_bases

    @property
    def knowledge_base_count(self) -> int:
        """Return the eagerly loaded knowledge base count."""
        return len(self._knowledge_bases)

    def get_knowledge_base(self, knowledge_base_id: str) -> Any:
        """Resolve one configured service without falling back silently."""
        try:
            return self._knowledge_bases[knowledge_base_id]
        except KeyError as error:
            raise KnowledgeBaseNotFoundError(
                f"unknown knowledge base: {knowledge_base_id}"
            ) from error

    async def answer(
        self,
        knowledge_base_id: str,
        question: str,
    ) -> AnswerExecution:
        """Wait for the global slot, then run blocking RAG work off-loop."""
        service = self.get_knowledge_base(knowledge_base_id)
        async with self._answer_semaphore:
            worker = asyncio.create_task(asyncio.to_thread(service.answer, question))
            try:
                return await asyncio.shield(worker)
            except asyncio.CancelledError:
                # A cancelled await does not stop its worker thread. Keep the
                # global GPU slot until that thread has really completed so a
                # following request cannot overlap shared-model inference.
                while not worker.done():
                    try:
                        await asyncio.shield(worker)
                    except asyncio.CancelledError:
                        continue
                    except Exception:
                        break
                if worker.done() and not worker.cancelled():
                    worker.exception()
                raise


def resolved_settings_for_profile(
    profile: RuntimeProfile,
    *,
    device: str,
) -> ResolvedRuntimeSettings:
    """Resolve immutable settings for an exact named runtime profile."""
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"unsupported device: {device}")
    return ResolvedRuntimeSettings(
        profile=profile,
        requested_device=device,
        generation_model=profile.generation_model,
        generation_model_revision=profile.generation_model_revision,
        dtype=profile.dtype,
        retriever=profile.retriever,
        answer_mode=profile.answer_mode,
        top_k=profile.top_k,
        max_input_tokens=profile.max_input_tokens,
        max_new_tokens=profile.max_new_tokens,
        field_rrf_k=profile.field_rrf_k,
        field_candidate_k=profile.field_candidate_k,
        reranker_model_key=profile.reranker_model_key,
        reranker_candidate_k=profile.reranker_candidate_k,
        reranker_batch_size=profile.reranker_batch_size,
        reranker_max_length=profile.reranker_max_length,
    )


def build_multi_knowledge_base_runtime(
    config: ServiceConfigProtocol,
    loaders: RuntimeLoaders | None = None,
) -> MultiKnowledgeBaseRuntime:
    """Validate every KB, load shared models once, then eagerly load all KBs."""
    profile = (
        config.profile
        if isinstance(config.profile, RuntimeProfile)
        else runtime_profile(config.profile)
    )
    settings = resolved_settings_for_profile(profile, device=config.device)
    using_production_loaders = loaders is None
    if using_production_loaders:
        loaders = production_runtime_loaders(settings)
    assert loaders is not None

    plans = tuple(
        loaders.validate_knowledge_base(entry, profile)
        for entry in config.knowledge_bases
    )
    identity = _shared_embedding_identity(plans)
    resolved_device = (
        _resolve_runtime_device(
            settings.requested_device,
            settings.dtype,
            module_importer=_import_runtime_module,
        )
        if using_production_loaders
        else config.device
    )
    LOGGER.info(
        "loading shared inference resource kind=embedding profile=%s",
        profile.name,
    )
    embedding_model = loaders.load_embedding(identity, resolved_device)
    _validate_loaded_embedding_dimension(embedding_model, identity)
    LOGGER.info(
        "loaded shared inference resource kind=embedding profile=%s",
        profile.name,
    )
    LOGGER.info(
        "loading shared inference resource kind=reranker profile=%s",
        profile.name,
    )
    reranker = loaders.load_reranker(profile, resolved_device)
    LOGGER.info(
        "loaded shared inference resource kind=reranker profile=%s",
        profile.name,
    )
    LOGGER.info(
        "loading shared inference resource kind=generator profile=%s",
        profile.name,
    )
    generator = loaders.load_generator(profile, resolved_device)
    _bound_server_generation_history(generator)
    LOGGER.info(
        "loaded shared inference resource kind=generator profile=%s",
        profile.name,
    )
    prompt_serializer = (
        _prompt_serializer(profile, generator) if using_production_loaders else None
    )
    shared = SharedInferenceResources(
        profile=profile,
        resolved_device=resolved_device,
        embedding_model=embedding_model,
        reranker_scorer=reranker,
        generator=generator,
        prompt_serializer=prompt_serializer,
        settings=settings,
    )
    services: dict[str, Any] = {}
    for plan in plans:
        service = loaders.build_knowledge_base_service(plan, shared)
        _bound_server_retrieval_history(service)
        services[plan.id] = service
        LOGGER.info("loaded knowledge base id=%s", plan.id)
    LOGGER.info(
        "multi-KB runtime ready profile=%s knowledge_base_count=%d",
        profile.name,
        len(services),
    )
    return MultiKnowledgeBaseRuntime(
        profile=profile,
        shared_resources=shared,
        knowledge_bases=services,
    )


def build_profile_service(
    data_root: Path,
    *,
    profile_name: str = "recommended-v2",
    device: str = "auto",
) -> RagService:
    """Build one reusable profile service for local CLI and evaluation smoke."""
    profile = runtime_profile(profile_name)
    settings = resolved_settings_for_profile(profile, device=device)
    return build_resolved_service(settings, data_root)


def build_resolved_service(
    settings: ResolvedRuntimeSettings,
    data_root: Path,
    *,
    module_importer: Callable[[str, str], Any] | None = None,
) -> RagService:
    """Build one service while preserving explicit default-CLI overrides."""
    importer = module_importer or _import_runtime_module
    loaders = production_runtime_loaders(settings, module_importer=importer)
    entry = _SingleKnowledgeBaseConfig(
        id="local",
        display_name=DEFAULT_DOCUMENT_SCOPE,
        data_root=data_root.expanduser().resolve(),
    )
    _require_artifact_files(entry.data_root, settings.profile)
    resolved_device = _resolve_runtime_device(
        settings.requested_device,
        settings.dtype,
        module_importer=importer,
    )
    plan = loaders.validate_knowledge_base(entry, settings.profile)
    embedding = loaders.load_embedding(plan.embedding_identity, resolved_device)
    _validate_loaded_embedding_dimension(embedding, plan.embedding_identity)
    reranker = loaders.load_reranker(settings.profile, resolved_device)
    generator = loaders.load_generator(settings.profile, resolved_device)
    shared = SharedInferenceResources(
        profile=settings.profile,
        resolved_device=resolved_device,
        embedding_model=embedding,
        reranker_scorer=reranker,
        generator=generator,
        prompt_serializer=_prompt_serializer(settings.profile, generator),
        settings=settings,
    )
    service = loaders.build_knowledge_base_service(plan, shared)
    if not isinstance(service, KnowledgeBaseService):
        raise TypeError("production service builder returned an invalid service")
    return service.answer_service


def production_runtime_loaders(
    settings: ResolvedRuntimeSettings,
    *,
    module_importer: Callable[[str, str], Any] | None = None,
) -> RuntimeLoaders:
    """Create production loaders while keeping all heavyweight imports lazy."""
    importer = module_importer or _import_runtime_module
    return RuntimeLoaders(
        validate_knowledge_base=_validate_knowledge_base,
        load_embedding=lambda identity, device: _load_embedding_model(
            identity,
            device,
            module_importer=importer,
        ),
        load_reranker=lambda profile, device: _load_reranker(
            settings,
            device,
        ),
        load_generator=lambda profile, device: _load_generator(
            settings,
            device,
        ),
        build_knowledge_base_service=_build_knowledge_base_service,
    )


def inspect_data_root(data_root: Path) -> DataInspection:
    """Inspect baseline corpus and vector artifacts without loading models."""
    paths = artifact_paths(data_root)
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
    chunk_count = _safe_chunk_count(paths.processed_jsonl, "processed JSONL", errors)
    metadata_count = _safe_chunk_count(paths.metadata_path, "metadata JSONL", errors)
    manifest = _safe_manifest(paths.manifest_path, errors)
    model_name = _manifest_string(manifest, "model_name", errors)
    manifest_count = _manifest_positive_integer(manifest, "chunk_count", errors)
    manifest_dimension = _manifest_positive_integer(
        manifest, "embedding_dimension", errors
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
        embedding_model_name=model_name,
        errors=tuple(errors),
    )


def artifact_paths(data_root: Path) -> ArtifactPaths:
    """Resolve legacy or generic baseline artifact paths."""
    from python_doc_rag.dataset_layout import resolve_dataset_artifacts

    resolved = resolve_dataset_artifacts(data_root)
    return ArtifactPaths(
        data_root=resolved.data_root,
        processed_jsonl=resolved.processed_jsonl,
        index_path=resolved.index_path,
        metadata_path=resolved.metadata_path,
        manifest_path=resolved.index_manifest_path,
    )


@dataclass(frozen=True, slots=True)
class _SingleKnowledgeBaseConfig:
    id: str
    display_name: str
    data_root: Path


def _validate_knowledge_base(
    entry: KnowledgeBaseConfigProtocol,
    profile: RuntimeProfile,
) -> KnowledgeBasePlan:
    inspection = inspect_data_root(entry.data_root)
    if not inspection.succeeded:
        raise ArtifactValidationError(
            f"knowledge base {entry.id} のbaseline artifactが不正です:\n"
            + "\n".join(f"- {error}" for error in inspection.errors)
        )
    paths = artifact_paths(entry.data_root)
    profile_paths = profile_artifact_paths(profile, entry.data_root)
    profile_validation = validate_profile_artifacts(
        profile,
        entry.data_root,
        expected_chunk_count=inspection.metadata_count,
        baseline_metadata_path=(
            paths.metadata_path if profile.embedding_model_key is not None else None
        ),
    )
    if not profile_validation.succeeded:
        raise ArtifactValidationError(
            f"knowledge base {entry.id} のprofile artifactが不正です:\n"
            + "\n".join(f"- {error}" for error in profile_validation.errors)
        )
    manifest = _read_json_object(paths.manifest_path)
    _validate_baseline_hashes(paths, manifest)
    identity = _embedding_identity(profile, manifest)
    from python_doc_rag.dataset_layout import resolve_dataset_artifacts

    dataset = resolve_dataset_artifacts(entry.data_root).dataset_manifest
    dataset_name = (
        dataset.dataset_name
        if dataset is not None
        else "Python 3.13 Japanese documentation"
    )
    return KnowledgeBasePlan(
        id=entry.id,
        display_name=entry.display_name,
        data_root=paths.data_root,
        dataset_name=dataset_name,
        embedding_identity=identity,
        artifacts=paths,
        profile_artifacts=profile_paths,
    )


def _embedding_identity(
    profile: RuntimeProfile,
    baseline_manifest: Mapping[str, Any],
) -> EmbeddingIdentity:
    if profile.embedding_model_key is not None:
        from python_doc_rag.embedding_models import retrieval_embedding_spec

        spec = retrieval_embedding_spec(profile.embedding_model_key)
        return EmbeddingIdentity(
            model_name=spec.model_name,
            model_revision=spec.revision,
            embedding_dimension=spec.embedding_dimension,
            normalized_embeddings=spec.normalize_embeddings,
            query_prefix=spec.query_prefix,
            document_prefix=spec.document_prefix,
            trust_remote_code=spec.trust_remote_code,
        )
    model_name = baseline_manifest.get("model_name")
    dimension = baseline_manifest.get("embedding_dimension")
    normalized = baseline_manifest.get("normalized_embeddings")
    revision = baseline_manifest.get("model_revision")
    query_prefix_value = baseline_manifest.get("query_prefix")
    document_prefix_value = baseline_manifest.get("document_prefix")
    trust_remote_code_value = baseline_manifest.get("trust_remote_code")
    query_prefix = "" if query_prefix_value is None else query_prefix_value
    document_prefix = (
        "" if document_prefix_value is None else document_prefix_value
    )
    trust_remote_code = (
        False if trust_remote_code_value is None else trust_remote_code_value
    )
    if not isinstance(model_name, str) or not model_name.strip():
        raise ArtifactValidationError("baseline manifest model_name is invalid")
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
        raise ArtifactValidationError("baseline manifest dimension is invalid")
    if normalized is not True:
        raise ArtifactValidationError("baseline embeddings must be normalized")
    if revision is not None and (
        not isinstance(revision, str) or not revision.strip()
    ):
        raise ArtifactValidationError("baseline model revision is invalid")
    if not isinstance(query_prefix, str) or not isinstance(document_prefix, str):
        raise ArtifactValidationError("baseline embedding prefixes are invalid")
    if trust_remote_code is not False:
        raise ArtifactValidationError("trust_remote_code must be false")
    return EmbeddingIdentity(
        model_name=model_name,
        model_revision=revision,
        embedding_dimension=dimension,
        normalized_embeddings=True,
        query_prefix=query_prefix,
        document_prefix=document_prefix,
        trust_remote_code=False,
    )


def _shared_embedding_identity(
    plans: tuple[KnowledgeBasePlan, ...],
) -> EmbeddingIdentity:
    if not plans:
        raise ArtifactValidationError("knowledge baseが1件もありません。")
    expected = plans[0].embedding_identity
    mismatches = [plan.id for plan in plans[1:] if plan.embedding_identity != expected]
    if mismatches:
        raise ArtifactValidationError(
            "knowledge base間でEmbedding仕様を共有できません: "
            + ", ".join(mismatches)
        )
    return expected


def _resolve_runtime_device(
    requested: str,
    dtype: str,
    *,
    module_importer: Callable[[str, str], Any],
) -> str:
    torch = module_importer(
        "torch",
        "PyTorchが必要です。inference依存関係をインストールしてください。",
    )
    from python_doc_rag.transformers_generation import resolve_device

    if requested == "cuda" and not torch.cuda.is_available():
        raise InferenceLoadError(
            "CUDAを利用できません。GPU環境を確認するか、device=cpuを指定してください。"
        )
    actual = resolve_device(requested, torch_module=torch)
    if actual == "cpu" and dtype in {"float16", "bfloat16"}:
        raise InferenceLoadError(
            f"CPUでは--dtype {dtype}を使用できません。"
            "--dtype autoまたはfloat32を指定してください。"
        )
    return actual


def _load_embedding_model(
    identity: EmbeddingIdentity,
    device: str,
    *,
    module_importer: Callable[[str, str], Any],
) -> Any:
    sentence_transformers = module_importer(
        "sentence_transformers",
        "Sentence Transformersが必要です。inference依存関係を"
        "インストールしてください。",
    )
    options: dict[str, Any] = {
        "device": device,
        "trust_remote_code": False,
    }
    if identity.model_revision is not None:
        options["revision"] = identity.model_revision
    try:
        return sentence_transformers.SentenceTransformer(
            identity.model_name,
            **options,
        )
    except Exception as error:
        raise InferenceLoadError(
            "Embeddingモデルのロードに失敗しました。モデルcache、ネットワーク、"
            f"device設定を確認してください: {_exception_message(error)}"
        ) from error


def _load_reranker(
    settings: ResolvedRuntimeSettings,
    device: str,
) -> Any | None:
    if settings.reranker_model_key is None:
        return None
    from python_doc_rag.reranking import (
        CrossEncoderPairScorer,
        model_spec_for_key,
    )

    spec = model_spec_for_key(settings.reranker_model_key)
    try:
        return CrossEncoderPairScorer.from_pretrained(
            spec,
            device=device,
            max_length=_required_positive(
                settings.reranker_max_length,
                "reranker max length",
            ),
        )
    except Exception as error:
        raise InferenceLoadError(
            "local rerankerをロードできません。モデルcache、ネットワーク、"
            f"device設定を確認してください: {_exception_message(error)}"
        ) from error


def _load_generator(
    settings: ResolvedRuntimeSettings,
    device: str,
) -> Any:
    from python_doc_rag.transformers_generation import TransformersAnswerGenerator

    generation_options: dict[str, Any] = {}
    sampling_seed: int | None = None
    if settings.profile.generation_model_key is not None:
        from python_doc_rag.generator_models import generator_model_spec

        spec = generator_model_spec(settings.profile.generation_model_key)
        generation_options = spec.generation_options()
        sampling_seed = spec.sampling_seed
    try:
        return TransformersAnswerGenerator.from_pretrained(
            settings.generation_model,
            revision=settings.generation_model_revision,
            device=device,
            dtype=settings.dtype,
            max_input_tokens=settings.max_input_tokens,
            trust_remote_code=False,
            generation_kwargs=generation_options,
            sampling_seed=sampling_seed,
        )
    except Exception as error:
        raise InferenceLoadError(
            "回答生成モデルのロードに失敗しました。モデルcache、ネットワーク、"
            f"device、dtypeを確認してください: {_exception_message(error)}"
        ) from error


def _prompt_serializer(profile: RuntimeProfile, generator: Any) -> Any:
    from python_doc_rag.generation import ChatTemplatePromptSerializer

    template_options: dict[str, object] = {}
    if profile.generation_model_key is not None:
        from python_doc_rag.generator_models import generator_model_spec

        template_options = generator_model_spec(
            profile.generation_model_key
        ).template_options()
    return ChatTemplatePromptSerializer(
        generator.tokenizer,
        template_kwargs=template_options,
    )


def _build_knowledge_base_service(
    plan: KnowledgeBasePlan,
    shared: SharedInferenceResources,
) -> KnowledgeBaseService:
    from python_doc_rag.pipeline import RagPipeline
    from python_doc_rag.reranking import RerankingRetriever
    from python_doc_rag.retrieval import (
        BM25Retriever,
        CodeAwareNgramTokenizer,
        ReciprocalRankFusionRetriever,
        VectorIndexRetriever,
    )
    from python_doc_rag.vector_store import load_chunks_jsonl, load_vector_index

    settings = shared.settings
    if settings is None:
        raise RuntimeError("shared runtime settings are missing")
    paths = _require_type(plan.artifacts, ArtifactPaths, "artifact paths")
    profile_paths = _require_type(
        plan.profile_artifacts,
        ProfileArtifactPaths,
        "profile artifact paths",
    )
    vector_index_path = paths.index_path
    vector_metadata_path = paths.metadata_path
    if settings.profile.embedding_model_key is not None:
        vector_index_path = _required_path(
            profile_paths.embedding_index_path,
            "embedding index",
        )
        vector_metadata_path = _required_path(
            profile_paths.embedding_metadata_path,
            "embedding metadata",
        )
    try:
        vector_index = load_vector_index(
            vector_index_path,
            vector_metadata_path,
            embedding_model=shared.embedding_model,
            query_prefix=plan.embedding_identity.query_prefix,
        )
        retriever: Any = VectorIndexRetriever(vector_index)
        if settings.retriever == "technical-field":
            retriever = _technical_retriever(
                retriever,
                vector_metadata_path,
                _required_path(profile_paths.symbol_index_path, "symbol index"),
                settings,
            )
        elif settings.retriever == "hybrid":
            bm25 = BM25Retriever(
                load_chunks_jsonl(paths.metadata_path),
                tokenizer=CodeAwareNgramTokenizer(_HYBRID_NGRAM_SIZES),
            )
            retriever = ReciprocalRankFusionRetriever(
                [retriever, bm25],
                rrf_k=_HYBRID_RRF_K,
                candidate_k=_HYBRID_CANDIDATE_K,
            )
        if settings.reranker_model_key is not None:
            if shared.reranker_scorer is None:
                raise RuntimeError("shared reranker scorer is missing")
            retriever = RerankingRetriever(
                retriever,
                shared.reranker_scorer,
                candidate_k=_required_positive(
                    settings.reranker_candidate_k,
                    "reranker candidate k",
                ),
                batch_size=_required_positive(
                    settings.reranker_batch_size,
                    "reranker batch size",
                ),
            )
    except Exception as error:
        raise ArtifactValidationError(
            f"knowledge base {plan.id} の検索artifactをロードできません: "
            f"{_exception_message(error)}"
        ) from error
    timing_retriever = TimingRetriever(retriever)
    pipeline = RagPipeline(
        retriever=timing_retriever,
        generator=shared.generator,
        tokenizer=shared.generator.prompt_tokenizer,
        prompt_serializer=shared.prompt_serializer,
        config=GenerationConfig(
            retrieval_limit=settings.top_k,
            max_prompt_tokens=settings.max_input_tokens,
            max_new_tokens=settings.max_new_tokens,
        ),
        generation_contract=generation_contract_for_mode(
            settings.answer_mode,
            document_scope=plan.display_name,
        ),
    )
    answer_service = RagService(
        pipeline=pipeline,
        timing_retriever=timing_retriever,
        generator=shared.generator,
    )
    return KnowledgeBaseService(
        id=plan.id,
        display_name=plan.display_name,
        dataset_name=plan.dataset_name,
        data_root=plan.data_root,
        answer_service=answer_service,
        retriever=retriever,
        artifacts=paths,
        profile_artifacts=profile_paths,
    )


def _technical_retriever(
    dense: Any,
    metadata_path: Path,
    symbol_path: Path,
    settings: ResolvedRuntimeSettings,
) -> Any:
    from python_doc_rag.technical_retrieval import (
        FieldBM25Retriever,
        SymbolRetriever,
        WeightedRankFusionRetriever,
        load_symbol_sidecar,
    )
    from python_doc_rag.vector_store import load_chunks_jsonl

    chunks = load_chunks_jsonl(metadata_path)
    symbols = load_symbol_sidecar(chunks, symbol_path)
    return WeightedRankFusionRetriever(
        [
            ("identifiers", SymbolRetriever(chunks, symbols), 1.0),
            (
                "section_title",
                FieldBM25Retriever(chunks, field="section_title"),
                1.0,
            ),
            ("page_title", FieldBM25Retriever(chunks, field="page_title"), 1.0),
            ("body_dense", dense, 1.0),
            ("body_lexical", FieldBM25Retriever(chunks, field="body"), 1.0),
        ],
        rrf_k=_required_positive(settings.field_rrf_k, "field RRF k"),
        candidate_k=_required_positive(
            settings.field_candidate_k,
            "field candidate k",
        ),
    )


def _validate_baseline_hashes(
    paths: ArtifactPaths,
    manifest: Mapping[str, Any],
) -> None:
    checks = (
        (paths.index_path, manifest.get("index_sha256"), "index"),
        (paths.metadata_path, manifest.get("metadata_sha256"), "metadata"),
        (
            paths.processed_jsonl,
            manifest.get("input_jsonl_sha256"),
            "processed JSONL",
        ),
    )
    for path, expected, label in checks:
        if expected is None:
            continue
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)
        ):
            raise ArtifactValidationError(
                f"baseline manifest {label} SHA-256 is invalid"
            )
        actual = _sha256(path)
        if actual != expected:
            raise ArtifactValidationError(
                f"baseline {label} SHA-256 does not match its manifest"
            )


def _validate_loaded_embedding_dimension(
    model: Any,
    identity: EmbeddingIdentity,
) -> None:
    getter = getattr(model, "get_embedding_dimension", None)
    if not callable(getter):
        getter = getattr(model, "get_sentence_embedding_dimension", None)
    if not callable(getter):
        return
    dimension = getter()
    if dimension is not None and int(dimension) != identity.embedding_dimension:
        raise InferenceLoadError(
            "Embeddingモデルの次元が検証済みartifactと一致しません: "
            f"{dimension} != {identity.embedding_dimension}"
        )


def _generation_cursor(generator: Any) -> int:
    cursor = getattr(generator, "generation_cursor", None)
    if isinstance(cursor, int) and not isinstance(cursor, bool):
        return cursor
    return len(generator.generation_history)


def _generation_metrics_since(generator: Any, cursor: int) -> tuple[Any, ...]:
    reader = getattr(generator, "generation_metrics_since", None)
    if callable(reader):
        return tuple(reader(cursor))
    return tuple(generator.generation_history[cursor:])


def _bound_server_generation_history(generator: Any) -> None:
    setter = getattr(generator, "set_generation_history_limit", None)
    if callable(setter):
        setter(_SERVER_GENERATION_HISTORY_LIMIT)


def _bound_server_retrieval_history(service: Any) -> None:
    answer_service = getattr(service, "answer_service", None)
    setter = getattr(answer_service, "set_retrieval_history_limit", None)
    if callable(setter):
        setter(_SERVER_GENERATION_HISTORY_LIMIT)


def _require_artifact_files(data_root: Path, profile: RuntimeProfile) -> None:
    paths = artifact_paths(data_root)
    if not paths.data_root.is_dir():
        raise ArtifactValidationError(f"data-rootが存在しません: {paths.data_root}")
    required = (
        paths.index_path,
        paths.metadata_path,
        paths.manifest_path,
        *profile_artifact_paths(profile, data_root).required_paths,
    )
    missing = tuple(path for path in required if not path.is_file())
    if missing:
        raise ArtifactValidationError(
            "質問応答に必要なartifactが不足しています:\n"
            + "\n".join(f"- {path}" for path in dict.fromkeys(missing))
        )


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
    except ValueError as error:
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

        index = _import_faiss().read_index(str(path))
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
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
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


def _required_path(path: Path | None, label: str) -> Path:
    if path is None:
        raise ArtifactValidationError(f"profileに{label} pathがありません。")
    return path


def _required_positive(value: int | None, label: str) -> int:
    if value is None or value < 1:
        raise ValueError(f"{label} must be positive")
    return value


def _require_type(value: Any, expected: type[Any], label: str) -> Any:
    if not isinstance(value, expected):
        raise TypeError(f"{label} are missing from the validated plan")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _import_runtime_module(name: str, message: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as error:
        raise InferenceLoadError(message) from error


def _exception_message(error: BaseException) -> str:
    message = str(error).strip()
    return message or error.__class__.__name__
