import asyncio
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from python_doc_rag.application import (
    AnswerExecution,
    AnswerServiceError,
    ArtifactValidationError,
    EmbeddingIdentity,
    KnowledgeBaseNotFoundError,
    KnowledgeBasePlan,
    MultiKnowledgeBaseRuntime,
    RagService,
    RuntimeLoaders,
    TimingRetriever,
    build_multi_knowledge_base_runtime,
)
from python_doc_rag.profiles import RuntimeProfile, runtime_profile


@dataclass(frozen=True, slots=True)
class FakeKnowledgeBaseConfig:
    id: str
    display_name: str
    data_root: Path


@dataclass(frozen=True, slots=True)
class FakeServiceConfig:
    profile: RuntimeProfile
    device: str
    knowledge_bases: tuple[FakeKnowledgeBaseConfig, ...]


@dataclass(slots=True)
class ConcurrencyProbe:
    active: int = 0
    maximum_active: int = 0

    def enter(self) -> None:
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)

    def leave(self) -> None:
        self.active -= 1


class FakeKnowledgeBaseService:
    def __init__(
        self,
        plan: KnowledgeBasePlan,
        *,
        probe: ConcurrencyProbe | None = None,
        delay_seconds: float = 0.0,
        started_event: threading.Event | None = None,
        release_event: threading.Event | None = None,
        failures: Mapping[str, Exception] | None = None,
    ) -> None:
        self.id = plan.id
        self.display_name = plan.display_name
        self.dataset_name = plan.dataset_name
        self.retriever = object()
        self.questions: list[str] = []
        self._probe = probe
        self._delay_seconds = delay_seconds
        self._started_event = started_event
        self._release_event = release_event
        self._failures = dict(failures or {})

    def answer(self, question: str) -> tuple[str, str]:
        self.questions.append(question)
        if self._probe is None:
            return self.id, question
        self._probe.enter()
        try:
            if self._started_event is not None:
                self._started_event.set()
            if self._release_event is not None:
                if not self._release_event.wait(timeout=2.0):
                    raise TimeoutError("test did not release the fake answer")
            else:
                time.sleep(self._delay_seconds)
            failure = self._failures.get(question)
            if failure is not None:
                raise failure
            return self.id, question
        finally:
            self._probe.leave()


@dataclass(frozen=True, slots=True)
class FakeGenerationMetric:
    """One deterministic fake generation measurement."""

    input_tokens: int
    generated_tokens: int
    elapsed_seconds: float


class FakeMonotonicGenerator:
    """Expose the bounded monotonic history contract used by RagService."""

    def __init__(self) -> None:
        self._history: list[FakeGenerationMetric] = []
        self._history_base = 0
        self._generation_count = 0
        self._history_limit: int | None = None

    @property
    def generation_cursor(self) -> int:
        """Return the total number of completed fake generations."""
        return self._generation_count

    @property
    def generation_history(self) -> tuple[FakeGenerationMetric, ...]:
        """Return only the retained fake generation measurements."""
        return tuple(self._history)

    @property
    def history_limit(self) -> int | None:
        """Return the limit applied by the server runtime builder."""
        return self._history_limit

    def generation_metrics_since(
        self,
        cursor: int,
    ) -> tuple[FakeGenerationMetric, ...]:
        """Return retained measurements after a monotonic cursor."""
        if cursor < self._history_base or cursor > self._generation_count:
            raise ValueError("fake generation cursor is outside retained history")
        return tuple(self._history[cursor - self._history_base :])

    def set_generation_history_limit(self, limit: int) -> None:
        """Bound retained fake measurements without resetting the cursor."""
        self._history_limit = limit
        self._trim_history()

    def record(self, *, input_tokens: int, generated_tokens: int) -> None:
        """Record one deterministic fake model call."""
        self._history.append(
            FakeGenerationMetric(
                input_tokens=input_tokens,
                generated_tokens=generated_tokens,
                elapsed_seconds=input_tokens / 1_000_000,
            )
        )
        self._generation_count += 1
        self._trim_history()

    def _trim_history(self) -> None:
        if self._history_limit is None:
            return
        overflow = len(self._history) - self._history_limit
        if overflow > 0:
            del self._history[:overflow]
            self._history_base += overflow


class FakeMetricsRetriever:
    """Return one KB-tagged result while TimingRetriever measures the call."""

    def __init__(self, knowledge_base_id: str) -> None:
        self._knowledge_base_id = knowledge_base_id

    def retrieve(self, question: str, *, limit: int) -> tuple[str, str, int]:
        """Return a deterministic result after a small measurable delay."""
        time.sleep(0.0001)
        return self._knowledge_base_id, question, limit


class FakeMetricsPipeline:
    """Exercise actual timing and generation cursors without loading models."""

    def __init__(
        self,
        *,
        knowledge_base_id: str,
        timing_retriever: TimingRetriever,
        generator: FakeMonotonicGenerator,
        retrieval_timings: dict[str, float],
    ) -> None:
        self._knowledge_base_id = knowledge_base_id
        self._timing_retriever = timing_retriever
        self._generator = generator
        self._retrieval_timings = retrieval_timings

    def answer(self, question: str) -> tuple[str, str]:
        """Record one retrieval and one or two question-specific generations."""
        self._timing_retriever.retrieve(question, limit=5)
        self._retrieval_timings[question] = self._timing_retriever.history[-1]
        question_index = int(question.rsplit("-", maxsplit=1)[-1])
        token_base = _metric_token_base(self._knowledge_base_id, question_index)
        self._generator.record(
            input_tokens=token_base,
            generated_tokens=token_base + 1,
        )
        if question_index % 7 == 0:
            self._generator.record(
                input_tokens=token_base + 2,
                generated_tokens=token_base + 3,
            )
        return self._knowledge_base_id, question


@dataclass(slots=True)
class FakeMetricsKnowledgeBaseService:
    """Expose an actual RagService and its per-KB timing state."""

    id: str
    display_name: str
    dataset_name: str
    answer_service: RagService
    timing_retriever: TimingRetriever
    retrieval_timings: dict[str, float]

    def answer(self, question: str) -> AnswerExecution:
        """Delegate one question to the actual RagService."""
        return self.answer_service.answer(question)


def _config(tmp_path: Path) -> FakeServiceConfig:
    return FakeServiceConfig(
        profile=runtime_profile("recommended-v2"),
        device="cpu",
        knowledge_bases=(
            FakeKnowledgeBaseConfig(
                id="python-docs",
                display_name="Python documentation",
                data_root=tmp_path / "python",
            ),
            FakeKnowledgeBaseConfig(
                id="uv-docs",
                display_name="uv documentation",
                data_root=tmp_path / "uv",
            ),
        ),
    )


def _embedding_identity(*, dimension: int = 1024) -> EmbeddingIdentity:
    return EmbeddingIdentity(
        model_name="BAAI/bge-m3",
        model_revision="pinned-revision",
        embedding_dimension=dimension,
        normalized_embeddings=True,
        query_prefix="",
        document_prefix="",
        trust_remote_code=False,
    )


def _plan(
    entry: FakeKnowledgeBaseConfig,
    *,
    identity: EmbeddingIdentity | None = None,
) -> KnowledgeBasePlan:
    return KnowledgeBasePlan(
        id=entry.id,
        display_name=entry.display_name,
        data_root=entry.data_root,
        dataset_name=f"dataset-{entry.id}",
        embedding_identity=identity or _embedding_identity(),
    )


def _metric_token_base(knowledge_base_id: str, question_index: int) -> int:
    """Return a distinct token range for each KB and question."""
    knowledge_base_offset = 1 if knowledge_base_id == "python-docs" else 10_000
    return knowledge_base_offset + question_index * 10


def _runtime_loaders(
    *,
    events: list[str] | None = None,
    validator: Callable[[Any, Any], KnowledgeBasePlan] | None = None,
    services: list[FakeKnowledgeBaseService] | None = None,
    shared_resources: list[Any] | None = None,
    probe: ConcurrencyProbe | None = None,
    delay_seconds: float = 0.0,
    started_event: threading.Event | None = None,
    release_event: threading.Event | None = None,
    failures: Mapping[str, Exception] | None = None,
) -> RuntimeLoaders:
    recorded = events if events is not None else []

    def validate(entry: Any, profile: Any) -> KnowledgeBasePlan:
        recorded.append(f"validate:{entry.id}:{profile.name}")
        if validator is not None:
            return validator(entry, profile)
        return _plan(entry)

    def load_embedding(identity: EmbeddingIdentity, device: str) -> object:
        recorded.append(f"embedding:{identity.model_name}:{device}")
        return object()

    def load_reranker(profile: Any, device: str) -> object:
        recorded.append(f"reranker:{profile.name}:{device}")
        return object()

    def load_generator(profile: Any, device: str) -> object:
        recorded.append(f"generator:{profile.name}:{device}")
        return object()

    def build_service(plan: KnowledgeBasePlan, shared: Any) -> Any:
        recorded.append(f"service:{plan.id}")
        if shared_resources is not None:
            shared_resources.append(shared)
        service = FakeKnowledgeBaseService(
            plan,
            probe=probe,
            delay_seconds=delay_seconds,
            started_event=started_event,
            release_event=release_event,
            failures=failures,
        )
        if services is not None:
            services.append(service)
        return service

    return RuntimeLoaders(
        validate_knowledge_base=validate,
        load_embedding=load_embedding,
        load_reranker=load_reranker,
        load_generator=load_generator,
        build_knowledge_base_service=build_service,
    )


def _metrics_runtime(
    tmp_path: Path,
) -> tuple[
    MultiKnowledgeBaseRuntime,
    FakeMonotonicGenerator,
    dict[str, FakeMetricsKnowledgeBaseService],
]:
    """Build a two-KB runtime with actual RagService timing boundaries."""
    generator = FakeMonotonicGenerator()
    services: dict[str, FakeMetricsKnowledgeBaseService] = {}

    def validate(
        entry: FakeKnowledgeBaseConfig,
        _profile: RuntimeProfile,
    ) -> KnowledgeBasePlan:
        return _plan(entry)

    def load_embedding(_identity: EmbeddingIdentity, _device: str) -> object:
        return object()

    def load_reranker(_profile: RuntimeProfile, _device: str) -> None:
        return None

    def load_generator(
        _profile: RuntimeProfile,
        _device: str,
    ) -> FakeMonotonicGenerator:
        return generator

    def build_service(
        plan: KnowledgeBasePlan,
        shared: Any,
    ) -> FakeMetricsKnowledgeBaseService:
        assert shared.generator is generator
        timing_retriever = TimingRetriever(FakeMetricsRetriever(plan.id))
        retrieval_timings: dict[str, float] = {}
        pipeline = FakeMetricsPipeline(
            knowledge_base_id=plan.id,
            timing_retriever=timing_retriever,
            generator=generator,
            retrieval_timings=retrieval_timings,
        )
        answer_service = RagService(
            pipeline=pipeline,
            timing_retriever=timing_retriever,
            generator=generator,
        )
        service = FakeMetricsKnowledgeBaseService(
            id=plan.id,
            display_name=plan.display_name,
            dataset_name=plan.dataset_name,
            answer_service=answer_service,
            timing_retriever=timing_retriever,
            retrieval_timings=retrieval_timings,
        )
        services[plan.id] = service
        return service

    runtime = build_multi_knowledge_base_runtime(
        _config(tmp_path),
        loaders=RuntimeLoaders(
            validate_knowledge_base=validate,
            load_embedding=load_embedding,
            load_reranker=load_reranker,
            load_generator=load_generator,
            build_knowledge_base_service=build_service,
        ),
    )
    return runtime, generator, services


def test_runtime_validates_all_kbs_before_loading_shared_models_once(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    services: list[FakeKnowledgeBaseService] = []
    shared_resources: list[Any] = []

    runtime = build_multi_knowledge_base_runtime(
        _config(tmp_path),
        loaders=_runtime_loaders(
            events=events,
            services=services,
            shared_resources=shared_resources,
        ),
    )

    assert events == [
        "validate:python-docs:recommended-v2",
        "validate:uv-docs:recommended-v2",
        "embedding:BAAI/bge-m3:cpu",
        "reranker:recommended-v2:cpu",
        "generator:recommended-v2:cpu",
        "service:python-docs",
        "service:uv-docs",
    ]
    assert runtime.ready
    assert runtime.profile_name == "recommended-v2"
    assert runtime.knowledge_base_count == 2
    assert services[0] is not services[1]
    assert services[0].retriever is not services[1].retriever
    assert shared_resources[0] is shared_resources[1]
    assert shared_resources[0].embedding_model is not None
    assert shared_resources[0].reranker_scorer is not None
    assert shared_resources[0].generator is not None


def test_runtime_validation_failure_prevents_every_model_load(
    tmp_path: Path,
) -> None:
    events: list[str] = []

    def reject_uv(entry: FakeKnowledgeBaseConfig, _profile: Any) -> KnowledgeBasePlan:
        if entry.id == "uv-docs":
            raise ArtifactValidationError("invalid uv artifacts")
        return _plan(entry)

    with pytest.raises(ArtifactValidationError, match="invalid uv artifacts"):
        build_multi_knowledge_base_runtime(
            _config(tmp_path),
            loaders=_runtime_loaders(events=events, validator=reject_uv),
        )

    assert events == [
        "validate:python-docs:recommended-v2",
        "validate:uv-docs:recommended-v2",
    ]


def test_runtime_rejects_incompatible_embedding_identity_before_model_load(
    tmp_path: Path,
) -> None:
    events: list[str] = []

    def incompatible(
        entry: FakeKnowledgeBaseConfig,
        _profile: Any,
    ) -> KnowledgeBasePlan:
        dimension = 1024 if entry.id == "python-docs" else 768
        return _plan(entry, identity=_embedding_identity(dimension=dimension))

    with pytest.raises(ArtifactValidationError, match="[Ee]mbedding"):
        build_multi_knowledge_base_runtime(
            _config(tmp_path),
            loaders=_runtime_loaders(events=events, validator=incompatible),
        )

    assert events == [
        "validate:python-docs:recommended-v2",
        "validate:uv-docs:recommended-v2",
    ]


def test_runtime_registry_is_immutable_and_preserves_configuration_order(
    tmp_path: Path,
) -> None:
    runtime = build_multi_knowledge_base_runtime(
        _config(tmp_path),
        loaders=_runtime_loaders(),
    )

    assert tuple(runtime.knowledge_bases) == ("python-docs", "uv-docs")
    assert runtime.get_knowledge_base("python-docs").id == "python-docs"
    with pytest.raises(TypeError):
        runtime.knowledge_bases["extra"] = object()  # type: ignore[index]
    with pytest.raises(KnowledgeBaseNotFoundError):
        runtime.get_knowledge_base("missing")


def test_runtime_routes_alternating_answers_to_only_the_selected_kb(
    tmp_path: Path,
) -> None:
    services: list[FakeKnowledgeBaseService] = []
    runtime = build_multi_knowledge_base_runtime(
        _config(tmp_path),
        loaders=_runtime_loaders(services=services),
    )

    async def alternate() -> tuple[tuple[str, str], ...]:
        return (
            await runtime.answer("python-docs", "python one"),
            await runtime.answer("uv-docs", "uv one"),
            await runtime.answer("python-docs", "python two"),
        )

    results = asyncio.run(alternate())

    assert results == (
        ("python-docs", "python one"),
        ("uv-docs", "uv one"),
        ("python-docs", "python two"),
    )
    assert services[0].questions == ["python one", "python two"]
    assert services[1].questions == ["uv one"]


def test_runtime_serializes_answers_across_knowledge_bases(tmp_path: Path) -> None:
    probe = ConcurrencyProbe()
    runtime = build_multi_knowledge_base_runtime(
        _config(tmp_path),
        loaders=_runtime_loaders(probe=probe, delay_seconds=0.03),
    )

    async def send_concurrently() -> tuple[tuple[str, str], ...]:
        first, second, third = await asyncio.gather(
            runtime.answer("python-docs", "one"),
            runtime.answer("uv-docs", "two"),
            runtime.answer("python-docs", "three"),
        )
        return first, second, third

    results = asyncio.run(send_concurrently())

    assert {result[1] for result in results} == {"one", "two", "three"}
    assert probe.active == 0
    assert probe.maximum_active == 1


def test_repeatedly_cancelled_requests_hold_slot_until_each_worker_finishes(
    tmp_path: Path,
) -> None:
    probe = ConcurrencyProbe()
    started = threading.Event()
    release = threading.Event()
    runtime = build_multi_knowledge_base_runtime(
        _config(tmp_path),
        loaders=_runtime_loaders(
            probe=probe,
            started_event=started,
            release_event=release,
        ),
    )

    async def cancel_then_follow_repeatedly() -> None:
        loop_errors: list[dict[str, Any]] = []
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(lambda _loop, context: loop_errors.append(context))
        for request_index in range(5):
            started.clear()
            release.clear()
            cancelled = asyncio.create_task(
                runtime.answer("python-docs", f"cancelled-{request_index}")
            )
            assert await asyncio.to_thread(started.wait, 1.0)
            cancelled.cancel()
            await asyncio.sleep(0)
            cancelled.cancel()
            cancelled.cancel()
            following = asyncio.create_task(
                runtime.answer("uv-docs", f"following-{request_index}")
            )
            await asyncio.sleep(0.02)
            assert probe.active == 1
            assert probe.maximum_active == 1
            assert not following.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await cancelled
            assert await following == (
                "uv-docs",
                f"following-{request_index}",
            )

        await asyncio.sleep(0)
        current = asyncio.current_task()
        pending = [
            task
            for task in asyncio.all_tasks()
            if task is not current and not task.done()
        ]
        assert pending == []
        assert loop_errors == []

    asyncio.run(cancel_then_follow_repeatedly())
    assert probe.active == 0
    assert probe.maximum_active == 1


def test_cancelled_request_collects_worker_failure_before_releasing_slot(
    tmp_path: Path,
) -> None:
    probe = ConcurrencyProbe()
    started = threading.Event()
    release = threading.Event()
    runtime = build_multi_knowledge_base_runtime(
        _config(tmp_path),
        loaders=_runtime_loaders(
            probe=probe,
            started_event=started,
            release_event=release,
            failures={
                "cancelled-failure": RuntimeError(
                    "cancelled worker failed /workspace/private/model"
                )
            },
        ),
    )

    async def cancel_failing_worker_then_follow() -> None:
        loop_errors: list[dict[str, Any]] = []
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(lambda _loop, context: loop_errors.append(context))
        cancelled = asyncio.create_task(
            runtime.answer("python-docs", "cancelled-failure")
        )
        assert await asyncio.to_thread(started.wait, 1.0)
        cancelled.cancel()
        await asyncio.sleep(0)
        cancelled.cancel()
        following = asyncio.create_task(
            runtime.answer("uv-docs", "after-cancelled-failure")
        )
        await asyncio.sleep(0.02)
        assert probe.active == 1
        assert probe.maximum_active == 1
        assert not following.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        assert await following == ("uv-docs", "after-cancelled-failure")
        await asyncio.sleep(0)
        current = asyncio.current_task()
        pending = [
            task
            for task in asyncio.all_tasks()
            if task is not current and not task.done()
        ]
        assert pending == []
        assert loop_errors == []

    asyncio.run(cancel_failing_worker_then_follow())
    assert probe.active == 0
    assert probe.maximum_active == 1


@pytest.mark.parametrize(
    "failure_type",
    [
        pytest.param(AnswerServiceError, id="expected"),
        pytest.param(LookupError, id="unexpected"),
    ],
)
def test_answer_failure_releases_semaphore_for_next_request(
    tmp_path: Path,
    failure_type: type[Exception],
) -> None:
    probe = ConcurrencyProbe()
    runtime = build_multi_knowledge_base_runtime(
        _config(tmp_path),
        loaders=_runtime_loaders(
            probe=probe,
            failures={"failing": failure_type("fake answer failure")},
        ),
    )

    async def fail_then_follow() -> tuple[str, str]:
        with pytest.raises(failure_type):
            await runtime.answer("python-docs", "failing")
        return await asyncio.wait_for(
            runtime.answer("uv-docs", "following"),
            timeout=1.0,
        )

    assert asyncio.run(fail_then_follow()) == ("uv-docs", "following")
    assert probe.active == 0
    assert probe.maximum_active == 1


def test_long_lived_runtime_isolates_metrics_after_history_trimming(
    tmp_path: Path,
) -> None:
    runtime, generator, services = _metrics_runtime(tmp_path)
    requests: list[tuple[str, str, int]] = []
    for request_index in range(140):
        knowledge_base_id = "python-docs" if request_index % 2 == 0 else "uv-docs"
        question_index = request_index // 2
        requests.append(
            (
                knowledge_base_id,
                f"question-{question_index}",
                question_index,
            )
        )

    async def answer_all() -> tuple[AnswerExecution, ...]:
        executions = await asyncio.gather(
            *(
                runtime.answer(knowledge_base_id, question)
                for knowledge_base_id, question, _question_index in requests
            )
        )
        return tuple(executions)

    executions = asyncio.run(answer_all())

    for execution, request in zip(executions, requests, strict=True):
        knowledge_base_id, question, question_index = request
        token_base = _metric_token_base(knowledge_base_id, question_index)
        generation_calls = 2 if question_index % 7 == 0 else 1
        expected_input_tokens = token_base
        expected_generated_tokens = token_base + 1
        if generation_calls == 2:
            expected_input_tokens += token_base + 2
            expected_generated_tokens += token_base + 3
        retrieval_seconds = services[knowledge_base_id].retrieval_timings[question]

        assert execution.answer == (knowledge_base_id, question)
        assert execution.retrieval_seconds == pytest.approx(retrieval_seconds)
        assert execution.generation_seconds == pytest.approx(
            expected_input_tokens / 1_000_000
        )
        assert execution.total_seconds >= execution.retrieval_seconds
        assert execution.input_tokens == expected_input_tokens
        assert execution.generated_tokens == expected_generated_tokens
        assert execution.generation_calls == generation_calls

    assert generator.history_limit == 64
    assert generator.generation_cursor == 160
    assert len(generator.generation_history) == 64
    assert tuple(services) == ("python-docs", "uv-docs")
    for service in services.values():
        assert service.timing_retriever.retrieval_cursor == 70
        assert len(service.timing_retriever.history) == 64
        assert len(service.retrieval_timings) == 70


def test_timing_retriever_uses_monotonic_cursor_with_bounded_history() -> None:
    retriever = FakeRetrieverForTiming()
    timing = TimingRetriever(retriever)
    timing.set_history_limit(2)
    cursor = timing.retrieval_cursor

    timing.retrieve("one", limit=1)
    assert len(timing.retrieval_metrics_since(cursor)) == 1
    timing.retrieve("two", limit=1)
    timing.retrieve("three", limit=1)

    assert timing.retrieval_cursor == 3
    assert len(timing.history) == 2
    assert len(timing.retrieval_metrics_since(1)) == 2
    with pytest.raises(ValueError, match="outside retained history"):
        timing.retrieval_metrics_since(0)


class FakeRetrieverForTiming:
    def retrieve(self, question: str, *, limit: int) -> tuple[str, int]:
        return question, limit
