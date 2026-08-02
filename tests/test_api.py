import asyncio
import importlib.util
import json
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

from python_doc_rag.api import create_app
from python_doc_rag.application import (
    AnswerExecution,
    AnswerServiceError,
    KnowledgeBaseNotFoundError,
)
from python_doc_rag.models import (
    AbstainedAnswer,
    CitationSource,
    CitedAnswer,
    SearchChunk,
)

_API_AVAILABLE = (
    importlib.util.find_spec("fastapi") is not None
    and importlib.util.find_spec("httpx2") is not None
)
_requires_api = pytest.mark.skipif(
    not _API_AVAILABLE,
    reason="API integration dependencies are not installed",
)


@dataclass(frozen=True, slots=True)
class FakeKnowledgeBase:
    id: str
    display_name: str
    dataset_name: str
    data_root: Path
    internal_manifest_path: Path


class FakeKnowledgeBaseNotFoundError(KnowledgeBaseNotFoundError):
    def __init__(self) -> None:
        Exception.__init__(self, "unknown knowledge base /workspace/private")


class FakeAnswerServiceError(AnswerServiceError):
    def __init__(self) -> None:
        Exception.__init__(self, "generation failed /workspace/private/model")


class FakeRuntime:
    """Select deterministic outcomes without importing or loading ML models."""

    def __init__(self, *, ready: bool = True) -> None:
        self.ready = ready
        self.profile_name = "recommended-v2"
        services = (
            FakeKnowledgeBase(
                id="python-docs",
                display_name="Python 3.13 日本語公式ドキュメント",
                dataset_name="Python 3.13 Japanese Documentation",
                data_root=Path("/workspace/private/python"),
                internal_manifest_path=Path("/workspace/private/python/manifest.json"),
            ),
            FakeKnowledgeBase(
                id="uv-docs",
                display_name="uv Documentation",
                dataset_name="uv documentation smoke",
                data_root=Path("/workspace/private/uv"),
                internal_manifest_path=Path("/workspace/private/uv/manifest.json"),
            ),
        )
        self.knowledge_bases = MappingProxyType(
            {service.id: service for service in services}
        )
        self.calls: list[tuple[str, str]] = []
        self.failure: Exception | None = None
        self.abstain = False
        self.source_url_override: str | None = None

    @property
    def knowledge_base_count(self) -> int:
        return len(self.knowledge_bases)

    async def answer(
        self,
        knowledge_base_id: str,
        question: str,
    ) -> AnswerExecution:
        if knowledge_base_id not in self.knowledge_bases:
            raise FakeKnowledgeBaseNotFoundError
        if self.failure is not None:
            raise self.failure
        self.calls.append((knowledge_base_id, question))
        answer = (
            _abstained_answer(knowledge_base_id)
            if self.abstain
            else _cited_answer(
                knowledge_base_id,
                question,
                source_url=self.source_url_override,
            )
        )
        return AnswerExecution(
            answer=answer,
            retrieval_seconds=0.125,
            generation_seconds=0.5,
            total_seconds=0.75,
            input_tokens=1234,
            generated_tokens=98,
            generation_calls=1,
        )


class SlowSerializedRuntime(FakeRuntime):
    """Mirror the production runtime's global answer semaphore."""

    def __init__(self) -> None:
        super().__init__()
        self._semaphore = asyncio.Semaphore(1)
        self.first_started = asyncio.Event()
        self.release = asyncio.Event()
        self.active = 0
        self.max_active = 0

    async def answer(
        self,
        knowledge_base_id: str,
        question: str,
    ) -> AnswerExecution:
        async with self._semaphore:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.first_started.set()
            try:
                await self.release.wait()
                return await super().answer(knowledge_base_id, question)
            finally:
                self.active -= 1


def _chunk(knowledge_base_id: str) -> SearchChunk:
    return SearchChunk(
        text=f"{knowledge_base_id} evidence",
        page_title=f"{knowledge_base_id} page",
        section_title=f"{knowledge_base_id} section",
        source_url=f"https://retrieved.example.test/{knowledge_base_id}/internal",
        category=knowledge_base_id,
        chunk_index=0,
        start_index=0,
    )


def _cited_answer(
    knowledge_base_id: str,
    question: str,
    *,
    source_url: str | None = None,
) -> CitedAnswer:
    chunk = _chunk(knowledge_base_id)
    return CitedAnswer(
        answer_text=f"{question}への回答 [S1]",
        sources=(
            CitationSource(
                label="S1",
                page_title=chunk.page_title,
                section_title=chunk.section_title,
                url=(
                    source_url
                    if source_url is not None
                    else f"https://trusted.example.test/{knowledge_base_id}/answer"
                ),
            ),
        ),
        retrieved_chunks=(chunk,),
        generation_attempts=1,
    )


def _abstained_answer(knowledge_base_id: str) -> AbstainedAnswer:
    return AbstainedAnswer(
        reason_code="insufficient_evidence",
        retrieved_chunks=(_chunk(knowledge_base_id),),
        generation_attempts=1,
    )


@contextmanager
def _client(runtime: FakeRuntime) -> Iterator[Any]:
    from fastapi.testclient import TestClient

    with TestClient(create_app(runtime_factory=lambda: runtime)) as client:
        yield client


def test_api_module_import_does_not_import_web_or_ml_dependencies() -> None:
    code = """
import builtins

original_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name.split('.', maxsplit=1)[0] in {
        'fastapi',
        'uvicorn',
        'torch',
        'transformers',
        'sentence_transformers',
    }:
        raise ModuleNotFoundError(name)
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
import python_doc_rag.api
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_create_app_requires_exactly_one_runtime_source() -> None:
    if not _API_AVAILABLE:
        pytest.skip("API integration dependencies are not installed")
    with pytest.raises(ValueError, match="runtime_factory or service_config"):
        create_app()
    with pytest.raises(ValueError, match="mutually exclusive"):
        create_app(
            runtime_factory=FakeRuntime,
            service_config=Path("service.toml"),
        )


@_requires_api
class TestFastApiAdapter:
    def test_health_is_live_while_readiness_is_unavailable(self) -> None:
        with _client(FakeRuntime(ready=False)) as client:
            health = client.get("/healthz")
            readiness = client.get("/readyz")

        assert health.status_code == 200
        assert health.json() == {"status": "ok"}
        assert readiness.status_code == 503
        assert readiness.json() == {
            "error": {
                "code": "service_not_ready",
                "message": "The RAG service is not ready.",
            }
        }

    def test_ready_and_knowledge_base_list_preserve_safe_config_order(self) -> None:
        runtime = FakeRuntime()
        with _client(runtime) as client:
            readiness = client.get("/readyz")
            listed = client.get("/v1/knowledge-bases")

        assert readiness.status_code == 200
        assert readiness.json() == {
            "status": "ready",
            "profile": "recommended-v2",
            "knowledge_base_count": 2,
        }
        assert [item["id"] for item in listed.json()["items"]] == [
            "python-docs",
            "uv-docs",
        ]
        serialized = json.dumps(listed.json())
        assert "data_root" not in serialized
        assert "manifest" not in serialized
        assert "/workspace" not in serialized

    def test_alternating_requests_select_only_the_requested_service(self) -> None:
        runtime = FakeRuntime()
        with _client(runtime) as client:
            responses = [
                client.post(
                    "/v1/knowledge-bases/python-docs/answers",
                    json={"question": "Python question"},
                ),
                client.post(
                    "/v1/knowledge-bases/uv-docs/answers",
                    json={"question": "uv question"},
                ),
                client.post(
                    "/v1/knowledge-bases/python-docs/answers",
                    json={"question": "second Python question"},
                ),
            ]

        assert all(response.status_code == 200 for response in responses)
        assert runtime.calls == [
            ("python-docs", "Python question"),
            ("uv-docs", "uv question"),
            ("python-docs", "second Python question"),
        ]
        for response, expected_id in zip(
            responses,
            ("python-docs", "uv-docs", "python-docs"),
            strict=True,
        ):
            payload = response.json()
            assert payload["knowledge_base_id"] == expected_id
            assert payload["sources"][0]["url"] == (
                f"https://trusted.example.test/{expected_id}/answer"
            )
            assert "retrieved.example.test" not in response.text

    def test_answer_response_contains_finalized_sources_timing_and_usage(self) -> None:
        with _client(FakeRuntime()) as client:
            response = client.post(
                "/v1/knowledge-bases/python-docs/answers",
                json={"question": "  list.sort()とは？  "},
            )

        assert response.status_code == 200
        assert response.json() == {
            "knowledge_base_id": "python-docs",
            "status": "answer",
            "answer_text": "list.sort()とは？への回答 [S1]",
            "reason_code": None,
            "sources": [
                {
                    "label": "S1",
                    "page_title": "python-docs page",
                    "section_title": "python-docs section",
                    "url": "https://trusted.example.test/python-docs/answer",
                }
            ],
            "timings": {
                "retrieval_seconds": 0.125,
                "generation_seconds": 0.5,
                "total_seconds": 0.75,
            },
            "usage": {
                "input_tokens": 1234,
                "generated_tokens": 98,
                "generation_calls": 1,
            },
        }

    def test_abstention_never_exposes_answer_text_or_sources(self) -> None:
        runtime = FakeRuntime()
        runtime.abstain = True
        with _client(runtime) as client:
            response = client.post(
                "/v1/knowledge-bases/python-docs/answers",
                json={"question": "outside scope"},
            )

        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "abstain"
        assert payload["answer_text"] is None
        assert payload["reason_code"] == "insufficient_evidence"
        assert payload["sources"] == []
        assert "retrieved.example.test" not in response.text

    def test_unknown_knowledge_base_has_stable_404_without_internal_detail(
        self,
    ) -> None:
        with _client(FakeRuntime()) as client:
            response = client.post(
                "/v1/knowledge-bases/missing/answers",
                json={"question": "question"},
            )

        assert response.status_code == 404
        assert response.json() == {
            "error": {
                "code": "knowledge_base_not_found",
                "message": "The requested knowledge base was not found.",
            }
        }
        assert "/workspace" not in response.text

    @pytest.mark.parametrize(
        "payload",
        [
            {"question": ""},
            {"question": " \n\t "},
            {"question": "x" * 4001},
            {"question": 123},
        ],
    )
    def test_question_validation_rejects_invalid_bodies(
        self,
        payload: dict[str, Any],
    ) -> None:
        runtime = FakeRuntime()
        with _client(runtime) as client:
            response = client.post(
                "/v1/knowledge-bases/python-docs/answers",
                json=payload,
            )

        assert response.status_code == 422
        assert runtime.calls == []

    def test_question_validation_accepts_exact_stripped_4000_characters(
        self,
    ) -> None:
        runtime = FakeRuntime()
        normalized = "x" * 4000
        with _client(runtime) as client:
            response = client.post(
                "/v1/knowledge-bases/python-docs/answers",
                json={"question": f"  {normalized}  "},
            )

        assert response.status_code == 200
        assert runtime.calls == [("python-docs", normalized)]

    def test_question_validation_rejects_unknown_field(self) -> None:
        runtime = FakeRuntime()
        with _client(runtime) as client:
            response = client.post(
                "/v1/knowledge-bases/python-docs/answers",
                json={"question": "question", "unknown": True},
            )

        assert response.status_code == 422
        assert runtime.calls == []

    def test_query_parameter_does_not_replace_required_post_body(self) -> None:
        runtime = FakeRuntime()
        with _client(runtime) as client:
            response = client.post(
                "/v1/knowledge-bases/python-docs/answers?question=hidden",
            )

        assert response.status_code == 422
        assert runtime.calls == []

    def test_answer_failure_is_stable_and_does_not_leak_exception_details(
        self,
    ) -> None:
        runtime = FakeRuntime()
        runtime.failure = FakeAnswerServiceError()
        with _client(runtime) as client:
            response = client.post(
                "/v1/knowledge-bases/python-docs/answers",
                json={"question": "question"},
            )

        assert response.status_code == 500
        assert response.json() == {
            "error": {
                "code": "answer_generation_failed",
                "message": "The answer could not be generated.",
            }
        }
        assert "/workspace" not in response.text
        assert "generation failed" not in response.text

    def test_unsafe_finalized_source_is_generic_500_without_path_leak(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        internal_path = "/workspace/private/python/index.faiss"
        runtime = FakeRuntime()
        runtime.source_url_override = internal_path

        with caplog.at_level("ERROR", logger="python_doc_rag.api"):
            with _client(runtime) as client:
                response = client.post(
                    "/v1/knowledge-bases/python-docs/answers",
                    json={"question": "question"},
                )

        assert response.status_code == 500
        assert response.json() == {
            "error": {
                "code": "answer_generation_failed",
                "message": "The answer could not be generated.",
            }
        }
        assert internal_path not in response.text
        assert internal_path not in caplog.text
        assert "absolute HTTP" not in caplog.text

    def test_success_logs_metadata_but_not_question_or_answer(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        question = "private-question-body"
        with caplog.at_level("INFO", logger="python_doc_rag.api"):
            with _client(FakeRuntime()) as client:
                response = client.post(
                    "/v1/knowledge-bases/python-docs/answers",
                    json={"question": question},
                )

        assert response.status_code == 200
        assert "knowledge_base_id=python-docs" in caplog.text
        assert question not in caplog.text
        assert response.json()["answer_text"] not in caplog.text

    def test_concurrent_answers_are_serialized_while_health_remains_responsive(
        self,
    ) -> None:
        asyncio.run(_assert_serialized_answers_and_live_health())


async def _assert_serialized_answers_and_live_health() -> None:
    import httpx2

    runtime = SlowSerializedRuntime()
    app = create_app(runtime_factory=lambda: runtime)
    transport = httpx2.ASGITransport(app=app)
    async with app.router.lifespan_context(app):
        async with httpx2.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            first = asyncio.create_task(
                client.post(
                    "/v1/knowledge-bases/python-docs/answers",
                    json={"question": "first"},
                )
            )
            await asyncio.wait_for(runtime.first_started.wait(), timeout=1.0)
            second = asyncio.create_task(
                client.post(
                    "/v1/knowledge-bases/uv-docs/answers",
                    json={"question": "second"},
                )
            )
            await asyncio.sleep(0)
            health = await asyncio.wait_for(client.get("/healthz"), timeout=0.5)
            assert health.status_code == 200
            assert runtime.active == 1
            runtime.release.set()
            responses = await asyncio.gather(first, second)

    assert all(response.status_code == 200 for response in responses)
    assert runtime.max_active == 1
