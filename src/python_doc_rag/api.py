"""Lazy FastAPI adapter for the multi-knowledge-base RAG runtime."""

import asyncio
import logging
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

LOGGER = logging.getLogger(__name__)

_SERVICE_NOT_READY = {
    "error": {
        "code": "service_not_ready",
        "message": "The RAG service is not ready.",
    }
}


def create_app(
    *,
    runtime_factory: Callable[[], Any] | None = None,
    service_config: Path | None = None,
) -> Any:
    """Create the ASGI app without importing web or ML dependencies at module load."""
    if runtime_factory is None and service_config is None:
        raise ValueError("runtime_factory or service_config is required")
    if runtime_factory is not None and service_config is not None:
        raise ValueError("runtime_factory and service_config are mutually exclusive")

    try:
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse
        from pydantic import BaseModel, ConfigDict, Field, field_validator
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "REST APIにはapi依存関係が必要です。"
            "`uv sync --frozen --extra inference --extra api`を実行してください。"
        ) from error

    from python_doc_rag.application import (
        AnswerServiceError,
        KnowledgeBaseNotFoundError,
    )
    from python_doc_rag.models import AbstainedAnswer, CitedAnswer

    class QuestionRequest(BaseModel):
        """One independent question with no conversation history."""

        model_config = ConfigDict(extra="forbid")

        question: str = Field(strict=True, min_length=1, max_length=4000)

        @field_validator("question", mode="before")
        @classmethod
        def normalize_question(cls, value: object) -> object:
            if isinstance(value, str):
                return value.strip()
            return value

    class SourceResponse(BaseModel):
        """One citation finalized from trusted retrieval metadata."""

        label: str
        page_title: str
        section_title: str
        url: str

        @field_validator("url")
        @classmethod
        def validate_source_url(cls, value: str) -> str:
            """Reject malformed metadata and local paths at the HTTP boundary."""
            parsed = urlsplit(value)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("source URL must be an absolute HTTP(S) URL")
            return value

    class TimingResponse(BaseModel):
        """Server-side RAG execution timings, excluding queue wait time."""

        retrieval_seconds: float
        generation_seconds: float
        total_seconds: float

    class UsageResponse(BaseModel):
        """Local model token and generation-call counts."""

        input_tokens: int
        generated_tokens: int
        generation_calls: int

    class AnswerResponse(BaseModel):
        """A cited answer or a normal fail-closed abstention."""

        knowledge_base_id: str
        status: Literal["answer", "abstain"]
        answer_text: str | None
        reason_code: str | None
        sources: list[SourceResponse]
        timings: TimingResponse
        usage: UsageResponse

    class ReadinessResponse(BaseModel):
        """Readiness of all configured knowledge bases and shared models."""

        status: Literal["ready"]
        profile: str
        knowledge_base_count: int

    class KnowledgeBaseResponse(BaseModel):
        """Public knowledge-base metadata with no local artifact paths."""

        id: str
        display_name: str
        dataset_name: str
        profile: str
        status: Literal["ready"]

    class KnowledgeBaseListResponse(BaseModel):
        """Knowledge bases in stable ServiceConfig order."""

        items: list[KnowledgeBaseResponse]

    selected_factory = runtime_factory or _service_config_runtime_factory(
        service_config
    )

    @asynccontextmanager
    async def lifespan(application: Any):
        application.state.runtime = None
        try:
            application.state.runtime = await asyncio.to_thread(selected_factory)
            yield
        finally:
            application.state.runtime = None

    app = FastAPI(
        title="Multi-Knowledge-Base Local RAG API",
        version="1.0.0",
        description=(
            "Read-only access to prepared local documentation knowledge bases. "
            "Authentication and authorization are not provided."
        ),
        lifespan=lifespan,
    )
    app.state.runtime = None

    def runtime_if_ready() -> Any | None:
        runtime = app.state.runtime
        if runtime is None or not runtime.ready:
            return None
        return runtime

    @app.get("/healthz", summary="Check whether the ASGI process is alive")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get(
        "/readyz",
        response_model=ReadinessResponse,
        summary="Check whether every configured knowledge base is ready",
    )
    async def readiness() -> Any:
        runtime = runtime_if_ready()
        if runtime is None:
            return JSONResponse(status_code=503, content=_SERVICE_NOT_READY)
        return ReadinessResponse(
            status="ready",
            profile=runtime.profile_name,
            knowledge_base_count=runtime.knowledge_base_count,
        )

    @app.get(
        "/v1/knowledge-bases",
        response_model=KnowledgeBaseListResponse,
        summary="List ready knowledge bases",
    )
    async def knowledge_bases() -> Any:
        runtime = runtime_if_ready()
        if runtime is None:
            return JSONResponse(status_code=503, content=_SERVICE_NOT_READY)
        return KnowledgeBaseListResponse(
            items=[
                KnowledgeBaseResponse(
                    id=service.id,
                    display_name=service.display_name,
                    dataset_name=service.dataset_name,
                    profile=runtime.profile_name,
                    status="ready",
                )
                for service in runtime.knowledge_bases.values()
            ]
        )

    @app.post(
        "/v1/knowledge-bases/{knowledge_base_id}/answers",
        response_model=AnswerResponse,
        summary="Answer one independent question from one knowledge base",
    )
    async def answer(
        knowledge_base_id: str,
        request: QuestionRequest,
    ) -> Any:
        runtime = runtime_if_ready()
        if runtime is None:
            return JSONResponse(status_code=503, content=_SERVICE_NOT_READY)
        try:
            execution = await runtime.answer(
                knowledge_base_id,
                request.question,
            )
            response = _answer_response(
                execution,
                knowledge_base_id=knowledge_base_id,
                cited_answer_type=CitedAnswer,
                abstained_answer_type=AbstainedAnswer,
                answer_response_type=AnswerResponse,
                source_response_type=SourceResponse,
                timing_response_type=TimingResponse,
                usage_response_type=UsageResponse,
            )
        except KnowledgeBaseNotFoundError:
            LOGGER.info(
                "answer rejected knowledge_base_id=%r error_code=%s",
                knowledge_base_id,
                "knowledge_base_not_found",
            )
            return JSONResponse(
                status_code=404,
                content={
                    "error": {
                        "code": "knowledge_base_not_found",
                        "message": "The requested knowledge base was not found.",
                    }
                },
            )
        except AnswerServiceError as error:
            LOGGER.error(
                "answer failed knowledge_base_id=%s error_code=%s error_type=%s",
                knowledge_base_id,
                "answer_generation_failed",
                type(error).__name__,
            )
            return _answer_failure(JSONResponse)
        except Exception as error:
            LOGGER.error(
                "answer failed knowledge_base_id=%s error_code=%s error_type=%s",
                knowledge_base_id,
                "answer_generation_failed",
                type(error).__name__,
            )
            return _answer_failure(JSONResponse)

        LOGGER.info(
            "answer completed knowledge_base_id=%s status=%s total_seconds=%.3f",
            knowledge_base_id,
            response.status,
            response.timings.total_seconds,
        )
        return response

    return app


def _service_config_runtime_factory(service_config: Path | None) -> Callable[[], Any]:
    if service_config is None:
        raise AssertionError("service_config must be present")
    config_path = service_config.expanduser()

    def build() -> Any:
        from python_doc_rag.application import build_multi_knowledge_base_runtime
        from python_doc_rag.service_config import load_service_config

        config = load_service_config(config_path)
        return build_multi_knowledge_base_runtime(config)

    return build


def _answer_response(
    execution: Any,
    *,
    knowledge_base_id: str,
    cited_answer_type: type[Any],
    abstained_answer_type: type[Any],
    answer_response_type: type[Any],
    source_response_type: type[Any],
    timing_response_type: type[Any],
    usage_response_type: type[Any],
) -> Any:
    """Serialize finalized domain objects without reparsing model output."""
    outcome = execution.answer
    if isinstance(outcome, cited_answer_type):
        status = "answer"
        answer_text = outcome.answer_text
        reason_code = None
        sources = [
            source_response_type(
                label=source.label,
                page_title=source.page_title,
                section_title=source.section_title,
                url=source.url,
            )
            for source in outcome.sources
        ]
    elif isinstance(outcome, abstained_answer_type):
        status = "abstain"
        answer_text = None
        reason_code = outcome.reason_code
        sources = []
    else:
        raise TypeError("answer service returned an unsupported outcome")
    return answer_response_type(
        knowledge_base_id=knowledge_base_id,
        status=status,
        answer_text=answer_text,
        reason_code=reason_code,
        sources=sources,
        timings=timing_response_type(
            retrieval_seconds=execution.retrieval_seconds,
            generation_seconds=execution.generation_seconds,
            total_seconds=execution.total_seconds,
        ),
        usage=usage_response_type(
            input_tokens=execution.input_tokens,
            generated_tokens=execution.generated_tokens,
            generation_calls=execution.generation_calls,
        ),
    )


def _answer_failure(json_response_type: type[Any]) -> Any:
    return json_response_type(
        status_code=500,
        content={
            "error": {
                "code": "answer_generation_failed",
                "message": "The answer could not be generated.",
            }
        },
    )
