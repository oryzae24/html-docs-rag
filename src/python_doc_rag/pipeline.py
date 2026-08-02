"""Orchestrate retrieval, bounded generation, and contract finalization."""

from collections.abc import Callable, Sequence
from time import perf_counter
from typing import Protocol

from python_doc_rag.answer_contract import (
    AnswerContractError,
    GenerationContract,
    LegacyGenerationContract,
)
from python_doc_rag.generation import (
    AnswerGenerator,
    GenerationConfig,
    IdentityPromptSerializer,
    PromptSerializer,
    TokenizerProtocol,
    count_prompt_tokens,
    select_contexts_within_budget,
)
from python_doc_rag.models import AnswerOutcome, SearchChunk
from python_doc_rag.retrieval import Retriever


class ContextSelector(Protocol):
    """Resolve ranked candidates into the exact context tuple to generate from."""

    def __call__(
        self,
        question: str,
        candidates: Sequence[SearchChunk],
        *,
        tokenizer: TokenizerProtocol,
        max_prompt_tokens: int,
        prompt_serializer: PromptSerializer,
        initial_prompt_builder: Callable[[str, Sequence[SearchChunk]], str],
        retry_prompt_builder: Callable[[str, Sequence[SearchChunk]], str],
    ) -> tuple[SearchChunk, ...]:
        """Return a deterministic, prompt-budget-safe context tuple."""
        ...


class AnswerGenerationFailedError(RuntimeError):
    """Raised after both generations violate the active output contract."""

    def __init__(
        self,
        *,
        first_reasons: tuple[str, ...],
        second_reasons: tuple[str, ...],
    ) -> None:
        self.first_reasons = first_reasons
        self.second_reasons = second_reasons
        super().__init__(
            "answer generation failed contract validation on both attempts"
        )


class RagPipeline:
    """Run the model-independent RAG answer workflow."""

    def __init__(
        self,
        *,
        retriever: Retriever,
        generator: AnswerGenerator,
        tokenizer: TokenizerProtocol,
        prompt_serializer: PromptSerializer | None = None,
        config: GenerationConfig | None = None,
        generation_contract: GenerationContract | None = None,
        context_selector: ContextSelector | None = None,
    ) -> None:
        self._retriever = retriever
        self._generator = generator
        self._tokenizer = tokenizer
        self._prompt_serializer = prompt_serializer or IdentityPromptSerializer()
        self._config = config or GenerationConfig()
        self._contract = generation_contract or LegacyGenerationContract()
        self._context_selector = context_selector or select_contexts_within_budget
        self._last_generation_seconds = 0.0
        self._last_retrieved_chunks: tuple[SearchChunk, ...] = ()
        self._last_selected_chunks: tuple[SearchChunk, ...] = ()
        self._last_prompt_token_counts: list[int] = []

    @property
    def answer_mode(self) -> str:
        """Return the active stable answer-mode name."""
        return self._contract.answer_mode

    @property
    def contract_revision(self) -> str:
        """Return the active model-output contract revision."""
        return self._contract.revision

    @property
    def last_generation_seconds(self) -> float:
        """Return model generation wall time for the latest question."""
        return self._last_generation_seconds

    @property
    def last_retrieved_chunks(self) -> tuple[SearchChunk, ...]:
        """Return the ranked chunks retrieved for the latest question."""
        return self._last_retrieved_chunks

    @property
    def last_selected_chunks(self) -> tuple[SearchChunk, ...]:
        """Return the fixed context tuple used by generation and retry."""
        return self._last_selected_chunks

    @property
    def last_prompt_token_counts(self) -> tuple[int, ...]:
        """Return exact token counts for prompts actually sent to the model."""
        return tuple(self._last_prompt_token_counts)

    def answer(self, question: str) -> AnswerOutcome:
        """Return a cited answer or a normal explicit abstention."""
        self._last_generation_seconds = 0.0
        self._last_retrieved_chunks = ()
        self._last_selected_chunks = ()
        self._last_prompt_token_counts = []
        question = _validate_question(question)
        retrieved_chunks = tuple(
            self._retriever.retrieve(
                question,
                limit=self._config.retrieval_limit,
            )
        )
        self._last_retrieved_chunks = retrieved_chunks
        if not retrieved_chunks:
            return self._contract.empty_result()

        selected_chunks = self._context_selector(
            question,
            retrieved_chunks,
            tokenizer=self._tokenizer,
            max_prompt_tokens=self._config.max_prompt_tokens,
            prompt_serializer=self._prompt_serializer,
            initial_prompt_builder=self._contract.build_initial_prompt,
            retry_prompt_builder=self._contract.build_retry_prompt,
        )
        self._last_selected_chunks = selected_chunks

        first_prompt = self._prompt_serializer.serialize(
            self._contract.build_initial_prompt(question, selected_chunks)
        )
        self._last_prompt_token_counts.append(self._verify_prompt_budget(first_prompt))
        first_answer = self._generate(first_prompt)
        try:
            return self._contract.finalize(
                first_answer,
                selected_chunks,
                generation_attempts=1,
            )
        except AnswerContractError as first_error:
            second_prompt = self._prompt_serializer.serialize(
                self._contract.build_retry_prompt(question, selected_chunks)
            )
            self._last_prompt_token_counts.append(
                self._verify_prompt_budget(second_prompt)
            )
            second_answer = self._generate(second_prompt)
            try:
                return self._contract.finalize(
                    second_answer,
                    selected_chunks,
                    generation_attempts=2,
                )
            except AnswerContractError as second_error:
                raise AnswerGenerationFailedError(
                    first_reasons=first_error.reasons,
                    second_reasons=second_error.reasons,
                ) from second_error

    def _generate(self, prompt: str) -> str:
        started_at = perf_counter()
        try:
            return self._generator.generate(
                prompt,
                max_new_tokens=self._config.max_new_tokens,
            )
        finally:
            self._last_generation_seconds += perf_counter() - started_at

    def _verify_prompt_budget(self, prompt: str) -> int:
        """Measure the exact string that will be handed to the generator."""
        token_count = count_prompt_tokens(self._tokenizer, prompt)
        if token_count > self._config.max_prompt_tokens:
            raise RuntimeError("selected chunks produced an over-budget prompt")
        return token_count


def _validate_question(question: str) -> str:
    """Normalize a non-empty question before retrieval."""
    if not isinstance(question, str):
        raise TypeError("question must be a string")
    normalized = question.strip()
    if not normalized:
        raise ValueError("question must not be blank")
    return normalized
