"""Core components for the Python documentation RAG proof of concept."""

from python_doc_rag.answer_contract import (
    ANSWER_OR_ABSTAIN_CONTRACT_REVISION,
    AnswerContractError,
    AnswerOrAbstainGenerationContract,
    LegacyGenerationContract,
    parse_answer_contract,
)
from python_doc_rag.generation import (
    ChatTemplatePromptSerializer,
    GenerationConfig,
)
from python_doc_rag.models import (
    AbstainedAnswer,
    AnswerOutcome,
    CitationSource,
    CitedAnswer,
    DocumentSection,
    SearchChunk,
    SearchResult,
)
from python_doc_rag.pipeline import AnswerGenerationFailedError, RagPipeline
from python_doc_rag.retrieval import (
    BM25Retriever,
    CodeAwareNgramTokenizer,
    ReciprocalRankFusionRetriever,
    VectorIndexRetriever,
)
from python_doc_rag.transformers_generation import (
    InputTokenLimitExceededError,
    TransformersAnswerGenerator,
)
from python_doc_rag.vector_store import (
    EmbeddingModelProtocol,
    VectorIndex,
    VectorIndexBuildResult,
    build_vector_index,
    load_vector_index,
)


def __getattr__(name: str) -> object:
    """Load the legacy Python corpus API only when explicitly requested."""
    if name in {"CorpusBuildResult", "build_corpus"}:
        from python_doc_rag.corpus import CorpusBuildResult, build_corpus

        exports = {
            "CorpusBuildResult": CorpusBuildResult,
            "build_corpus": build_corpus,
        }
        globals().update(exports)
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ANSWER_OR_ABSTAIN_CONTRACT_REVISION",
    "AbstainedAnswer",
    "AnswerContractError",
    "AnswerOrAbstainGenerationContract",
    "AnswerOutcome",
    "AnswerGenerationFailedError",
    "BM25Retriever",
    "ChatTemplatePromptSerializer",
    "CitedAnswer",
    "CitationSource",
    "CodeAwareNgramTokenizer",
    "CorpusBuildResult",
    "DocumentSection",
    "EmbeddingModelProtocol",
    "GenerationConfig",
    "InputTokenLimitExceededError",
    "LegacyGenerationContract",
    "RagPipeline",
    "ReciprocalRankFusionRetriever",
    "SearchChunk",
    "SearchResult",
    "TransformersAnswerGenerator",
    "VectorIndex",
    "VectorIndexBuildResult",
    "VectorIndexRetriever",
    "build_corpus",
    "build_vector_index",
    "load_vector_index",
    "parse_answer_contract",
]
