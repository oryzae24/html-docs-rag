"""Named runtime profiles that preserve the baseline CLI defaults."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from python_doc_rag.answer_contract import ANSWER_MODE_LEGACY
from python_doc_rag.config import DEFAULT_GENERATION_MODEL

DEFAULT_PROFILE_NAME = "default"
RECOMMENDED_V1_PROFILE_NAME = "recommended-v1"
RECOMMENDED_V2_PROFILE_NAME = "recommended-v2"
RECOMMENDED_PROFILE_NAME = "recommended"
PROFILE_NAMES = (
    DEFAULT_PROFILE_NAME,
    RECOMMENDED_V1_PROFILE_NAME,
    RECOMMENDED_V2_PROFILE_NAME,
    RECOMMENDED_PROFILE_NAME,
)

_BGE_ROOT = "experiments/final_quality_sprint_v2/phase_a/embedding_indexes/bge-m3"
_SYMBOL_PATH = "experiments/final_quality_sprint_v2/phase_a/symbol_fields.jsonl"


@dataclass(frozen=True, slots=True)
class RuntimeProfile:
    """One reproducible retrieval and generation configuration."""

    name: str
    revision: str
    alias_of: str | None
    description: str
    retriever: str
    retrieval_revision: str | None
    embedding_model_key: str | None
    embedding_index_path: str | None
    embedding_metadata_path: str | None
    embedding_manifest_path: str | None
    symbol_index_path: str | None
    symbol_index_sha256: str | None
    field_rrf_k: int | None
    field_candidate_k: int | None
    answer_mode: str
    answer_contract_revision: str
    output_constraint_revision: str
    generation_model_key: str | None
    generation_model: str
    generation_model_revision: str | None
    dtype: str
    top_k: int
    max_input_tokens: int
    max_new_tokens: int
    reranker_model_key: str | None
    reranker_candidate_k: int | None
    reranker_batch_size: int | None
    reranker_max_length: int | None
    required_artifacts: tuple[str, ...]
    optional_dependency: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-compatible description."""
        return asdict(self)


def _recommended_v1(name: str, *, alias_of: str | None = None) -> RuntimeProfile:
    return RuntimeProfile(
        name=name,
        revision="recommended-v1",
        alias_of=alias_of,
        description=(
            "First final-quality winner: Hybrid candidates, local mMARCO "
            "MiniLM reranking, Qwen3-4B, and answer-or-abstain-v1."
        ),
        retriever="hybrid",
        retrieval_revision="hybrid-rrf-v1",
        embedding_model_key=None,
        embedding_index_path=None,
        embedding_metadata_path=None,
        embedding_manifest_path=None,
        symbol_index_path=None,
        symbol_index_sha256=None,
        field_rrf_k=None,
        field_candidate_k=30,
        answer_mode="answer-or-abstain",
        answer_contract_revision="answer-or-abstain-v1",
        output_constraint_revision="answer-or-abstain-v1",
        generation_model_key="baseline-qwen3-4b",
        generation_model="Qwen/Qwen3-4B-Instruct-2507",
        generation_model_revision=("cdbee75f17c01a7cc42f958dc650907174af0554"),
        dtype="bfloat16",
        top_k=5,
        max_input_tokens=8192,
        max_new_tokens=512,
        reranker_model_key="mmarco-minilm",
        reranker_candidate_k=30,
        reranker_batch_size=16,
        reranker_max_length=512,
        required_artifacts=(
            "indexes/python_3_13_ja.faiss",
            "indexes/python_3_13_ja_metadata.jsonl",
            "indexes/python_3_13_ja_index_manifest.json",
        ),
        optional_dependency="inference",
    )


def _recommended_v2(name: str, *, alias_of: str | None = None) -> RuntimeProfile:
    return RuntimeProfile(
        name=name,
        revision="recommended-v2",
        alias_of=alias_of,
        description=(
            "Second final-quality winner: equal technical-field fusion with "
            "BGE-M3, local mMARCO reranking, Qwen3-8B non-thinking, and "
            "answer-or-abstain-v1."
        ),
        retriever="technical-field",
        retrieval_revision="technical-field-retrieval-v1",
        embedding_model_key="bge-m3",
        embedding_index_path=f"{_BGE_ROOT}/index.faiss",
        embedding_metadata_path=f"{_BGE_ROOT}/metadata.jsonl",
        embedding_manifest_path=f"{_BGE_ROOT}/manifest.json",
        symbol_index_path=_SYMBOL_PATH,
        symbol_index_sha256=(
            "15dc7f8c9d83a16a91ffbf11dc9015b4a5ce6f71545fb81837f3babbc8545c1a"
        ),
        field_rrf_k=10,
        field_candidate_k=30,
        answer_mode="answer-or-abstain",
        answer_contract_revision="answer-or-abstain-v1",
        output_constraint_revision="answer-or-abstain-v1",
        generation_model_key="qwen3-8b",
        generation_model="Qwen/Qwen3-8B",
        generation_model_revision=("b968826d9c46dd6066d109eabc6255188de91218"),
        dtype="bfloat16",
        top_k=5,
        max_input_tokens=8192,
        max_new_tokens=512,
        reranker_model_key="mmarco-minilm",
        reranker_candidate_k=30,
        reranker_batch_size=16,
        reranker_max_length=512,
        required_artifacts=(
            f"{_BGE_ROOT}/index.faiss",
            f"{_BGE_ROOT}/metadata.jsonl",
            f"{_BGE_ROOT}/manifest.json",
            _SYMBOL_PATH,
        ),
        optional_dependency="inference",
    )


_PROFILES = {
    DEFAULT_PROFILE_NAME: RuntimeProfile(
        name=DEFAULT_PROFILE_NAME,
        revision="default-v1",
        alias_of=None,
        description="Backward-compatible Dense + legacy answer contract.",
        retriever="dense",
        retrieval_revision=None,
        embedding_model_key=None,
        embedding_index_path=None,
        embedding_metadata_path=None,
        embedding_manifest_path=None,
        symbol_index_path=None,
        symbol_index_sha256=None,
        field_rrf_k=None,
        field_candidate_k=None,
        answer_mode=ANSWER_MODE_LEGACY,
        answer_contract_revision="legacy",
        output_constraint_revision="legacy",
        generation_model_key=None,
        generation_model=DEFAULT_GENERATION_MODEL,
        generation_model_revision=None,
        dtype="auto",
        top_k=5,
        max_input_tokens=8192,
        max_new_tokens=512,
        reranker_model_key=None,
        reranker_candidate_k=None,
        reranker_batch_size=None,
        reranker_max_length=None,
        required_artifacts=(
            "indexes/python_3_13_ja.faiss",
            "indexes/python_3_13_ja_metadata.jsonl",
            "indexes/python_3_13_ja_index_manifest.json",
        ),
        optional_dependency=None,
    ),
    RECOMMENDED_V1_PROFILE_NAME: _recommended_v1(RECOMMENDED_V1_PROFILE_NAME),
    RECOMMENDED_V2_PROFILE_NAME: _recommended_v2(RECOMMENDED_V2_PROFILE_NAME),
    RECOMMENDED_PROFILE_NAME: _recommended_v2(
        RECOMMENDED_PROFILE_NAME,
        alias_of=RECOMMENDED_V2_PROFILE_NAME,
    ),
}


def runtime_profile(name: str) -> RuntimeProfile:
    """Resolve a fixed profile name without falling back silently."""
    try:
        return _PROFILES[name]
    except KeyError as error:
        raise ValueError(f"unsupported runtime profile: {name}") from error
