"""Pinned multilingual embedding candidates for retrieval-only experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetrievalEmbeddingSpec:
    """Reproducible model-card metadata and input formatting."""

    key: str
    model_name: str
    revision: str
    license: str
    model_card_url: str
    language_evidence: str
    embedding_dimension: int
    max_sequence_length: int
    pooling: str
    query_prefix: str
    document_prefix: str
    normalize_embeddings: bool = True
    trust_remote_code: bool = False


RETRIEVAL_EMBEDDING_SPECS = (
    RetrievalEmbeddingSpec(
        key="bge-m3",
        model_name="BAAI/bge-m3",
        revision="5617a9f61b028005a4858fdac845db406aefb181",
        license="MIT",
        model_card_url="https://huggingface.co/BAAI/bge-m3",
        language_evidence=(
            "The model card states support for more than 100 languages and "
            "documents dense retrieval through Sentence Transformers."
        ),
        embedding_dimension=1024,
        max_sequence_length=8192,
        pooling="CLS",
        query_prefix="",
        document_prefix="",
    ),
    RetrievalEmbeddingSpec(
        key="multilingual-e5-base",
        model_name="intfloat/multilingual-e5-base",
        revision="d128750597153bb5987e10b1c3493a34e5a4502a",
        license="MIT",
        model_card_url=("https://huggingface.co/intfloat/multilingual-e5-base"),
        language_evidence=(
            "The model card lists 94 languages and requires query/passage "
            "prefixes for multilingual asymmetric retrieval."
        ),
        embedding_dimension=768,
        max_sequence_length=512,
        pooling="mean",
        query_prefix="query: ",
        document_prefix="passage: ",
    ),
)


def retrieval_embedding_spec(key: str) -> RetrievalEmbeddingSpec:
    """Resolve one of the two bounded retrieval candidates."""
    for spec in RETRIEVAL_EMBEDDING_SPECS:
        if spec.key == key:
            return spec
    raise ValueError(f"unsupported retrieval embedding key: {key}")
