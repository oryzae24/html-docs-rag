import pytest

from python_doc_rag.embedding_models import (
    RETRIEVAL_EMBEDDING_SPECS,
    retrieval_embedding_spec,
)


def test_retrieval_embedding_candidates_are_bounded_and_pinned() -> None:
    assert len(RETRIEVAL_EMBEDDING_SPECS) == 2
    assert {spec.key for spec in RETRIEVAL_EMBEDDING_SPECS} == {
        "bge-m3",
        "multilingual-e5-base",
    }
    for spec in RETRIEVAL_EMBEDDING_SPECS:
        assert len(spec.revision) == 40
        assert spec.license == "MIT"
        assert spec.trust_remote_code is False
        assert spec.normalize_embeddings is True
        assert spec.embedding_dimension > 0
        assert spec.max_sequence_length >= 512


def test_multilingual_e5_prefixes_are_not_silently_omitted() -> None:
    spec = retrieval_embedding_spec("multilingual-e5-base")

    assert spec.query_prefix == "query: "
    assert spec.document_prefix == "passage: "
    assert spec.pooling == "mean"


def test_bge_m3_sentence_transformer_spec_is_prefix_free() -> None:
    spec = retrieval_embedding_spec("bge-m3")

    assert spec.query_prefix == ""
    assert spec.document_prefix == ""
    assert spec.pooling == "CLS"
    assert spec.embedding_dimension == 1024


def test_unknown_embedding_candidate_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        retrieval_embedding_spec("unbounded-model")
