import hashlib
import json
import pickle
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from python_doc_rag.models import SearchChunk
from python_doc_rag.vector_store import (
    _get_embedding_dimension,
    build_vector_index,
    load_chunks_jsonl,
    load_vector_index,
)


class DummyEmbeddingModel:
    """Return deterministic normalized vectors without loading a real model."""

    device = "cpu"

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def encode(
        self,
        sentences: Sequence[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
        show_progress_bar: bool,
    ) -> NDArray[np.float32]:
        del batch_size, convert_to_numpy, show_progress_bar
        self.calls.append(tuple(sentences))
        vectors = np.asarray([self._vector(text) for text in sentences], dtype=np.float32)
        if normalize_embeddings:
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors

    def get_sentence_embedding_dimension(self) -> int:
        return 3

    @staticmethod
    def _vector(text: str) -> list[float]:
        if "アルファ" in text:
            return [1.0, 0.0, 0.0]
        if "ベータ" in text:
            return [0.8, 0.6, 0.0]
        if "ガンマ" in text:
            return [0.0, 1.0, 0.0]
        raise AssertionError(f"unexpected embedding input: {text}")


class NewApiEmbeddingModel(DummyEmbeddingModel):
    """Expose both dimension APIs while rejecting use of the legacy one."""

    def get_embedding_dimension(self) -> int:
        return 3

    def get_sentence_embedding_dimension(self) -> int:
        raise AssertionError("legacy dimension API must not be used")


class FakeIndexFlatIP:
    """Small NumPy implementation of the FAISS methods exercised by unit tests."""

    def __init__(self, dimension: int) -> None:
        self.d = dimension
        self._vectors = np.empty((0, dimension), dtype=np.float32)

    @property
    def ntotal(self) -> int:
        return len(self._vectors)

    def add(self, vectors: NDArray[np.float32]) -> None:
        self._vectors = np.concatenate((self._vectors, vectors), axis=0)

    def search(
        self,
        queries: NDArray[np.float32],
        limit: int,
    ) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
        scores = queries @ self._vectors.T
        positions = np.argsort(-scores, axis=1, kind="stable")[:, :limit]
        ordered_scores = np.take_along_axis(scores, positions, axis=1)
        return ordered_scores.astype(np.float32), positions.astype(np.int64)


class FakeFaiss:
    """Non-pickle FAISS persistence double for offline unit tests."""

    __version__ = "test-double"
    IndexFlatIP = FakeIndexFlatIP

    @staticmethod
    def write_index(index: FakeIndexFlatIP, path: str) -> None:
        data = {"dimension": index.d, "vectors": index._vectors.tolist()}
        Path(path).write_text(json.dumps(data), encoding="utf-8")

    @staticmethod
    def read_index(path: str) -> FakeIndexFlatIP:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        index = FakeIndexFlatIP(data["dimension"])
        index.add(np.asarray(data["vectors"], dtype=np.float32))
        return index


def _chunk(name: str, index: int) -> SearchChunk:
    return SearchChunk(
        text=f"{name}の本文",
        page_title=f"{name}ページ",
        section_title=f"{name}節",
        source_url=f"https://docs.python.org/ja/3.13/tutorial/{name}.html",
        category="tutorial",
        chunk_index=index,
        start_index=index * 10,
        extra_metadata={"source_path": f"tutorial/{name}.html"},
    )


def _write_chunks(path: Path, chunks: Sequence[SearchChunk]) -> None:
    path.write_text(
        "".join(
            f"{json.dumps(chunk.to_dict(), ensure_ascii=False)}\n" for chunk in chunks
        ),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_embedding_dimension_prefers_new_api_and_supports_legacy_api() -> None:
    assert _get_embedding_dimension(NewApiEmbeddingModel()) == 3
    assert _get_embedding_dimension(DummyEmbeddingModel()) == 3


@pytest.fixture
def built_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path, Path, DummyEmbeddingModel]:
    input_path = tmp_path / "chunks.jsonl"
    index_path = tmp_path / "index.faiss"
    metadata_path = tmp_path / "metadata.jsonl"
    manifest_path = tmp_path / "manifest.json"
    model = DummyEmbeddingModel()
    _write_chunks(
        input_path,
        [_chunk("アルファ", 0), _chunk("ベータ", 1), _chunk("ガンマ", 2)],
    )
    monkeypatch.setattr(
        "python_doc_rag.vector_store._import_faiss",
        lambda: FakeFaiss,
    )
    monkeypatch.setattr(
        "python_doc_rag.vector_store._load_sentence_transformer",
        lambda model_name, **kwargs: model,
    )
    monkeypatch.setattr(
        pickle,
        "dump",
        lambda *args, **kwargs: pytest.fail("pickle.dump must not be used"),
    )
    build_vector_index(
        input_path,
        index_path,
        metadata_path,
        manifest_path,
        model_name="fixture-model",
        batch_size=2,
        device="cpu",
    )
    return input_path, index_path, metadata_path, manifest_path, model


def test_load_chunks_jsonl_preserves_order_blanks_and_all_metadata(
    tmp_path: Path,
) -> None:
    path = tmp_path / "chunks.jsonl"
    first = _chunk("アルファ", 0)
    second = _chunk("ベータ", 1)
    path.write_text(
        f"{json.dumps(first.to_dict(), ensure_ascii=False)}\n\n"
        f"{json.dumps(second.to_dict(), ensure_ascii=False)}\n",
        encoding="utf-8",
    )

    loaded = load_chunks_jsonl(path)

    assert [chunk.text for chunk in loaded] == ["アルファの本文", "ベータの本文"]
    assert loaded[0].extra_metadata == {"source_path": "tutorial/アルファ.html"}
    assert loaded[0].to_dict() == first.to_dict()


@pytest.mark.parametrize(
    "source_url",
    (
        "/workspace/private/python/index.faiss",
        "file:///workspace/private/python/index.faiss",
        "https:///missing-host",
    ),
)
def test_search_chunk_rejects_non_http_or_non_absolute_source_url(
    source_url: str,
) -> None:
    with pytest.raises(ValueError, match="absolute HTTP\\(S\\) URL"):
        SearchChunk(
            text="本文",
            page_title="ページ",
            section_title="節",
            source_url=source_url,
            category="tutorial",
            chunk_index=0,
            start_index=0,
        )


def test_load_chunks_jsonl_reports_invalid_json_line(tmp_path: Path) -> None:
    path = tmp_path / "broken.jsonl"
    path.write_text("\n{}\n{broken}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"line 2:.*(?:category|chunk_index)"):
        load_chunks_jsonl(path)

    path.write_text("\n\n{broken}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"invalid JSON.*line 3"):
        load_chunks_jsonl(path)


def test_load_chunks_jsonl_reports_missing_required_key(tmp_path: Path) -> None:
    path = tmp_path / "missing.jsonl"
    data = _chunk("アルファ", 0).to_dict()
    del data["source_url"]
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match=r"line 1:.*source_url"):
        load_chunks_jsonl(path)


def test_load_chunks_jsonl_reports_unsafe_source_url_line(tmp_path: Path) -> None:
    path = tmp_path / "unsafe-source.jsonl"
    first = _chunk("アルファ", 0).to_dict()
    second = _chunk("ベータ", 1).to_dict()
    second["source_url"] = "/workspace/private/python/index.faiss"
    path.write_text(
        f"{json.dumps(first, ensure_ascii=False)}\n"
        f"{json.dumps(second, ensure_ascii=False)}\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"invalid chunk.*line 2:.*absolute HTTP\(S\) URL",
    ):
        load_chunks_jsonl(path)


def test_build_rejects_empty_corpus(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "empty.jsonl"
    input_path.write_text("\n", encoding="utf-8")
    monkeypatch.setattr(
        "python_doc_rag.vector_store._import_faiss",
        lambda: pytest.fail("FAISS must not load for an empty corpus"),
    )

    with pytest.raises(ValueError, match="at least one chunk"):
        build_vector_index(
            input_path,
            tmp_path / "index.faiss",
            tmp_path / "metadata.jsonl",
            tmp_path / "manifest.json",
            model_name="fixture-model",
        )


def test_index_flat_ip_search_rank_score_and_metadata(
    built_index: tuple[Path, Path, Path, Path, DummyEmbeddingModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, index_path, metadata_path, _, model = built_index
    monkeypatch.setattr(
        "python_doc_rag.vector_store._import_faiss",
        lambda: FakeFaiss,
    )
    vector_index = load_vector_index(
        index_path,
        metadata_path,
        embedding_model=model,
    )

    results = vector_index.search("アルファについて", top_k=10)

    assert [result.rank for result in results] == [1, 2, 3]
    assert [result.score for result in results] == pytest.approx([1.0, 0.8, 0.0])
    assert [result.chunk.text for result in results] == [
        "アルファの本文",
        "ベータの本文",
        "ガンマの本文",
    ]
    assert results[1].page_title == "ベータページ"
    assert results[1].source_url.endswith("tutorial/ベータ.html")
    assert results[1].chunk.extra_metadata["source_path"] == "tutorial/ベータ.html"


def test_search_rejects_empty_query_and_invalid_top_k(
    built_index: tuple[Path, Path, Path, Path, DummyEmbeddingModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, index_path, metadata_path, _, model = built_index
    monkeypatch.setattr(
        "python_doc_rag.vector_store._import_faiss",
        lambda: FakeFaiss,
    )
    vector_index = load_vector_index(
        index_path,
        metadata_path,
        embedding_model=model,
    )

    with pytest.raises(ValueError, match="query"):
        vector_index.search("  \n")
    with pytest.raises(ValueError, match="top_k"):
        vector_index.search("アルファ", top_k=0)


def test_load_rejects_index_metadata_count_mismatch(
    built_index: tuple[Path, Path, Path, Path, DummyEmbeddingModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, index_path, metadata_path, _, model = built_index
    metadata_path.write_text(
        metadata_path.read_text(encoding="utf-8").splitlines()[0] + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "python_doc_rag.vector_store._import_faiss",
        lambda: FakeFaiss,
    )

    with pytest.raises(ValueError, match="count and metadata count differ"):
        load_vector_index(index_path, metadata_path, embedding_model=model)


def test_build_save_reload_is_stable_and_does_not_use_pickle(
    built_index: tuple[Path, Path, Path, Path, DummyEmbeddingModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, index_path, metadata_path, _, model = built_index
    monkeypatch.setattr(
        "python_doc_rag.vector_store._import_faiss",
        lambda: FakeFaiss,
    )
    monkeypatch.setattr(
        pickle,
        "dump",
        lambda *args, **kwargs: pytest.fail("pickle.dump must not be used"),
    )

    first = load_vector_index(
        index_path,
        metadata_path,
        embedding_model=model,
    ).search("アルファ", top_k=3)
    second = load_vector_index(
        index_path,
        metadata_path,
        embedding_model=model,
    ).search("アルファ", top_k=3)

    assert [(item.score, item.chunk) for item in first] == [
        (item.score, item.chunk) for item in second
    ]
    assert json.loads(index_path.read_text(encoding="utf-8"))["dimension"] == 3


def test_loaded_vector_index_applies_query_prefix_to_exact_search_input(
    built_index: tuple[Path, Path, Path, Path, DummyEmbeddingModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, index_path, metadata_path, _, model = built_index
    monkeypatch.setattr(
        "python_doc_rag.vector_store._import_faiss",
        lambda: FakeFaiss,
    )
    index = load_vector_index(
        index_path,
        metadata_path,
        embedding_model=model,
        query_prefix="query: ",
    )

    index.search("アルファ", top_k=1)

    assert model.calls[-1] == ("query: アルファ",)


def test_manifest_counts_hashes_versions_and_document_format(
    built_index: tuple[Path, Path, Path, Path, DummyEmbeddingModel],
) -> None:
    input_path, index_path, metadata_path, manifest_path, model = built_index
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["model_name"] == "fixture-model"
    assert manifest["embedding_dimension"] == 3
    assert manifest["chunk_count"] == 3
    assert manifest["batch_size"] == 2
    assert manifest["device"] == "cpu"
    assert manifest["index_type"] == "IndexFlatIP"
    assert manifest["normalized_embeddings"] is True
    assert manifest["model_revision"] is None
    assert manifest["document_prefix"] == ""
    assert manifest["query_prefix"] == ""
    assert manifest["trust_remote_code"] is False
    assert manifest["input_jsonl_sha256"] == _sha256(input_path)
    assert manifest["index_sha256"] == _sha256(index_path)
    assert manifest["metadata_sha256"] == _sha256(metadata_path)
    assert manifest["python_version"]
    assert manifest["numpy_version"] == np.__version__
    assert manifest["sentence_transformers_version"]
    assert manifest["faiss_version"] == "test-double"
    document_call = model.calls[0][0]
    assert "ページタイトル: アルファページ" in document_call
    assert "セクションタイトル: アルファ節" in document_call
    assert "本文: アルファの本文" in document_call
