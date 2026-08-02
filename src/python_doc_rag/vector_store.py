"""Build, persist, load, and query a normalized FAISS vector index."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import tempfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from python_doc_rag.models import SearchChunk, SearchResult


class EmbeddingModelProtocol(Protocol):
    """Minimal SentenceTransformer-compatible interface used by this module."""

    def encode(
        self,
        sentences: Sequence[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
        show_progress_bar: bool,
    ) -> NDArray[np.float32]:
        """Encode and L2-normalize a batch of strings."""
        ...

    def get_sentence_embedding_dimension(self) -> int | None:
        """Return the output vector dimension when known."""
        ...


@dataclass(frozen=True, slots=True)
class VectorIndexBuildResult:
    """Summary and output paths for a completed vector-index build."""

    index_path: Path
    metadata_path: Path
    manifest_path: Path
    chunk_count: int
    embedding_dimension: int
    elapsed_seconds: float


class VectorIndex:
    """Loaded FAISS index paired with ordered chunk metadata and an encoder."""

    def __init__(
        self,
        index: Any,
        chunks: Sequence[SearchChunk],
        embedding_model: EmbeddingModelProtocol,
        *,
        query_prefix: str = "",
    ) -> None:
        self._index = index
        self._chunks = tuple(chunks)
        self._embedding_model = embedding_model
        self._query_prefix = query_prefix

    @property
    def chunk_count(self) -> int:
        """Return the number of indexed chunks."""
        return len(self._chunks)

    @property
    def embedding_dimension(self) -> int:
        """Return the FAISS vector dimension."""
        return int(self._index.d)

    def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        """Return nearest chunks in FAISS rank order using normalized query vectors."""
        if not query.strip():
            raise ValueError("query must not be empty or whitespace")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be at least 1")

        query_embedding = _encode_texts(
            self._embedding_model,
            [f"{self._query_prefix}{query}"],
            batch_size=1,
        )
        _validate_embeddings(
            query_embedding,
            expected_rows=1,
            expected_dimension=self.embedding_dimension,
        )
        limit = min(top_k, self.chunk_count)
        if limit == 0:
            return []
        scores, positions = self._index.search(query_embedding, limit)

        results: list[SearchResult] = []
        for rank, (score, position) in enumerate(
            zip(scores[0], positions[0], strict=True),
            start=1,
        ):
            position_value = int(position)
            if position_value < 0 or position_value >= self.chunk_count:
                raise RuntimeError(f"FAISS returned invalid position: {position_value}")
            chunk = self._chunks[position_value]
            results.append(
                SearchResult(
                    rank=rank,
                    score=float(score),
                    chunk=chunk,
                    page_title=chunk.page_title,
                    section_title=chunk.section_title,
                    source_url=chunk.source_url,
                    category=chunk.category,
                )
            )
        return results


def iter_chunks_jsonl(input_jsonl: Path) -> Iterator[SearchChunk]:
    """Read UTF-8 JSONL in source order and report malformed rows by line number."""
    with input_jsonl.expanduser().open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON in {input_jsonl} at line {line_number}: {error.msg}"
                ) from error
            if not isinstance(data, dict):
                raise ValueError(
                    f"invalid chunk in {input_jsonl} at line {line_number}: "
                    "expected a JSON object"
                )
            try:
                chunk = SearchChunk.from_dict(data)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid chunk in {input_jsonl} at line {line_number}: {error}"
                ) from error
            yield chunk


def load_chunks_jsonl(input_jsonl: Path) -> list[SearchChunk]:
    """Load ordered chunk metadata from a UTF-8 JSONL file."""
    return list(iter_chunks_jsonl(input_jsonl))


def build_vector_index(
    input_jsonl: Path,
    index_path: Path,
    metadata_path: Path,
    manifest_path: Path,
    *,
    model_name: str,
    model_revision: str | None = None,
    batch_size: int = 64,
    device: str | None = None,
    document_prefix: str = "",
    query_prefix: str = "",
    trust_remote_code: bool = False,
) -> VectorIndexBuildResult:
    """Encode a processed corpus and atomically persist FAISS index artifacts."""
    if not model_name.strip():
        raise ValueError("model_name must not be empty")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        raise ValueError("batch_size must be at least 1")
    _validate_distinct_paths(input_jsonl, index_path, metadata_path, manifest_path)

    started_at = datetime.now(UTC)
    input_jsonl_sha256 = _sha256(input_jsonl)
    chunks = load_chunks_jsonl(input_jsonl)
    if _sha256(input_jsonl) != input_jsonl_sha256:
        raise RuntimeError("input JSONL changed while it was being read")
    if not chunks:
        raise ValueError("input corpus must contain at least one chunk")

    faiss = _import_faiss()
    embedding_model = _load_sentence_transformer(
        model_name,
        revision=model_revision,
        device=device,
        trust_remote_code=trust_remote_code,
    )
    index: Any | None = None
    embedding_dimension: int | None = None

    for offset in range(0, len(chunks), batch_size):
        batch = chunks[offset : offset + batch_size]
        embeddings = _encode_texts(
            embedding_model,
            [f"{document_prefix}{_document_text(chunk)}" for chunk in batch],
            batch_size=batch_size,
        )
        if embedding_dimension is None:
            if embeddings.ndim != 2 or embeddings.shape[1] < 1:
                raise ValueError("embedding model returned an invalid dimension")
            embedding_dimension = int(embeddings.shape[1])
            declared_dimension = _get_embedding_dimension(embedding_model)
            if (
                declared_dimension is not None
                and declared_dimension != embedding_dimension
            ):
                raise ValueError(
                    "embedding dimension does not match model declaration: "
                    f"{embedding_dimension} != {declared_dimension}"
                )
            index = faiss.IndexFlatIP(embedding_dimension)
        _validate_embeddings(
            embeddings,
            expected_rows=len(batch),
            expected_dimension=embedding_dimension,
        )
        index.add(embeddings)

    if index is None or embedding_dimension is None:
        raise RuntimeError("vector index was not initialized")
    if int(index.ntotal) != len(chunks):
        raise RuntimeError("FAISS index count did not match metadata count")

    temporary_paths = [
        _temporary_sibling(path.expanduser())
        for path in (index_path, metadata_path, manifest_path)
    ]
    temporary_index, temporary_metadata, temporary_manifest = temporary_paths
    try:
        faiss.write_index(index, str(temporary_index))
        _fsync_existing_file(temporary_index)
        _write_chunks_jsonl(chunks, temporary_metadata)
        finished_at = datetime.now(UTC)
        elapsed_seconds = (finished_at - started_at).total_seconds()
        sentence_transformers_version = _package_version("sentence-transformers")
        faiss_version = str(getattr(faiss, "__version__", "unknown"))
        manifest = {
            "model_name": model_name,
            "model_revision": model_revision,
            "embedding_dimension": embedding_dimension,
            "chunk_count": len(chunks),
            "batch_size": batch_size,
            "device": _resolved_device(embedding_model, device),
            "index_type": "IndexFlatIP",
            "normalized_embeddings": True,
            "document_prefix": document_prefix,
            "query_prefix": query_prefix,
            "trust_remote_code": trust_remote_code,
            "input_jsonl": str(input_jsonl),
            "input_jsonl_sha256": input_jsonl_sha256,
            "index_sha256": _sha256(temporary_index),
            "metadata_sha256": _sha256(temporary_metadata),
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "elapsed_seconds": elapsed_seconds,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "sentence_transformers_version": sentence_transformers_version,
            "faiss_version": faiss_version,
        }
        _write_json(manifest, temporary_manifest)
        for temporary, destination in zip(
            temporary_paths,
            (index_path, metadata_path, manifest_path),
            strict=True,
        ):
            os.replace(temporary, destination.expanduser())
    except BaseException:
        for path in temporary_paths:
            path.unlink(missing_ok=True)
        raise

    return VectorIndexBuildResult(
        index_path=index_path,
        metadata_path=metadata_path,
        manifest_path=manifest_path,
        chunk_count=len(chunks),
        embedding_dimension=embedding_dimension,
        elapsed_seconds=elapsed_seconds,
    )


def load_vector_index(
    index_path: Path,
    metadata_path: Path,
    *,
    embedding_model: EmbeddingModelProtocol,
    query_prefix: str = "",
) -> VectorIndex:
    """Load and validate a FAISS index and its ordered JSONL metadata."""
    faiss = _import_faiss()
    index = faiss.read_index(str(index_path.expanduser()))
    chunks = load_chunks_jsonl(metadata_path)
    if int(index.ntotal) != len(chunks):
        raise ValueError(
            "FAISS index count and metadata count differ: "
            f"{int(index.ntotal)} != {len(chunks)}"
        )
    dimension = int(index.d)
    if dimension < 1:
        raise ValueError(f"FAISS index has invalid dimension: {dimension}")
    model_dimension = _get_embedding_dimension(embedding_model)
    if model_dimension is not None and int(model_dimension) != dimension:
        raise ValueError(
            "FAISS index dimension and embedding model dimension differ: "
            f"{dimension} != {model_dimension}"
        )
    return VectorIndex(
        index,
        chunks,
        embedding_model,
        query_prefix=query_prefix,
    )


def _encode_texts(
    model: EmbeddingModelProtocol,
    texts: Sequence[str],
    *,
    batch_size: int,
) -> NDArray[np.float32]:
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return np.ascontiguousarray(np.asarray(embeddings, dtype=np.float32))


def _get_embedding_dimension(model: EmbeddingModelProtocol) -> int | None:
    """Read the model dimension using the newest available API."""
    getter = getattr(model, "get_embedding_dimension", None)
    if callable(getter):
        return getter()
    return model.get_sentence_embedding_dimension()


def _validate_embeddings(
    embeddings: NDArray[np.float32],
    *,
    expected_rows: int,
    expected_dimension: int,
) -> None:
    if embeddings.ndim != 2 or embeddings.shape != (
        expected_rows,
        expected_dimension,
    ):
        raise ValueError(
            "unexpected embedding shape: "
            f"{embeddings.shape}, expected ({expected_rows}, {expected_dimension})"
        )
    if not np.isfinite(embeddings).all():
        raise ValueError("embeddings must contain only finite values")


def _document_text(chunk: SearchChunk) -> str:
    return (
        f"ページタイトル: {chunk.page_title}\n"
        f"セクションタイトル: {chunk.section_title}\n"
        f"本文: {chunk.text}"
    )


def _load_sentence_transformer(
    model_name: str,
    *,
    revision: str | None,
    device: str | None,
    trust_remote_code: bool,
) -> EmbeddingModelProtocol:
    try:
        module = importlib.import_module("sentence_transformers")
    except ImportError as error:
        raise RuntimeError(
            "sentence-transformers is required to build a vector index"
        ) from error
    kwargs: dict[str, Any] = {
        "device": device,
        "trust_remote_code": trust_remote_code,
    }
    if revision is not None:
        kwargs["revision"] = revision
    return module.SentenceTransformer(model_name, **kwargs)


def _import_faiss() -> Any:
    try:
        return importlib.import_module("faiss")
    except ImportError as error:
        raise RuntimeError(
            "faiss-cpu is required for vector index operations"
        ) from error


def _resolved_device(model: EmbeddingModelProtocol, requested: str | None) -> str:
    if requested is not None:
        return requested
    return str(getattr(model, "device", "unknown"))


def _write_chunks_jsonl(chunks: Sequence[SearchChunk], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for chunk in chunks:
            json.dump(chunk.to_dict(), stream, ensure_ascii=False)
            stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_json(data: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _temporary_sibling(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    return Path(name)


def _validate_distinct_paths(*paths: Path) -> None:
    resolved = [path.expanduser().resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("input and output paths must all differ")


def _fsync_existing_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_version(distribution_name: str) -> str:
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"
