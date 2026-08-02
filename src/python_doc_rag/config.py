"""Shared configuration for corpus processing."""

from dataclasses import dataclass
from importlib import import_module
from typing import Any

MIN_SECTION_TEXT_LENGTH = 20
DEFAULT_EMBEDDING_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
DEFAULT_EMBEDDING_BATCH_SIZE = 64
DEFAULT_GENERATION_MODEL = "Qwen/Qwen3-4B-Instruct-2507"

_LEGACY_PYTHON_CONSTANTS = frozenset(
    {"PYTHON_DOC_VERSION", "PYTHON_DOC_BASE_URL", "TARGET_CATEGORIES"}
)


def __getattr__(name: str) -> Any:
    """Lazily expose historical Python defaults without coupling shared config."""
    if name not in _LEGACY_PYTHON_CONSTANTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    constants = import_module("python_doc_rag.sites.python_docs.constants")
    value = getattr(constants, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazily provided compatibility constants in introspection."""
    return sorted(set(globals()).union(_LEGACY_PYTHON_CONSTANTS))


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """Settings for splitting long document sections."""

    chunk_size: int = 1000
    chunk_overlap: int = 150

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must not be negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
