"""Compatibility loader for caller-supplied expanded Python HTML trees."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from python_doc_rag.loaders.urls import build_source_url
from python_doc_rag.models import SourceDocument
from python_doc_rag.sites.python_docs.constants import (
    PYTHON_DOC_BASE_URL,
    PYTHON_DOC_VERSION,
    TARGET_CATEGORIES,
)


@dataclass(frozen=True, slots=True)
class LocalHtmlEnumeration:
    """Portable diagnostics for deterministic local HTML enumeration."""

    document_root: Path
    files: tuple[Path, ...]
    available_html_count: int
    missing_prefixes: tuple[str, ...]
    skipped_unsafe_paths: tuple[str, ...]

    @property
    def missing_categories(self) -> tuple[str, ...]:
        """Return the historical name for missing top-level prefixes."""
        return tuple(item.rstrip("/") for item in self.missing_prefixes)


class LocalHtmlTreeLoader:
    """Load an expanded HTML tree using configured URL and path boundaries."""

    def __init__(
        self,
        document_root: Path,
        *,
        source_base_url: str | None = None,
        include_path_prefixes: Sequence[str] | None = None,
        categories: Sequence[str] | None = None,
        test_mode: bool = False,
        max_files_per_prefix: int | None = None,
        max_files_per_category: int | None = None,
    ) -> None:
        if categories is not None and (
            source_base_url is not None or include_path_prefixes is not None
        ):
            raise ValueError(
                "categories must not be combined with source_base_url or "
                "include_path_prefixes"
            )
        if max_files_per_prefix is not None and max_files_per_category is not None:
            raise ValueError(
                "max_files_per_prefix and max_files_per_category are aliases; "
                "provide only one"
            )
        legacy_mode = source_base_url is None and include_path_prefixes is None
        if not legacy_mode and (
            source_base_url is None or include_path_prefixes is None
        ):
            raise ValueError(
                "source_base_url and include_path_prefixes must be provided together"
            )
        if legacy_mode:
            legacy_categories = tuple(
                TARGET_CATEGORIES if categories is None else categories
            )
            _validate_legacy_categories(legacy_categories)
            source_base_url = PYTHON_DOC_BASE_URL
            include_path_prefixes = tuple(f"{item}/" for item in legacy_categories)
        self._document_root = document_root
        self._source_base_url = source_base_url
        self._prefixes = tuple(include_path_prefixes)
        self._test_mode = test_mode
        self._max_files_per_prefix = (
            max_files_per_category
            if max_files_per_category is not None
            else max_files_per_prefix
            if max_files_per_prefix is not None
            else 3
        )
        self._legacy_mode = legacy_mode
        self._enumeration: LocalHtmlEnumeration | None = None

    @property
    def enumeration(self) -> LocalHtmlEnumeration | None:
        """Return diagnostics after enumeration has started."""
        return self._enumeration

    def load(self) -> Iterator[SourceDocument]:
        """Yield selected local HTML in global logical-path order."""
        enumeration = _enumerate_local_html(
            self._document_root,
            self._prefixes,
            test_mode=self._test_mode,
            max_files_per_prefix=self._max_files_per_prefix,
        )
        self._enumeration = enumeration
        for html_path in enumeration.files:
            logical_path = html_path.relative_to(enumeration.document_root).as_posix()
            content = html_path.read_text(encoding="utf-8")
            source_url = build_source_url(self._source_base_url, logical_path)
            yield SourceDocument(
                source_url=source_url,
                canonical_url=source_url,
                content=content,
                content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
                source_kind="local-html-tree",
                logical_path=logical_path,
                category=PurePosixPath(logical_path).parts[0],
                metadata=(
                    {"python_version": PYTHON_DOC_VERSION} if self._legacy_mode else {}
                ),
            )


def _validate_legacy_categories(categories: Sequence[str]) -> None:
    if not categories:
        raise ValueError("categories must not be empty")
    for category in categories:
        if (
            not isinstance(category, str)
            or not category.strip()
            or "/" in category
            or "\\" in category
            or category in {".", ".."}
        ):
            raise ValueError(f"unsafe category: {category!r}")


def _enumerate_local_html(
    document_root: Path,
    prefixes: Sequence[str],
    *,
    test_mode: bool,
    max_files_per_prefix: int,
) -> LocalHtmlEnumeration:
    if not prefixes:
        raise ValueError("include_path_prefixes must not be empty")
    if test_mode and max_files_per_prefix < 1:
        raise ValueError("max_files_per_prefix must be greater than zero")
    root = document_root.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"document_root is not a directory: {document_root}")
    resolved_root = root.resolve()
    selected: dict[str, Path] = {}
    missing: list[str] = []
    skipped: list[str] = []
    available = 0
    for prefix in prefixes:
        relative = PurePosixPath(prefix.rstrip("/"))
        prefix_root = root.joinpath(*relative.parts)
        if not prefix_root.is_dir() or prefix_root.is_symlink():
            missing.append(prefix)
            continue
        candidates: list[tuple[str, Path]] = []
        for candidate in prefix_root.rglob("*.html"):
            if candidate.is_symlink() or not candidate.is_file():
                continue
            try:
                resolved = candidate.resolve(strict=True)
                logical = resolved.relative_to(resolved_root).as_posix()
            except (OSError, ValueError):
                try:
                    skipped.append(candidate.relative_to(root).as_posix())
                except ValueError:
                    skipped.append(candidate.name)
                continue
            candidates.append((logical, resolved))
        candidates.sort(key=lambda item: item[0])
        available += len(candidates)
        if test_mode:
            candidates = candidates[:max_files_per_prefix]
        for logical, resolved in candidates:
            selected[logical] = resolved
    ordered = tuple(selected[key] for key in sorted(selected))
    return LocalHtmlEnumeration(
        document_root=root,
        files=ordered,
        available_html_count=available,
        missing_prefixes=tuple(missing),
        skipped_unsafe_paths=tuple(sorted(skipped)),
    )
