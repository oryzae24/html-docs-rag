"""Strict TOML configuration for one static documentation dataset."""

from __future__ import annotations

import hashlib
import math
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from urllib.parse import urlsplit

import soupsieve

import python_doc_rag.config as shared_config
from python_doc_rag.config import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    MIN_SECTION_TEXT_LENGTH,
    ChunkingConfig,
)

SITE_CONFIG_REVISION = "site-config-v2"
MAX_CONFIGURED_PAGES = 200
MAX_ARCHIVE_BYTES_LIMIT = 2_000_000_000
MAX_ARCHIVE_MEMBERS_LIMIT = 1_000_000
MAX_EXTRACTED_BYTES_LIMIT = 8_000_000_000
_SLUG_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_WINDOWS_DRIVE_PATTERN = re.compile(r"^[A-Za-z]:")
_TOP_LEVEL_KEYS = frozenset(
    {"dataset", "loader", "parser", "chunking", "index", "profile"}
)


@dataclass(frozen=True, slots=True)
class DatasetSettings:
    """Portable identity and display metadata for one dataset."""

    name: str
    slug: str
    language: str
    description: str


@dataclass(frozen=True, slots=True)
class LocalLoaderSettings:
    """Historical settings for an expanded local Python documentation tree."""

    type: Literal["local-html-tree"]
    categories: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LocalCompatibilityLoaderSettings:
    """Settings for a caller-supplied expanded local HTML tree."""

    type: Literal["local-html-tree"]
    source_base_url: str
    include_path_prefixes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BoundedHttpLoaderSettings:
    """Bounded HTTP crawling and cache settings."""

    type: Literal["bounded-http"]
    start_urls: tuple[str, ...]
    max_pages: int
    timeout_seconds: float
    request_delay_seconds: float
    max_response_bytes: int
    user_agent: str
    respect_robots_txt: bool
    retain_query: bool
    max_retries: int


# Historical import name retained for callers.
HttpLoaderSettings = BoundedHttpLoaderSettings


@dataclass(frozen=True, slots=True)
class PinnedLocalArchiveLoaderSettings:
    """A repository-local archive pinned by its exact byte digest."""

    type: Literal["pinned-local-archive"]
    archive_path: str
    archive_sha256: str
    archive_format: Literal["zip"]
    archive_root: str
    original_archive_url: str
    source_base_url: str
    include_path_prefixes: tuple[str, ...]
    max_archive_bytes: int
    max_members: int
    max_member_bytes: int
    max_extracted_bytes: int


@dataclass(frozen=True, slots=True)
class SnapshotHttpArchiveLoaderSettings:
    """A mutable HTTPS archive locked to bytes on first acquisition."""

    type: Literal["snapshot-http-archive"]
    archive_url: str
    archive_format: Literal["zip"]
    archive_root: str
    source_base_url: str
    include_path_prefixes: tuple[str, ...]
    update_policy: Literal["manual"]
    timeout_seconds: float
    max_retries: int
    max_archive_bytes: int
    max_members: int
    max_member_bytes: int
    max_extracted_bytes: int
    user_agent: str


ResolvedLoaderSettings = (
    LocalCompatibilityLoaderSettings
    | BoundedHttpLoaderSettings
    | PinnedLocalArchiveLoaderSettings
    | SnapshotHttpArchiveLoaderSettings
)
LoaderSettings = LocalLoaderSettings | ResolvedLoaderSettings


@dataclass(frozen=True, slots=True)
class ParserSettings:
    """Historical settings shared by Python and generic HTML parsers."""

    type: Literal["python-sphinx", "generic-html"]
    content_selectors: tuple[str, ...]
    exclude_selectors: tuple[str, ...]
    title_selectors: tuple[str, ...]
    heading_levels: tuple[int, ...]
    minimum_section_text_length: int
    fallback_to_body: bool
    include_lead_text: bool


@dataclass(frozen=True, slots=True)
class PythonSphinxParserSettings:
    """Settings owned exclusively by the Python/Sphinx parser."""

    type: Literal["python-sphinx"]
    python_version: str
    minimum_section_text_length: int


@dataclass(frozen=True, slots=True)
class GenericHtmlParserSettings:
    """Selector-driven settings for the site-neutral generic HTML parser."""

    type: Literal["generic-html"]
    content_selectors: tuple[str, ...]
    exclude_selectors: tuple[str, ...]
    title_selectors: tuple[str, ...]
    heading_levels: tuple[int, ...]
    minimum_section_text_length: int
    fallback_to_body: bool
    include_lead_text: bool


ResolvedParserSettings = PythonSphinxParserSettings | GenericHtmlParserSettings


@dataclass(frozen=True, slots=True)
class IndexSettings:
    """Pinned baseline dense-index build settings."""

    embedding_model: str
    embedding_batch_size: int


@dataclass(frozen=True, slots=True)
class ProfilePreparationSettings:
    """Optional algorithm profile artifacts to prepare for this dataset."""

    prepare: Literal["none", "recommended-v2"]


@dataclass(frozen=True, slots=True)
class SiteConfig:
    """Validated, order-independent site configuration and source hash."""

    revision: str
    config_sha256: str
    dataset: DatasetSettings
    loader: ResolvedLoaderSettings
    parser: ResolvedParserSettings
    chunking: ChunkingConfig
    index: IndexSettings
    profile: ProfilePreparationSettings


def resolve_loader_settings(settings: LoaderSettings) -> ResolvedLoaderSettings:
    """Convert a historical local-loader object to current generic settings."""
    if isinstance(settings, LocalLoaderSettings):
        _validate_legacy_category_values(settings.categories)
        return LocalCompatibilityLoaderSettings(
            type="local-html-tree",
            source_base_url=_legacy_python_base_url(),
            include_path_prefixes=tuple(f"{item}/" for item in settings.categories),
        )
    return settings


def resolve_parser_settings(
    settings: ParserSettings | ResolvedParserSettings,
) -> ResolvedParserSettings:
    """Convert a historical parser object to its current concrete variant."""
    if not isinstance(settings, ParserSettings):
        return settings
    if settings.type == "python-sphinx":
        return PythonSphinxParserSettings(
            type="python-sphinx",
            python_version=_legacy_python_version(),
            minimum_section_text_length=settings.minimum_section_text_length,
        )
    if settings.type == "generic-html":
        return GenericHtmlParserSettings(
            type="generic-html",
            content_selectors=settings.content_selectors,
            exclude_selectors=settings.exclude_selectors,
            title_selectors=settings.title_selectors,
            heading_levels=settings.heading_levels,
            minimum_section_text_length=settings.minimum_section_text_length,
            fallback_to_body=settings.fallback_to_body,
            include_lead_text=settings.include_lead_text,
        )
    raise ValueError(f"unsupported parser.type: {settings.type}")


def load_site_config(path: Path) -> SiteConfig:
    """Load one strict TOML file and reject unknown or unsafe values."""
    raw = path.expanduser().read_bytes()
    try:
        data = tomllib.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"invalid site config {path}: {error}") from error
    if not isinstance(data, dict):
        raise ValueError("site config must be a TOML table")
    _reject_unknown(data, _TOP_LEVEL_KEYS, "top level")
    missing = _TOP_LEVEL_KEYS.difference(data)
    if missing:
        raise ValueError(f"missing top-level config sections: {sorted(missing)}")
    return SiteConfig(
        revision=SITE_CONFIG_REVISION,
        config_sha256=hashlib.sha256(raw).hexdigest(),
        dataset=_parse_dataset(_table(data, "dataset")),
        loader=_parse_loader(_table(data, "loader")),
        parser=_parse_parser(_table(data, "parser")),
        chunking=_parse_chunking(_table(data, "chunking")),
        index=_parse_index(_table(data, "index")),
        profile=_parse_profile(_table(data, "profile")),
    )


def _parse_dataset(table: dict[str, Any]) -> DatasetSettings:
    _reject_unknown(table, {"name", "slug", "language", "description"}, "dataset")
    name = _string(table, "name")
    slug = _string(table, "slug")
    language = _string(table, "language")
    description = _string(table, "description")
    if not _SLUG_PATTERN.fullmatch(slug):
        raise ValueError("dataset.slug must be lowercase kebab-case")
    return DatasetSettings(name, slug, language, description)


def _parse_loader(table: dict[str, Any]) -> ResolvedLoaderSettings:
    loader_type = _string(table, "type")
    if loader_type == "local-html-tree":
        new_keys = {"source_base_url", "include_path_prefixes"}
        if "categories" in table or not new_keys.intersection(table):
            _reject_unknown(table, {"type", "categories"}, "loader")
            categories = _legacy_categories(table.get("categories"))
            return LocalCompatibilityLoaderSettings(
                type="local-html-tree",
                source_base_url=_legacy_python_base_url(),
                include_path_prefixes=tuple(f"{item}/" for item in categories),
            )
        _reject_unknown(
            table,
            {"type", "source_base_url", "include_path_prefixes"},
            "loader",
        )
        return LocalCompatibilityLoaderSettings(
            type="local-html-tree",
            source_base_url=_source_base_url(table, "source_base_url"),
            include_path_prefixes=_path_prefixes(table.get("include_path_prefixes")),
        )
    if loader_type == "bounded-http":
        return _parse_bounded_http_loader(table)
    if loader_type == "pinned-local-archive":
        return _parse_pinned_local_archive_loader(table)
    if loader_type == "snapshot-http-archive":
        return _parse_snapshot_http_archive_loader(table)
    raise ValueError(f"unsupported loader.type: {loader_type}")


def _parse_bounded_http_loader(table: dict[str, Any]) -> BoundedHttpLoaderSettings:
    allowed = {
        "type",
        "start_urls",
        "max_pages",
        "timeout_seconds",
        "request_delay_seconds",
        "max_response_bytes",
        "user_agent",
        "respect_robots_txt",
        "retain_query",
        "max_retries",
    }
    _reject_unknown(table, allowed, "loader")
    start_urls = _unique_string_list(table.get("start_urls"), "start_urls")
    retain_query = _boolean(table, "retain_query", default=False)
    for value in start_urls:
        _validate_start_url(value, retain_query=retain_query)
    max_pages = _integer(table, "max_pages", default=40, minimum=1)
    if max_pages > MAX_CONFIGURED_PAGES:
        raise ValueError(f"loader.max_pages must not exceed {MAX_CONFIGURED_PAGES}")
    delay = _number(table, "request_delay_seconds", default=0.5)
    if delay < 0:
        raise ValueError("loader.request_delay_seconds must not be negative")
    timeout = _bounded_timeout(table)
    max_retries = _bounded_retries(table)
    return BoundedHttpLoaderSettings(
        type="bounded-http",
        start_urls=start_urls,
        max_pages=max_pages,
        timeout_seconds=timeout,
        request_delay_seconds=delay,
        max_response_bytes=_bounded_positive_integer(
            table,
            "max_response_bytes",
            default=3_000_000,
            maximum=100_000_000,
        ),
        user_agent=_string(table, "user_agent"),
        respect_robots_txt=_boolean(table, "respect_robots_txt", default=True),
        retain_query=retain_query,
        max_retries=max_retries,
    )


def _parse_pinned_local_archive_loader(
    table: dict[str, Any],
) -> PinnedLocalArchiveLoaderSettings:
    allowed = {
        "type",
        "archive_path",
        "archive_sha256",
        "archive_format",
        "archive_root",
        "original_archive_url",
        "source_base_url",
        "include_path_prefixes",
        "max_archive_bytes",
        "max_members",
        "max_member_bytes",
        "max_extracted_bytes",
    }
    _reject_unknown(table, allowed, "loader")
    archive_path = _string(table, "archive_path")
    _validate_config_relative_archive_path(archive_path)
    common = _archive_common(table)
    return PinnedLocalArchiveLoaderSettings(
        type="pinned-local-archive",
        archive_path=archive_path,
        archive_sha256=_sha256(table, "archive_sha256"),
        original_archive_url=_absolute_url(
            table,
            "original_archive_url",
            schemes={"https"},
        ),
        max_archive_bytes=_bounded_positive_integer(
            table,
            "max_archive_bytes",
            maximum=MAX_ARCHIVE_BYTES_LIMIT,
        ),
        **common,
    )


def _parse_snapshot_http_archive_loader(
    table: dict[str, Any],
) -> SnapshotHttpArchiveLoaderSettings:
    allowed = {
        "type",
        "archive_url",
        "archive_format",
        "archive_root",
        "source_base_url",
        "include_path_prefixes",
        "update_policy",
        "timeout_seconds",
        "max_retries",
        "max_archive_bytes",
        "max_members",
        "max_member_bytes",
        "max_extracted_bytes",
        "user_agent",
    }
    _reject_unknown(table, allowed, "loader")
    common = _archive_common(table)
    policy = _string(table, "update_policy")
    if policy != "manual":
        raise ValueError("loader.update_policy must be manual")
    return SnapshotHttpArchiveLoaderSettings(
        type="snapshot-http-archive",
        archive_url=_archive_url(table, "archive_url"),
        update_policy="manual",
        timeout_seconds=_bounded_timeout(table),
        max_retries=_bounded_retries(table),
        max_archive_bytes=_bounded_positive_integer(
            table,
            "max_archive_bytes",
            maximum=MAX_ARCHIVE_BYTES_LIMIT,
        ),
        user_agent=_string(table, "user_agent"),
        **common,
    )


def _archive_common(table: dict[str, Any]) -> dict[str, Any]:
    archive_format = _string(table, "archive_format")
    if archive_format != "zip":
        raise ValueError("loader.archive_format must be zip")
    archive_root = _safe_relative_posix_path(
        _string(table, "archive_root"),
        "archive_root",
        allow_trailing_slash=False,
    )
    max_member_bytes = _bounded_positive_integer(
        table,
        "max_member_bytes",
        maximum=MAX_EXTRACTED_BYTES_LIMIT,
    )
    max_extracted_bytes = _bounded_positive_integer(
        table,
        "max_extracted_bytes",
        maximum=MAX_EXTRACTED_BYTES_LIMIT,
    )
    if max_member_bytes > max_extracted_bytes:
        raise ValueError("loader.max_member_bytes must not exceed max_extracted_bytes")
    return {
        "archive_format": "zip",
        "archive_root": archive_root,
        "source_base_url": _source_base_url(table, "source_base_url"),
        "include_path_prefixes": _path_prefixes(table.get("include_path_prefixes")),
        "max_members": _bounded_positive_integer(
            table,
            "max_members",
            maximum=MAX_ARCHIVE_MEMBERS_LIMIT,
        ),
        "max_member_bytes": max_member_bytes,
        "max_extracted_bytes": max_extracted_bytes,
    }


def _parse_parser(table: dict[str, Any]) -> ResolvedParserSettings:
    parser_type = _string(table, "type")
    if parser_type == "python-sphinx":
        _reject_unknown(
            table,
            {"type", "python_version", "minimum_section_text_length"},
            "parser",
        )
        return PythonSphinxParserSettings(
            type="python-sphinx",
            python_version=_string(
                table,
                "python_version",
                default=_legacy_python_version(),
            ),
            minimum_section_text_length=_integer(
                table,
                "minimum_section_text_length",
                default=MIN_SECTION_TEXT_LENGTH,
                minimum=1,
            ),
        )
    if parser_type != "generic-html":
        raise ValueError(f"unsupported parser.type: {parser_type}")
    allowed = {
        "type",
        "content_selectors",
        "exclude_selectors",
        "title_selectors",
        "heading_levels",
        "minimum_section_text_length",
        "fallback_to_body",
        "include_lead_text",
    }
    _reject_unknown(table, allowed, "parser")
    content = _selector_list(table.get("content_selectors"), "content_selectors")
    if not content:
        raise ValueError("parser.content_selectors must not be empty")
    headings = _integer_list(table.get("heading_levels", [1, 2, 3]), "heading_levels")
    if not headings or any(level < 1 or level > 6 for level in headings):
        raise ValueError("parser.heading_levels must contain values from 1 through 6")
    return GenericHtmlParserSettings(
        type="generic-html",
        content_selectors=content,
        exclude_selectors=_selector_list(
            table.get("exclude_selectors", []), "exclude_selectors"
        ),
        title_selectors=_selector_list(
            table.get("title_selectors", ["title", "h1"]), "title_selectors"
        ),
        heading_levels=tuple(dict.fromkeys(headings)),
        minimum_section_text_length=_integer(
            table,
            "minimum_section_text_length",
            default=MIN_SECTION_TEXT_LENGTH,
            minimum=1,
        ),
        fallback_to_body=_boolean(table, "fallback_to_body", default=False),
        include_lead_text=_boolean(table, "include_lead_text", default=True),
    )


def _parse_chunking(table: dict[str, Any]) -> ChunkingConfig:
    _reject_unknown(table, {"chunk_size", "chunk_overlap"}, "chunking")
    return ChunkingConfig(
        chunk_size=_integer(table, "chunk_size", minimum=1),
        chunk_overlap=_integer(table, "chunk_overlap", minimum=0),
    )


def _parse_index(table: dict[str, Any]) -> IndexSettings:
    _reject_unknown(table, {"embedding_model", "embedding_batch_size"}, "index")
    return IndexSettings(
        embedding_model=_string(
            table,
            "embedding_model",
            default=DEFAULT_EMBEDDING_MODEL,
        ),
        embedding_batch_size=_integer(
            table,
            "embedding_batch_size",
            default=DEFAULT_EMBEDDING_BATCH_SIZE,
            minimum=1,
        ),
    )


def _parse_profile(table: dict[str, Any]) -> ProfilePreparationSettings:
    _reject_unknown(table, {"prepare"}, "profile")
    value = _string(table, "prepare", default="none")
    if value not in {"none", "recommended-v2"}:
        raise ValueError("profile.prepare must be none or recommended-v2")
    return ProfilePreparationSettings(value)  # type: ignore[arg-type]


def _validate_start_url(value: str, *, retain_query: bool) -> None:
    parsed = _parsed_absolute_url(value, schemes={"http", "https"}, label="start URL")
    if parsed.fragment:
        raise ValueError("start URL must not contain a fragment")
    if parsed.query and not retain_query:
        raise ValueError("start URL query requires loader.retain_query=true")


def _archive_url(table: dict[str, Any], key: str) -> str:
    value = _absolute_url(table, key, schemes={"https"})
    parsed = urlsplit(value)
    if parsed.query or parsed.fragment:
        raise ValueError(f"loader.{key} must not contain a query or fragment")
    return value


def _source_base_url(table: dict[str, Any], key: str) -> str:
    value = _absolute_url(table, key, schemes={"http", "https"})
    parsed = urlsplit(value)
    if parsed.query or parsed.fragment:
        raise ValueError(f"loader.{key} must not contain a query or fragment")
    if not parsed.path.endswith("/"):
        raise ValueError(f"loader.{key} must end with /")
    return value


def _absolute_url(table: dict[str, Any], key: str, *, schemes: set[str]) -> str:
    value = _string(table, key)
    _parsed_absolute_url(value, schemes=schemes, label=f"loader.{key}")
    return value


def _parsed_absolute_url(value: str, *, schemes: set[str], label: str):  # type: ignore[no-untyped-def]
    parsed = urlsplit(value)
    if parsed.scheme not in schemes or not parsed.hostname:
        expected = "/".join(sorted(item.upper() for item in schemes))
        raise ValueError(f"{label} must be an absolute {expected} URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{label} must not contain authentication information")
    return parsed


def _legacy_categories(value: object) -> tuple[str, ...]:
    if value is None:
        defaults = shared_config.TARGET_CATEGORIES
        if not isinstance(defaults, tuple):  # pragma: no cover - defensive boundary
            raise TypeError("legacy TARGET_CATEGORIES must be a tuple")
        categories = defaults
    else:
        categories = _string_list(value, "categories")
    _validate_legacy_category_values(categories)
    return categories


def _validate_legacy_category_values(categories: tuple[str, ...]) -> None:
    if not categories:
        raise ValueError("loader.categories must not be empty")
    for category in categories:
        if "/" in category or "\\" in category or category in {".", ".."}:
            raise ValueError(f"unsafe loader category: {category!r}")


def _legacy_python_base_url() -> str:
    value = shared_config.PYTHON_DOC_BASE_URL
    if not isinstance(value, str):  # pragma: no cover - defensive boundary
        raise TypeError("legacy PYTHON_DOC_BASE_URL must be a string")
    return value


def _legacy_python_version() -> str:
    value = shared_config.PYTHON_DOC_VERSION
    if not isinstance(value, str):  # pragma: no cover - defensive boundary
        raise TypeError("legacy PYTHON_DOC_VERSION must be a string")
    return value


def _validate_config_relative_archive_path(value: str) -> None:
    if "\x00" in value or "\\" in value or _WINDOWS_DRIVE_PATTERN.match(value):
        raise ValueError("loader.archive_path must be a relative POSIX path")
    if PurePosixPath(value).is_absolute() or value.endswith("/"):
        raise ValueError("loader.archive_path must be a relative archive file path")


def _path_prefixes(value: object) -> tuple[str, ...]:
    raw = _unique_string_list(value, "include_path_prefixes")
    prefixes = tuple(
        _safe_relative_posix_path(
            item,
            "include_path_prefixes",
            allow_trailing_slash=True,
        )
        for item in raw
    )
    normalized = tuple(item.rstrip("/") + "/" for item in prefixes)
    if len(set(normalized)) != len(normalized):
        raise ValueError("loader.include_path_prefixes must not contain duplicates")
    return normalized


def _safe_relative_posix_path(
    value: str,
    label: str,
    *,
    allow_trailing_slash: bool,
) -> str:
    if "\x00" in value or "\\" in value or _WINDOWS_DRIVE_PATTERN.match(value):
        raise ValueError(f"unsafe loader.{label}: {value!r}")
    if value.startswith("/") or value.startswith("//"):
        raise ValueError(f"unsafe loader.{label}: {value!r}")
    trailing = value.endswith("/")
    path = PurePosixPath(value)
    if path == PurePosixPath(".") or ".." in path.parts or "." in path.parts:
        raise ValueError(f"unsafe loader.{label}: {value!r}")
    normalized = path.as_posix()
    if value.rstrip("/") != normalized or (trailing and not allow_trailing_slash):
        raise ValueError(f"loader.{label} must be a normalized relative POSIX path")
    return normalized + ("/" if trailing else "")


def _sha256(table: dict[str, Any], key: str) -> str:
    value = _string(table, key)
    if not _SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"loader.{key} must be 64 lowercase hexadecimal characters")
    return value


def _bounded_timeout(table: dict[str, Any]) -> float:
    timeout = _number(table, "timeout_seconds", default=15.0)
    if timeout <= 0 or timeout > 300:
        raise ValueError("loader.timeout_seconds must be greater than 0 and at most 300")
    return timeout


def _bounded_retries(table: dict[str, Any]) -> int:
    retries = _integer(table, "max_retries", default=2, minimum=0)
    if retries > 5:
        raise ValueError("loader.max_retries must not exceed 5")
    return retries


def _bounded_positive_integer(
    table: dict[str, Any],
    key: str,
    *,
    maximum: int,
    default: int | None = None,
) -> int:
    value = _integer(table, key, default=default, minimum=1)
    if value > maximum:
        raise ValueError(f"loader.{key} must not exceed {maximum}")
    return value


def _selector_list(value: object, label: str) -> tuple[str, ...]:
    selectors = _string_list(value, label)
    for selector in selectors:
        try:
            soupsieve.compile(selector)
        except Exception as error:
            raise ValueError(f"invalid CSS selector in parser.{label}: {selector}") from error
    return selectors


def _table(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"config section {key} must be a table")
    return value


def _reject_unknown(
    table: dict[str, Any],
    allowed: set[str] | frozenset[str],
    label: str,
) -> None:
    unknown = sorted(set(table).difference(allowed))
    if unknown:
        raise ValueError(f"unknown {label} config keys: {unknown}")


def _string(
    table: dict[str, Any],
    key: str,
    *,
    default: str | None = None,
) -> str:
    value = table.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{key} must be a non-empty string")
    return value.strip()


def _string_list(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise TypeError(f"{label} must be a list of non-empty strings")
    return tuple(item.strip() for item in value)


def _unique_string_list(value: object, label: str) -> tuple[str, ...]:
    items = _string_list(value, label)
    if not items:
        raise ValueError(f"loader.{label} must not be empty")
    if len(set(items)) != len(items):
        raise ValueError(f"loader.{label} must not contain duplicates")
    return items


def _integer_list(value: object, label: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    ):
        raise TypeError(f"{label} must be a list of integers")
    return tuple(value)


def _integer(
    table: dict[str, Any],
    key: str,
    *,
    default: int | None = None,
    minimum: int,
) -> int:
    value = table.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise TypeError(f"{key} must be an integer >= {minimum}")
    return value


def _number(table: dict[str, Any], key: str, *, default: float) -> float:
    value = table.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{key} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{key} must be finite")
    return result


def _boolean(table: dict[str, Any], key: str, *, default: bool) -> bool:
    value = table.get(key, default)
    if not isinstance(value, bool):
        raise TypeError(f"{key} must be a boolean")
    return value
