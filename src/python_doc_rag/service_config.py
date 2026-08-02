"""Strict configuration for one multi-knowledge-base RAG service."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from python_doc_rag.profiles import RuntimeProfile, runtime_profile

SERVICE_CONFIG_REVISION = "multi-kb-service-v1"
_TOP_LEVEL_KEYS = frozenset(
    {"revision", "profile", "device", "knowledge_bases"}
)
_KNOWLEDGE_BASE_KEYS = frozenset({"id", "display_name", "data_root"})
_KNOWLEDGE_BASE_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
_DEVICES = frozenset({"auto", "cpu", "cuda"})


class ServiceConfigError(ValueError):
    """Represent an invalid or unreadable multi-knowledge-base config."""


@dataclass(frozen=True, slots=True)
class KnowledgeBaseConfig:
    """One validated knowledge base with a canonical local data root."""

    id: str
    display_name: str
    data_root: Path


@dataclass(frozen=True, slots=True)
class ServiceConfig:
    """One immutable server-wide profile and ordered knowledge-base registry."""

    revision: str
    profile: RuntimeProfile
    device: Literal["auto", "cpu", "cuda"]
    knowledge_bases: tuple[KnowledgeBaseConfig, ...]


def load_service_config(path: Path) -> ServiceConfig:
    """Load a strict TOML service config and resolve its knowledge-base roots."""
    config_path = _resolve_config_path(path)
    try:
        raw = config_path.read_bytes()
    except OSError as error:
        raise ServiceConfigError(
            f"service configを読み取れません: {config_path}: {error}"
        ) from error
    try:
        data = tomllib.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ServiceConfigError(
            f"service config TOMLが不正です: {config_path}: {error}"
        ) from error
    if not isinstance(data, dict):
        raise ServiceConfigError("service configはTOML tableである必要があります。")

    _require_exact_keys(data, _TOP_LEVEL_KEYS, "service config")
    revision = _required_string(data, "revision", "service config")
    if revision != SERVICE_CONFIG_REVISION:
        raise ServiceConfigError(
            "service config revisionが未対応です: "
            f"{revision!r}（必要: {SERVICE_CONFIG_REVISION!r}）"
        )

    profile_name = _required_string(data, "profile", "service config")
    try:
        profile = runtime_profile(profile_name)
    except ValueError as error:
        raise ServiceConfigError(str(error)) from error

    device = _required_string(data, "device", "service config")
    if device not in _DEVICES:
        raise ServiceConfigError(
            "service config.deviceはauto、cpu、cudaのいずれかである必要が"
            f"あります: {device!r}"
        )

    knowledge_bases = _parse_knowledge_bases(
        data["knowledge_bases"],
        config_directory=config_path.parent,
    )
    return ServiceConfig(
        revision=revision,
        profile=profile,
        device=device,
        knowledge_bases=knowledge_bases,
    )


def _resolve_config_path(path: Path) -> Path:
    try:
        return path.expanduser().resolve()
    except (OSError, RuntimeError) as error:
        raise ServiceConfigError(
            f"service config pathを解決できません: {path}: {error}"
        ) from error


def _parse_knowledge_bases(
    value: object,
    *,
    config_directory: Path,
) -> tuple[KnowledgeBaseConfig, ...]:
    if not isinstance(value, list):
        raise ServiceConfigError(
            "service config.knowledge_basesはTOML table arrayである必要があります。"
        )
    if not value:
        raise ServiceConfigError(
            "service config.knowledge_basesは1件以上必要です。"
        )

    parsed: list[KnowledgeBaseConfig] = []
    seen_ids: set[str] = set()
    seen_roots: set[Path] = set()
    for position, item in enumerate(value):
        label = f"knowledge_bases[{position}]"
        if not isinstance(item, dict):
            raise ServiceConfigError(f"{label}はTOML tableである必要があります。")
        _require_exact_keys(item, _KNOWLEDGE_BASE_KEYS, label)

        knowledge_base_id = _required_string(item, "id", label)
        if not _KNOWLEDGE_BASE_ID_PATTERN.fullmatch(knowledge_base_id):
            raise ServiceConfigError(
                f"{label}.idは[a-z0-9][a-z0-9_-]{{0,63}}に一致する必要が"
                f"あります: {knowledge_base_id!r}"
            )
        if knowledge_base_id in seen_ids:
            raise ServiceConfigError(
                f"knowledge base idが重複しています: {knowledge_base_id!r}"
            )

        display_name = _required_string(item, "display_name", label)
        data_root_text = _required_string(item, "data_root", label)
        data_root = _resolve_data_root(
            data_root_text,
            config_directory=config_directory,
            label=label,
        )
        if data_root in seen_roots:
            raise ServiceConfigError(
                "knowledge base data_rootが重複しています: "
                f"{knowledge_base_id!r}"
            )

        seen_ids.add(knowledge_base_id)
        seen_roots.add(data_root)
        parsed.append(
            KnowledgeBaseConfig(
                id=knowledge_base_id,
                display_name=display_name,
                data_root=data_root,
            )
        )
    return tuple(parsed)


def _resolve_data_root(
    value: str,
    *,
    config_directory: Path,
    label: str,
) -> Path:
    if "\x00" in value:
        raise ServiceConfigError(f"{label}.data_rootにNUL文字は使用できません。")
    try:
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = config_directory / candidate
        return candidate.resolve()
    except (OSError, RuntimeError) as error:
        raise ServiceConfigError(
            f"{label}.data_rootを解決できません: {error}"
        ) from error


def _require_exact_keys(
    table: dict[str, Any],
    expected: frozenset[str],
    label: str,
) -> None:
    actual = set(table)
    unknown = sorted(actual.difference(expected))
    missing = sorted(expected.difference(actual))
    if unknown or missing:
        raise ServiceConfigError(
            f"{label} schemaが不正です: unknown={unknown}, missing={missing}"
        )


def _required_string(table: dict[str, Any], key: str, label: str) -> str:
    value = table[key]
    if not isinstance(value, str):
        raise ServiceConfigError(f"{label}.{key}は文字列である必要があります。")
    normalized = value.strip()
    if not normalized:
        raise ServiceConfigError(f"{label}.{key}を空にすることはできません。")
    return normalized
