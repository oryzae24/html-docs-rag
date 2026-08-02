"""Stable fingerprints for source acquisition and downstream processing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any

from python_doc_rag.models import SourceDocument
from python_doc_rag.site_config import (
    BoundedHttpLoaderSettings,
    LoaderSettings,
    LocalCompatibilityLoaderSettings,
    PinnedLocalArchiveLoaderSettings,
    SiteConfig,
    SnapshotHttpArchiveLoaderSettings,
    resolve_loader_settings,
    resolve_parser_settings,
)

SOURCE_CONFIG_FINGERPRINT_REVISION = "source-config-v1"
SOURCE_SNAPSHOT_FINGERPRINT_REVISION = "source-snapshot-v1"
PROCESSING_CONFIG_FINGERPRINT_REVISION = "processing-config-v1"


def canonical_json_sha256(value: Any) -> str:
    """Hash one JSON-compatible value using a stable UTF-8 serialization."""
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def source_config_payload(
    settings: LoaderSettings,
    *,
    category: str,
) -> dict[str, Any]:
    """Return only settings that can change emitted source documents."""
    settings = resolve_loader_settings(settings)
    if isinstance(settings, BoundedHttpLoaderSettings):
        identity: dict[str, Any] = {
            "type": settings.type,
            "start_urls": list(settings.start_urls),
            "max_pages": settings.max_pages,
            "max_response_bytes": settings.max_response_bytes,
            "respect_robots_txt": settings.respect_robots_txt,
            "retain_query": settings.retain_query,
            "user_agent": settings.user_agent,
            "category": category,
        }
    elif isinstance(settings, LocalCompatibilityLoaderSettings):
        identity = {
            "type": settings.type,
            "source_base_url": settings.source_base_url,
            "include_path_prefixes": list(settings.include_path_prefixes),
        }
    elif isinstance(settings, PinnedLocalArchiveLoaderSettings):
        identity = {
            "type": settings.type,
            "archive_path": settings.archive_path,
            "archive_sha256": settings.archive_sha256,
            "archive_format": settings.archive_format,
            "archive_root": settings.archive_root,
            "original_archive_url": settings.original_archive_url,
            "source_base_url": settings.source_base_url,
            "include_path_prefixes": list(settings.include_path_prefixes),
        }
    elif isinstance(settings, SnapshotHttpArchiveLoaderSettings):
        identity = {
            "type": settings.type,
            "archive_url": settings.archive_url,
            "archive_format": settings.archive_format,
            "archive_root": settings.archive_root,
            "source_base_url": settings.source_base_url,
            "include_path_prefixes": list(settings.include_path_prefixes),
            "update_policy": settings.update_policy,
        }
    else:  # pragma: no cover - exhaustive defensive boundary
        raise TypeError(f"unsupported loader settings: {type(settings).__name__}")
    return {
        "revision": SOURCE_CONFIG_FINGERPRINT_REVISION,
        "source": identity,
    }


def source_config_sha256(settings: LoaderSettings, *, category: str) -> str:
    """Hash the acquisition identity without timeout/retry formatting noise."""
    return canonical_json_sha256(source_config_payload(settings, category=category))


def source_snapshot_sha256(documents: tuple[SourceDocument, ...]) -> str:
    """Hash ordered portable source identities for local trees and crawls."""
    pages = [
        {
            "source_url": item.source_url,
            "canonical_url": item.canonical_url,
            "logical_path": item.logical_path,
            "category": item.category,
            "content_sha256": item.content_sha256,
            "source_kind": item.source_kind,
        }
        for item in documents
    ]
    return canonical_json_sha256(
        {"revision": SOURCE_SNAPSHOT_FINGERPRINT_REVISION, "pages": pages}
    )


def processing_config_payload(config: SiteConfig) -> dict[str, Any]:
    """Return every configured value that can alter derived artifacts."""
    parser_payload = asdict(resolve_parser_settings(config.parser))
    return {
        "revision": PROCESSING_CONFIG_FINGERPRINT_REVISION,
        "dataset": asdict(config.dataset),
        "parser": parser_payload,
        "chunking": asdict(config.chunking),
        "index": asdict(config.index),
        "profile": asdict(config.profile),
    }


def processing_config_sha256(config: SiteConfig) -> str:
    """Hash parser, chunking, index, profile, and emitted dataset metadata."""
    return canonical_json_sha256(processing_config_payload(config))
