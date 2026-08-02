"""Portable source URL construction for local and archive loaders."""

from __future__ import annotations

from pathlib import PurePosixPath
from urllib.parse import quote, urlsplit, urlunsplit


def safe_logical_path(value: str) -> str:
    """Normalize one relative POSIX path without permitting traversal."""
    if not value or "\x00" in value or "\\" in value:
        raise ValueError("logical path must be a non-empty POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or path == PurePosixPath(".") or ".." in path.parts:
        raise ValueError("logical path must be a safe relative POSIX path")
    normalized = path.as_posix()
    if normalized != value or "." in path.parts:
        raise ValueError("logical path must already be normalized")
    return normalized


def build_source_url(source_base_url: str, logical_path: str) -> str:
    """Append a quoted logical path while remaining inside the configured base."""
    base = urlsplit(source_base_url)
    if base.scheme not in {"http", "https"} or not base.hostname:
        raise ValueError("source_base_url must be an absolute HTTP(S) URL")
    if base.username is not None or base.password is not None:
        raise ValueError("source_base_url authentication is not allowed")
    if base.query or base.fragment or not base.path.endswith("/"):
        raise ValueError("source_base_url must be a query-free directory URL")
    relative = safe_logical_path(logical_path)
    path = f"{base.path}{quote(relative, safe='/-._~')}"
    result = urlunsplit((base.scheme, base.netloc, path, "", ""))
    parsed = urlsplit(result)
    if (parsed.scheme, parsed.netloc) != (base.scheme, base.netloc):
        raise ValueError("generated source URL escaped its configured origin")
    if not parsed.path.startswith(base.path):
        raise ValueError("generated source URL escaped its configured base path")
    return result
