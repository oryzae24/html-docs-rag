"""Document loading boundaries for local and bounded remote HTML sources."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import posixpath
import re
import shutil
import stat
import tempfile
import time
import urllib.error
import urllib.request
import urllib.robotparser
import uuid
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Protocol
from urllib.parse import (
    quote,
    unquote,
    urljoin,
    urlsplit,
    urlunsplit,
)

from bs4 import BeautifulSoup, UnicodeDammit

from python_doc_rag.dataset_layout import (
    DATASET_MANIFEST_REVISION,
    load_dataset_manifest,
)
from python_doc_rag.models import SourceDocument
from python_doc_rag.site_config import BoundedHttpLoaderSettings
from python_doc_rag.source_identity import (
    source_config_sha256,
    source_snapshot_sha256,
)

FETCH_MANIFEST_REVISION = "bounded-http-fetch-v2"
LEGACY_FETCH_MANIFEST_REVISION = "bounded-http-fetch-v1"
_MAX_CACHED_HTML_BYTES = 100_000_000
_MAX_FETCH_MANIFEST_BYTES = 64 * 1024 * 1024
_COMMITTED_FETCH_MANIFEST_KEYS = frozenset(
    {
        "schema_revision",
        "complete",
        "loader_type",
        "cache_reused",
        "offline",
        "source_config_fingerprint_revision",
        "source_config_sha256",
        "start_urls",
        "boundaries",
        "retain_query",
        "max_pages",
        "fetched_page_count",
        "failure_count",
        "duplicate_canonical_count",
        "duplicate_content_count",
        "excluded_external_url_count",
        "pages",
        "failures",
        "finished_at",
        "parser_type",
        "parser_settings",
        "source_page_count",
        "site_config_file_sha256",
        "processing_config_fingerprint_revision",
        "processing_config_sha256",
        "source_snapshot_sha256",
        "configured_source",
    }
)
_PAGE_RECORD_KEYS = frozenset(
    {
        "source_url",
        "canonical_url",
        "logical_path",
        "category",
        "fetched_at",
        "content_sha256",
        "cache_path",
        "http_status",
        "content_type",
        "response_bytes",
    }
)
_NON_HTML_SUFFIXES = frozenset(
    {
        ".7z",
        ".avi",
        ".css",
        ".csv",
        ".doc",
        ".docx",
        ".epub",
        ".gif",
        ".gz",
        ".ico",
        ".jpeg",
        ".jpg",
        ".js",
        ".json",
        ".mp3",
        ".mp4",
        ".pdf",
        ".png",
        ".ppt",
        ".pptx",
        ".svg",
        ".tar",
        ".tgz",
        ".txt",
        ".webp",
        ".xls",
        ".xlsx",
        ".xml",
        ".zip",
    }
)


@dataclass(frozen=True, slots=True)
class CrawlBoundary:
    """One normalized scheme/origin/path-prefix crawl boundary."""

    scheme: str
    netloc: str
    path_prefix: str

    def contains(self, url: str) -> bool:
        """Return whether a normalized URL is within this exact boundary."""
        parsed = urlsplit(url)
        if parsed.scheme != self.scheme or parsed.netloc != self.netloc:
            return False
        prefix = self.path_prefix
        return parsed.path == prefix.rstrip("/") or parsed.path.startswith(prefix)


class CrawlScope:
    """Normalize links and enforce the union of configured start boundaries."""

    def __init__(self, start_urls: Sequence[str], *, retain_query: bool = False) -> None:
        if not start_urls:
            raise ValueError("start_urls must not be empty")
        self._retain_query = retain_query
        normalized = tuple(
            dict.fromkeys(normalize_http_url(value, retain_query=retain_query) for value in start_urls)
        )
        self._start_urls = normalized
        self._boundaries = tuple(_boundary_for(value) for value in normalized)

    @property
    def start_urls(self) -> tuple[str, ...]:
        """Return normalized starts in deterministic configured order."""
        return self._start_urls

    @property
    def boundaries(self) -> tuple[CrawlBoundary, ...]:
        """Return the allowed union derived only from start URLs."""
        return self._boundaries

    def resolve(self, base_url: str, reference: str) -> str | None:
        """Resolve a link and return it only if it is a crawlable in-scope page."""
        try:
            normalized = normalize_http_url(
                urljoin(base_url, reference),
                retain_query=self._retain_query,
            )
        except ValueError:
            return None
        if not self.contains(normalized) or not _is_html_candidate(normalized):
            return None
        return normalized

    def contains(self, url: str) -> bool:
        """Return whether a normalized URL is inside any configured boundary."""
        return any(boundary.contains(url) for boundary in self._boundaries)


@dataclass(frozen=True, slots=True)
class HttpResponse:
    """Bounded HTTP response returned by an injectable transport."""

    requested_url: str
    final_url: str
    status: int
    content_type: str
    body: bytes


class HttpTransport(Protocol):
    """Small HTTP boundary used by the loader and offline Fake tests."""

    def get(
        self,
        url: str,
        *,
        timeout_seconds: float,
        user_agent: str,
        max_response_bytes: int,
    ) -> HttpResponse:
        """Fetch one bounded response or raise a page-local exception."""
        ...


class UrllibHttpTransport:
    """Fetch bounded public pages with standard-library urllib."""

    def get(
        self,
        url: str,
        *,
        timeout_seconds: float,
        user_agent: str,
        max_response_bytes: int,
    ) -> HttpResponse:
        """Fetch one URL and reject bodies above the configured limit."""
        request = urllib.request.Request(
            url,
            headers={"User-Agent": user_agent, "Accept": "text/html,*/*;q=0.1"},
        )
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read(max_response_bytes + 1)
            if len(body) > max_response_bytes:
                raise ValueError("response body exceeded max_response_bytes")
            return HttpResponse(
                requested_url=url,
                final_url=response.geturl(),
                status=int(getattr(response, "status", 200)),
                content_type=response.headers.get("Content-Type", ""),
                body=body,
            )


@dataclass(frozen=True, slots=True)
class FetchFailure:
    """One bounded page failure retained in the fetch manifest."""

    url: str
    reason: str


@dataclass(frozen=True, slots=True)
class HttpLoadSummary:
    """Diagnostics for one completed network crawl or offline replay."""

    fetched_page_count: int
    failure_count: int
    duplicate_canonical_count: int
    duplicate_content_count: int
    excluded_external_url_count: int
    cache_reused: bool
    offline: bool
    failures: tuple[FetchFailure, ...]


class BoundedHttpHtmlLoader:
    """Crawl static HTML within start-derived boundaries and an atomic cache."""

    def __init__(
        self,
        settings: BoundedHttpLoaderSettings,
        cache_root: Path,
        *,
        category: str,
        offline: bool = False,
        refresh: bool = False,
        resume: bool = False,
        transport: HttpTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._settings = settings
        self._cache_root = cache_root.expanduser()
        self._category = category
        self._offline = offline
        self._refresh = refresh
        self._resume = resume
        self._transport = transport or UrllibHttpTransport()
        self._sleep = sleep
        self._now = now or (lambda: datetime.now(UTC))
        self._scope = CrawlScope(
            settings.start_urls,
            retain_query=settings.retain_query,
        )
        self._summary: HttpLoadSummary | None = None
        self._robots: dict[tuple[str, str], urllib.robotparser.RobotFileParser | None] = {}
        self._source_config_sha256 = source_config_sha256(settings, category=category)
        self._cache_backup: Path | None = None

    @property
    def scope(self) -> CrawlScope:
        """Return the immutable start-derived crawl scope."""
        return self._scope

    @property
    def summary(self) -> HttpLoadSummary | None:
        """Return crawl diagnostics after load completes."""
        return self._summary

    def load(self) -> tuple[SourceDocument, ...]:
        """Reuse a valid complete cache or run one bounded deterministic crawl."""
        recover_bounded_http_refresh(self._cache_root)
        cache_present = self._cache_root.exists() or _is_link_like(self._cache_root)
        cached = self._load_complete_cache()
        if cached is not None and not self._refresh:
            documents, summary = cached
            self._summary = replace_summary(summary, cache_reused=True, offline=self._offline)
            self._record_cache_replay()
            return documents
        if cache_present and not self._refresh:
            raise RuntimeError(
                "existing fetch cache is invalid or does not match the source config; "
                "use explicit --refresh in an online run"
            )
        if self._offline:
            raise RuntimeError("offline replay requires a complete valid fetch cache")
        documents, summary = self._crawl()
        self._summary = summary
        return documents

    def _crawl(self) -> tuple[tuple[SourceDocument, ...], HttpLoadSummary]:
        staging = self._cache_root.parent / f".{self._cache_root.name}.staging"
        if _is_link_like(staging) or (staging.exists() and not staging.is_dir()):
            raise RuntimeError(f"partial crawl staging is unsafe: {staging}")
        pages = staging / "pages"
        if _is_link_like(pages) or (pages.exists() and not pages.is_dir()):
            raise RuntimeError(f"partial crawl pages directory is unsafe: {pages}")
        if staging.exists():
            if not self._resume:
                raise RuntimeError(
                    f"partial crawl staging exists; use --resume or inspect it: {staging}"
                )
        pages.mkdir(parents=True, exist_ok=True)
        queue = deque(self._scope.start_urls)
        queued = set(queue)
        attempted: set[str] = set()
        canonical_seen: set[str] = set()
        content_seen: set[str] = set()
        documents: list[SourceDocument] = []
        page_records: list[dict[str, Any]] = []
        failures: list[FetchFailure] = []
        duplicate_canonical = 0
        duplicate_content = 0
        excluded_external = 0
        if self._resume and (staging / "fetch_manifest.json").is_file():
            (
                documents,
                page_records,
                failures,
                duplicate_canonical,
                duplicate_content,
                excluded_external,
            ) = self._load_partial_cache(staging)
            attempted.update(document.source_url for document in documents)
            canonical_seen.update(document.canonical_url for document in documents)
            content_seen.update(document.content_sha256 for document in documents)
            for document in documents:
                soup = BeautifulSoup(document.content, "lxml")
                for link in soup.find_all("a", href=True):
                    resolved = self._scope.resolve(
                        document.source_url,
                        str(link.get("href", "")),
                    )
                    if resolved is not None and resolved not in queued:
                        queue.append(resolved)
                        queued.add(resolved)
        maximum_attempts = self._settings.max_pages * 10
        while (
            queue
            and len(documents) < self._settings.max_pages
            and len(attempted) < maximum_attempts
        ):
            url = queue.popleft()
            if url in attempted:
                continue
            attempted.add(url)
            if not self._robots_allowed(url):
                failures.append(FetchFailure(url, "robots.txt denied or unavailable"))
                continue
            try:
                response = self._fetch_with_retries(url)
                final_url = normalize_http_url(
                    response.final_url,
                    retain_query=self._settings.retain_query,
                )
                if not self._scope.contains(final_url):
                    raise ValueError("redirect escaped configured crawl boundaries")
                if response.status < 200 or response.status >= 300:
                    raise ValueError(f"unexpected HTTP status: {response.status}")
                if not _is_html_content_type(response.content_type):
                    raise ValueError(f"non-HTML Content-Type: {response.content_type}")
                content = _decode_html(response.body, response.content_type)
            except Exception as error:  # noqa: BLE001 - continue one page failure
                failures.append(FetchFailure(url, _safe_error(error)))
                continue
            soup = BeautifulSoup(content, "lxml")
            canonical_url = _canonical_url(soup, final_url, self._scope)
            if canonical_url in canonical_seen:
                duplicate_canonical += 1
                continue
            canonical_seen.add(canonical_url)
            content_sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
            if content_sha in content_seen:
                duplicate_content += 1
            content_seen.add(content_sha)
            fetched_at = self._now().isoformat()
            logical_path = _logical_path(final_url)
            document = SourceDocument(
                source_url=final_url,
                canonical_url=canonical_url,
                content=content,
                content_sha256=content_sha,
                source_kind="bounded-http",
                logical_path=logical_path,
                category=self._category,
                fetched_at=fetched_at,
                metadata={
                    "http_status": response.status,
                    "content_type": response.content_type,
                    "response_bytes": len(response.body),
                },
            )
            documents.append(document)
            cache_relative = f"pages/{hashlib.sha256(final_url.encode()).hexdigest()}.html"
            cache_path = staging / cache_relative
            _write_text_atomic(content, cache_path)
            page_records.append(
                {
                    "source_url": final_url,
                    "canonical_url": canonical_url,
                    "logical_path": logical_path,
                    "category": self._category,
                    "fetched_at": fetched_at,
                    "content_sha256": content_sha,
                    "cache_path": cache_relative,
                    "http_status": response.status,
                    "content_type": response.content_type,
                    "response_bytes": len(response.body),
                }
            )
            discovered: list[str] = []
            for link in soup.find_all("a", href=True):
                raw = str(link.get("href", ""))
                resolved = self._scope.resolve(final_url, raw)
                if resolved is None:
                    if _looks_external_or_unsupported(final_url, raw, self._scope):
                        excluded_external += 1
                    continue
                if resolved not in attempted and resolved not in queued:
                    discovered.append(resolved)
            for discovered_url in sorted(set(discovered)):
                queue.append(discovered_url)
                queued.add(discovered_url)
            _write_fetch_manifest(
                staging / "fetch_manifest.json",
                self._manifest_payload(
                    page_records,
                    failures,
                    complete=False,
                    duplicate_canonical=duplicate_canonical,
                    duplicate_content=duplicate_content,
                    excluded_external=excluded_external,
                ),
            )
        summary = HttpLoadSummary(
            fetched_page_count=len(documents),
            failure_count=len(failures),
            duplicate_canonical_count=duplicate_canonical,
            duplicate_content_count=duplicate_content,
            excluded_external_url_count=excluded_external,
            cache_reused=False,
            offline=False,
            failures=tuple(failures),
        )
        _write_fetch_manifest(
            staging / "fetch_manifest.json",
            self._manifest_payload(
                page_records,
                failures,
                complete=True,
                duplicate_canonical=duplicate_canonical,
                duplicate_content=duplicate_content,
                excluded_external=excluded_external,
            ),
        )
        self._cache_backup = _publish_cache(
            staging,
            self._cache_root,
            refresh=self._refresh,
        )
        return tuple(documents), summary

    def commit_source(self) -> None:
        """Discard a retained refresh backup after downstream publication."""
        if self._cache_backup is not None:
            _retire_cache_path(self._cache_backup)
            self._cache_backup = None

    def rollback_source(self) -> None:
        """Restore the previous crawl cache after a downstream refresh failure."""
        if self._cache_backup is None:
            return
        recover_bounded_http_refresh(self._cache_root, force_rollback=True)
        self._cache_backup = None

    def _load_partial_cache(
        self,
        staging: Path,
    ) -> tuple[
        list[SourceDocument],
        list[dict[str, Any]],
        list[FetchFailure],
        int,
        int,
        int,
    ]:
        """Recover validated pages from an interrupted staging crawl."""
        manifest_path = staging / "fetch_manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                not isinstance(manifest, dict)
                or not self._manifest_identity_matches(manifest)
                or manifest.get("complete") is not False
            ):
                raise ValueError("partial fetch manifest settings do not match")
            page_records = list(manifest["pages"])
            documents = _documents_from_page_records(staging, page_records)
            failures = [
                FetchFailure(item["url"], item["reason"])
                for item in manifest.get("failures", ())
            ]
            return (
                documents,
                page_records,
                failures,
                int(manifest.get("duplicate_canonical_count", 0)),
                int(manifest.get("duplicate_content_count", 0)),
                int(manifest.get("excluded_external_url_count", 0)),
            )
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"partial crawl staging is invalid and was not overwritten: {error}"
            ) from error

    def _fetch_with_retries(self, url: str) -> HttpResponse:
        last_error: Exception | None = None
        for attempt in range(self._settings.max_retries + 1):
            if attempt or self._settings.request_delay_seconds:
                self._sleep(self._settings.request_delay_seconds)
            try:
                return self._transport.get(
                    url,
                    timeout_seconds=self._settings.timeout_seconds,
                    user_agent=self._settings.user_agent,
                    max_response_bytes=self._settings.max_response_bytes,
                )
            except urllib.error.HTTPError as error:
                if error.code in {404, 410}:
                    raise
                last_error = error
            except Exception as error:  # noqa: BLE001 - bounded retry boundary
                last_error = error
        raise RuntimeError(f"fetch failed after bounded retries: {last_error}") from last_error

    def _robots_allowed(self, url: str) -> bool:
        if not self._settings.respect_robots_txt:
            return True
        parsed = urlsplit(url)
        key = (parsed.scheme, parsed.netloc)
        if key not in self._robots:
            robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
            try:
                response = self._fetch_with_retries(robots_url)
                if response.status < 200 or response.status >= 300:
                    raise ValueError(f"unexpected robots HTTP status: {response.status}")
                final = urlsplit(response.final_url)
                if (final.scheme, final.netloc.lower()) != key:
                    raise ValueError("robots redirect escaped origin")
                text = response.body.decode("utf-8")
                parser = urllib.robotparser.RobotFileParser()
                parser.set_url(robots_url)
                parser.parse(text.splitlines())
                self._robots[key] = parser
            except urllib.error.HTTPError as error:
                if error.code in {404, 410}:
                    parser = urllib.robotparser.RobotFileParser()
                    parser.set_url(robots_url)
                    parser.parse([])
                    self._robots[key] = parser
                else:
                    self._robots[key] = None
            except Exception:  # fail closed; never mass-fetch without robots policy
                self._robots[key] = None
        parser = self._robots[key]
        return parser is not None and parser.can_fetch(self._settings.user_agent, url)

    def _manifest_payload(
        self,
        pages: list[dict[str, Any]],
        failures: list[FetchFailure],
        *,
        complete: bool,
        duplicate_canonical: int,
        duplicate_content: int,
        excluded_external: int,
    ) -> dict[str, Any]:
        return {
            "schema_revision": FETCH_MANIFEST_REVISION,
            "complete": complete,
            "loader_type": "bounded-http",
            "cache_reused": False,
            "offline": False,
            "source_config_fingerprint_revision": "source-config-v1",
            "source_config_sha256": self._source_config_sha256,
            "start_urls": list(self._scope.start_urls),
            "boundaries": [asdict(item) for item in self._scope.boundaries],
            "retain_query": self._settings.retain_query,
            "max_pages": self._settings.max_pages,
            "fetched_page_count": len(pages),
            "failure_count": len(failures),
            "duplicate_canonical_count": duplicate_canonical,
            "duplicate_content_count": duplicate_content,
            "excluded_external_url_count": excluded_external,
            "pages": pages,
            "failures": [asdict(item) for item in failures],
            "finished_at": self._now().isoformat() if complete else None,
        }

    def _load_complete_cache(
        self,
    ) -> tuple[tuple[SourceDocument, ...], HttpLoadSummary] | None:
        manifest_path = self._cache_root / "fetch_manifest.json"
        if not manifest_path.is_file():
            return None
        try:
            manifest = _read_bounded_json_object(manifest_path)
            if (
                not isinstance(manifest, dict)
                or not self._manifest_identity_matches(manifest)
                or manifest.get("complete") is not True
            ):
                return None
            documents = _documents_from_page_records(
                self._cache_root,
                list(manifest["pages"]),
            )
            failures = tuple(
                FetchFailure(item["url"], item["reason"])
                for item in manifest.get("failures", ())
            )
            summary = HttpLoadSummary(
                fetched_page_count=len(documents),
                failure_count=len(failures),
                duplicate_canonical_count=int(
                    manifest.get("duplicate_canonical_count", 0)
                ),
                duplicate_content_count=int(manifest.get("duplicate_content_count", 0)),
                excluded_external_url_count=int(
                    manifest.get("excluded_external_url_count", 0)
                ),
                cache_reused=True,
                offline=self._offline,
                failures=failures,
            )
            return tuple(documents), summary
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def _manifest_identity_matches(self, manifest: dict[str, Any]) -> bool:
        """Require a complete fingerprint rather than guessing a legacy identity."""
        revision = manifest.get("schema_revision")
        if revision == FETCH_MANIFEST_REVISION:
            return manifest.get("source_config_sha256") == self._source_config_sha256
        if revision == LEGACY_FETCH_MANIFEST_REVISION:
            return False
        return False

    def _record_cache_replay(self) -> None:
        """Record replay provenance without changing cached page identities."""
        path = self._cache_root / "fetch_manifest.json"
        manifest = _read_bounded_json_object(path)
        manifest["cache_reused"] = True
        manifest["offline"] = self._offline
        _write_fetch_manifest(path, manifest)


def normalize_http_url(value: str, *, retain_query: bool = False) -> str:
    """Normalize an HTTP(S) fetch identity and reject traversal/auth fragments."""
    parsed = urlsplit(value)
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("URL must use HTTP(S)")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("URL authentication information is not allowed")
    decoded_path = unquote(parsed.path or "/")
    if any(part == ".." for part in decoded_path.replace("\\", "/").split("/")):
        raise ValueError("URL path traversal is not allowed")
    normalized_path = posixpath.normpath(decoded_path)
    if not normalized_path.startswith("/"):
        normalized_path = f"/{normalized_path}"
    if decoded_path.endswith("/") and not normalized_path.endswith("/"):
        normalized_path += "/"
    host = parsed.hostname.casefold()
    port = parsed.port
    if port is not None and not (
        (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    ):
        netloc = f"{host}:{port}"
    else:
        netloc = host
    query = parsed.query if retain_query else ""
    return urlunsplit((scheme, netloc, quote(normalized_path, safe="/%:@-._~"), query, ""))


def replace_summary(
    summary: HttpLoadSummary,
    *,
    cache_reused: bool,
    offline: bool,
) -> HttpLoadSummary:
    """Return a summary with replay flags changed and metrics preserved."""
    return HttpLoadSummary(
        fetched_page_count=summary.fetched_page_count,
        failure_count=summary.failure_count,
        duplicate_canonical_count=summary.duplicate_canonical_count,
        duplicate_content_count=summary.duplicate_content_count,
        excluded_external_url_count=summary.excluded_external_url_count,
        cache_reused=cache_reused,
        offline=offline,
        failures=summary.failures,
    )


def _boundary_for(url: str) -> CrawlBoundary:
    parsed = urlsplit(url)
    path = parsed.path
    if not path.endswith("/"):
        path = f"{path.rsplit('/', maxsplit=1)[0]}/"
    return CrawlBoundary(parsed.scheme, parsed.netloc, path)


def _is_html_candidate(url: str) -> bool:
    suffix = PurePosixPath(unquote(urlsplit(url).path)).suffix.casefold()
    return suffix not in _NON_HTML_SUFFIXES


def _is_html_content_type(value: str) -> bool:
    media_type = value.split(";", maxsplit=1)[0].strip().casefold()
    return media_type in {"text/html", "application/xhtml+xml"}


def _decode_html(body: bytes, content_type: str) -> str:
    match = re.search(r"charset=([^;\s]+)", content_type, flags=re.IGNORECASE)
    if match:
        try:
            return body.decode(match.group(1).strip('"\''))
        except (LookupError, UnicodeDecodeError):
            pass
    decoded = UnicodeDammit(body, is_html=True).unicode_markup
    if decoded is None:
        raise UnicodeDecodeError("utf-8", body, 0, len(body), "could not decode HTML")
    return decoded


def _canonical_url(soup: BeautifulSoup, source_url: str, scope: CrawlScope) -> str:
    link = soup.select_one("link[rel~='canonical'][href]")
    if link is None:
        return source_url
    candidate = scope.resolve(source_url, str(link.get("href", "")))
    return candidate or source_url


def _logical_path(url: str) -> str:
    parsed = urlsplit(url)
    path = unquote(parsed.path).lstrip("/")
    if not path or path.endswith("/"):
        path = f"{path}index.html"
    return f"{parsed.netloc}/{path}"


def _looks_external_or_unsupported(base: str, raw: str, scope: CrawlScope) -> bool:
    lowered = raw.strip().casefold()
    if lowered.startswith(("mailto:", "tel:", "javascript:", "data:")):
        return True
    try:
        normalized = normalize_http_url(urljoin(base, raw))
    except ValueError:
        return False
    return not scope.contains(normalized)


def _safe_cache_relative(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path == Path("."):
        raise ValueError("unsafe cache path")
    return path


def _documents_from_page_records(
    cache_root: Path,
    records: list[dict[str, Any]],
) -> list[SourceDocument]:
    documents: list[SourceDocument] = []
    resolved_root = cache_root.resolve()
    for record in records:
        relative = _safe_cache_relative(record["cache_path"])
        path = (cache_root / relative).resolve()
        if not path.is_relative_to(resolved_root):
            raise ValueError("cache path escaped cache root")
        content = _read_cached_html(path)
        if hashlib.sha256(content.encode()).hexdigest() != record["content_sha256"]:
            raise ValueError("cached page SHA-256 mismatch")
        documents.append(
            SourceDocument(
                source_url=record["source_url"],
                canonical_url=record["canonical_url"],
                content=content,
                content_sha256=record["content_sha256"],
                source_kind="bounded-http",
                logical_path=record["logical_path"],
                category=record["category"],
                fetched_at=record["fetched_at"],
                metadata={
                    "http_status": record["http_status"],
                    "content_type": record["content_type"],
                    "response_bytes": record["response_bytes"],
                },
            )
        )
    return documents


def _read_cached_html(path: Path) -> str:
    """Read a bounded no-follow regular UTF-8 cache page."""
    return _read_bounded_regular_bytes(
        path,
        max_bytes=_MAX_CACHED_HTML_BYTES,
        label="cached page",
    ).decode("utf-8")


def _read_bounded_json_object(path: Path) -> dict[str, Any]:
    encoded = _read_bounded_regular_bytes(
        path,
        max_bytes=_MAX_FETCH_MANIFEST_BYTES,
        label="fetch manifest",
    )
    value = json.loads(encoded.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("fetch manifest must be a JSON object")
    return value


def _read_bounded_regular_bytes(
    path: Path,
    *,
    max_bytes: int,
    label: str,
) -> bytes:
    """Read a stable, bounded regular file without following a final symlink."""
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > max_bytes:
        raise ValueError(f"{label} must be a bounded regular file")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as stream:
        opened = os.fstat(stream.fileno())
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError(f"{label} must be a regular file")
        payload = stream.read(max_bytes + 1)
        after = os.fstat(stream.fileno())
    if (
        len(payload) > max_bytes
        or opened.st_dev != before.st_dev
        or opened.st_ino != before.st_ino
        or after.st_size != len(payload)
        or after.st_mtime_ns != opened.st_mtime_ns
        or after.st_ctime_ns != opened.st_ctime_ns
    ):
        raise ValueError(f"{label} changed while it was read")
    return payload


def _write_text_atomic(value: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise


def _write_fetch_manifest(path: Path, payload: dict[str, Any]) -> None:
    _write_text_atomic(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        path,
    )


def validate_bounded_http_cache(
    cache_root: Path,
    *,
    expected_source_config_sha256: str,
    expected_processing_config_sha256: str,
    expected_source_snapshot_sha256: str,
    expected_page_count: int,
) -> tuple[SourceDocument, ...]:
    """Validate a committed v2 crawl manifest, page bytes, and snapshot identity."""
    manifest_path = cache_root / "fetch_manifest.json"
    try:
        source = _read_bounded_json_object(manifest_path)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("bounded source manifest is invalid") from error
    if not isinstance(source, dict) or set(source) != _COMMITTED_FETCH_MANIFEST_KEYS:
        raise RuntimeError("bounded source manifest schema is invalid")
    expected_values = {
        "schema_revision": FETCH_MANIFEST_REVISION,
        "complete": True,
        "loader_type": "bounded-http",
        "source_config_fingerprint_revision": "source-config-v1",
        "source_config_sha256": expected_source_config_sha256,
        "processing_config_fingerprint_revision": "processing-config-v1",
        "processing_config_sha256": expected_processing_config_sha256,
        "source_snapshot_sha256": expected_source_snapshot_sha256,
        "source_page_count": expected_page_count,
        "fetched_page_count": expected_page_count,
    }
    mismatched = [
        key for key, value in expected_values.items() if source.get(key) != value
    ]
    if mismatched:
        raise RuntimeError(
            "bounded source manifest identity is inconsistent: "
            f"{', '.join(mismatched)}"
        )
    raw_records = source.get("pages")
    if not isinstance(raw_records, list) or len(raw_records) != expected_page_count:
        raise RuntimeError("bounded source manifest page count is inconsistent")
    if any(not isinstance(record, dict) or set(record) != _PAGE_RECORD_KEYS for record in raw_records):
        raise RuntimeError("bounded source manifest page record schema is invalid")
    failures = source.get("failures")
    if (
        not isinstance(failures, list)
        or any(
            not isinstance(item, dict) or set(item) != {"url", "reason"}
            for item in failures
        )
        or source.get("failure_count") != len(failures)
    ):
        raise RuntimeError("bounded source manifest failures are inconsistent")
    try:
        documents = tuple(_documents_from_page_records(cache_root, raw_records))
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise RuntimeError("bounded source cache pages are invalid") from error
    if source_snapshot_sha256(documents) != expected_source_snapshot_sha256:
        raise RuntimeError("bounded source snapshot identity is inconsistent")
    return documents


def validate_bounded_http_replayable_cache(
    cache_root: Path,
    *,
    expected_source_config_sha256: str,
) -> tuple[SourceDocument, ...]:
    """Validate raw crawl identity without consulting a derived dataset manifest."""
    try:
        source = _read_bounded_json_object(cache_root / "fetch_manifest.json")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError("bounded source manifest is invalid") from error
    page_count = source.get("source_page_count")
    processing_config = source.get("processing_config_sha256")
    source_snapshot = source.get("source_snapshot_sha256")
    if (
        set(source) != _COMMITTED_FETCH_MANIFEST_KEYS
        or source.get("source_config_sha256") != expected_source_config_sha256
        or type(page_count) is not int
        or page_count < 0
        or not isinstance(processing_config, str)
        or not isinstance(source_snapshot, str)
    ):
        raise RuntimeError("bounded source manifest replay identity is invalid")
    return validate_bounded_http_cache(
        cache_root,
        expected_source_config_sha256=expected_source_config_sha256,
        expected_processing_config_sha256=processing_config,
        expected_source_snapshot_sha256=source_snapshot,
        expected_page_count=page_count,
    )


def recover_bounded_http_refresh(
    cache_root: Path,
    *,
    force_rollback: bool = False,
) -> bool:
    """Recover a refresh cache swap by consulting the committed dataset identity."""
    cache = cache_root.expanduser()
    backup = cache.parent / f".{cache.name}.backup"
    failed = cache.parent / f".{cache.name}.failed-refresh"
    staging = cache.parent / f".{cache.name}.staging"
    for path in (cache, backup, failed, staging):
        if _is_link_like(path):
            raise RuntimeError(
                f"bounded HTTP cache transaction path is a symlink/junction: {path}"
            )
        if path.exists() and not path.is_dir():
            raise RuntimeError(f"bounded HTTP cache transaction path is not a directory: {path}")
    if not backup.exists():
        if failed.exists():
            if not cache.exists():
                raise RuntimeError("bounded HTTP cache recovery lost its committed cache")
            _remove_cache_path(failed)
        return True
    if not force_rollback and bounded_http_refresh_candidate_is_committed(cache):
        _retire_cache_path(backup)
        return True
    if failed.exists():
        if cache.exists():
            raise RuntimeError(f"failed refresh quarantine already exists: {failed}")
    elif cache.exists():
        os.rename(cache, failed)
    try:
        os.rename(backup, cache)
    except BaseException:
        if failed.exists() and not cache.exists():
            os.rename(failed, cache)
        raise
    _remove_cache_path(failed)
    if staging.exists():
        _retire_cache_path(staging)
    return False


def bounded_http_refresh_candidate_is_committed(cache_root: Path) -> bool:
    """Return whether a swapped crawl cache matches the strict committed dataset."""
    dataset_path = cache_root.parent.parent / "dataset_manifest.json"
    try:
        dataset = load_dataset_manifest(dataset_path)
    except (OSError, ValueError):
        return False
    try:
        if (
            dataset.schema_revision != DATASET_MANIFEST_REVISION
            or dataset.loader_type != "bounded-http"
            or dataset.source_config_sha256 is None
            or dataset.processing_config_sha256 is None
            or dataset.source_snapshot_sha256 is None
        ):
            return False
        validate_bounded_http_cache(
            cache_root,
            expected_source_config_sha256=dataset.source_config_sha256,
            expected_processing_config_sha256=dataset.processing_config_sha256,
            expected_source_snapshot_sha256=dataset.source_snapshot_sha256,
            expected_page_count=dataset.source_page_count,
        )
        return True
    except RuntimeError:
        return False


def _remove_cache_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif _is_link_like(path):
        raise RuntimeError(f"refusing to remove a bounded HTTP cache junction: {path}")
    elif path.exists():
        shutil.rmtree(path)


def _retire_cache_path(path: Path) -> None:
    """Atomically deactivate a committed backup before best-effort cleanup."""
    if not path.exists() and not _is_link_like(path):
        return
    if _is_link_like(path) or not path.is_dir():
        raise RuntimeError(f"bounded HTTP cache backup is unsafe: {path}")
    retired = path.with_name(f"{path.name}.retired-{uuid.uuid4().hex}")
    os.rename(path, retired)
    try:
        shutil.rmtree(retired)
    except BaseException:
        # The transaction committed at rename; an orphan is safer than rollback.
        pass


def _publish_cache(
    staging: Path,
    destination: Path,
    *,
    refresh: bool,
) -> Path | None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        os.rename(staging, destination)
        return None
    if not refresh:
        raise RuntimeError("cache appeared during crawl; refusing implicit replacement")
    backup = destination.parent / f".{destination.name}.backup"
    if backup.exists():
        raise RuntimeError(f"cache backup already exists: {backup}")
    os.rename(destination, backup)
    try:
        os.rename(staging, destination)
    except BaseException:
        os.rename(backup, destination)
        raise
    return backup


def _is_link_like(path: Path) -> bool:
    """Return whether a path is a symlink or native Windows junction."""
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(callable(is_junction) and is_junction())


def _safe_error(error: BaseException) -> str:
    message = str(error).strip() or error.__class__.__name__
    return f"{error.__class__.__name__}: {message}"


def __getattr__(name: str) -> object:
    """Resolve historical loader exports without restoring static site coupling."""
    targets = {
        "DocumentLoader": ("python_doc_rag.ingestion.protocols", "DocumentLoader"),
        "LocalHtmlTreeLoader": (
            "python_doc_rag.sites.python_docs.local_compat",
            "LocalHtmlTreeLoader",
        ),
    }
    try:
        module_name, attribute = targets[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(importlib.import_module(module_name), attribute)
    globals()[name] = value
    return value
