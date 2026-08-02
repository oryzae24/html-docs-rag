import hashlib
import json
import os
import shutil
import urllib.error
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

import python_doc_rag.loading as loading
from python_doc_rag.loading import (
    BoundedHttpHtmlLoader,
    CrawlScope,
    HttpResponse,
    normalize_http_url,
)
from python_doc_rag.models import SourceDocument
from python_doc_rag.parsing import GenericHtmlParser
from python_doc_rag.site_config import (
    GenericHtmlParserSettings,
    HttpLoaderSettings,
)
from python_doc_rag.source_identity import (
    source_config_sha256,
    source_snapshot_sha256,
)


def _settings(
    *,
    start_urls: tuple[str, ...] = ("https://example.com/docs/",),
    max_pages: int = 10,
    max_retries: int = 1,
) -> HttpLoaderSettings:
    return HttpLoaderSettings(
        type="bounded-http",
        start_urls=start_urls,
        max_pages=max_pages,
        timeout_seconds=2,
        request_delay_seconds=0,
        max_response_bytes=10_000,
        user_agent="fixture/1.0",
        respect_robots_txt=True,
        retain_query=False,
        max_retries=max_retries,
    )


def _response(
    url: str,
    body: str,
    *,
    final_url: str | None = None,
    content_type: str = "text/html; charset=utf-8",
) -> HttpResponse:
    return HttpResponse(
        requested_url=url,
        final_url=final_url or url,
        status=200,
        content_type=content_type,
        body=body.encode(),
    )


class FakeTransport:
    def __init__(self, responses: dict[str, HttpResponse | Exception]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    def get(self, url: str, **kwargs: Any) -> HttpResponse:
        del kwargs
        self.calls.append(url)
        value = self.responses.get(url, RuntimeError(f"unexpected URL: {url}"))
        if isinstance(value, Exception):
            raise value
        return value


def _robots(origin: str, *, allow: bool = True) -> tuple[str, HttpResponse]:
    url = f"{origin}/robots.txt"
    body = "User-agent: *\nAllow: /\n" if allow else "User-agent: *\nDisallow: /\n"
    return url, _response(url, body, content_type="text/plain; charset=utf-8")


def _document(html: str) -> SourceDocument:
    return SourceDocument(
        source_url="https://example.com/docs/page/",
        canonical_url="https://example.com/docs/page/",
        content=html,
        content_sha256=hashlib.sha256(html.encode()).hexdigest(),
        source_kind="bounded-http",
        logical_path="example.com/docs/page/index.html",
        category="example-docs",
        fetched_at="2026-08-01T00:00:00+00:00",
    )


def _parser_settings(**changes: Any) -> GenericHtmlParserSettings:
    values: dict[str, Any] = {
        "type": "generic-html",
        "content_selectors": ("article", "main"),
        "exclude_selectors": ("nav", "footer", ".remove"),
        "title_selectors": ("h1", "title"),
        "heading_levels": (1, 2, 3),
        "minimum_section_text_length": 5,
        "fallback_to_body": False,
        "include_lead_text": True,
    }
    values.update(changes)
    return GenericHtmlParserSettings(**values)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("HTTPS://EXAMPLE.COM:443/docs/#part", "https://example.com/docs/"),
        ("http://EXAMPLE.COM:80/docs", "http://example.com/docs"),
        ("https://example.com/a/./b/", "https://example.com/a/b/"),
        ("https://example.com/docs/?x=1", "https://example.com/docs/"),
    ],
)
def test_url_normalization_default_port_fragment_and_query(
    value: str,
    expected: str,
) -> None:
    assert normalize_http_url(value) == expected


@pytest.mark.parametrize(
    "value",
    (
        "ftp://example.com/docs/",
        "https://example.com/docs/%2e%2e/private/",
        "https://user:pass@example.com/docs/",
    ),
)
def test_url_normalization_rejects_invalid_scheme_traversal_and_auth(
    value: str,
) -> None:
    with pytest.raises(ValueError):
        normalize_http_url(value)


def test_crawl_scope_supports_multiple_origins_and_prefix_union() -> None:
    scope = CrawlScope(("https://example.com/docs/", "https://other.example/guide/"))

    assert scope.resolve("https://example.com/docs/", "page/") == (
        "https://example.com/docs/page/"
    )
    assert scope.resolve("https://example.com/docs/", "//other.example/guide/x/") == (
        "https://other.example/guide/x/"
    )
    assert scope.resolve("https://example.com/docs/", "../private/") is None
    assert scope.resolve("https://example.com/docs/", "https://external.test/") is None
    assert scope.resolve("https://example.com/docs/", "mailto:x@example.com") is None
    assert scope.resolve("https://example.com/docs/", "javascript:alert(1)") is None
    assert scope.resolve("https://example.com/docs/", "data:text/plain,x") is None
    assert scope.resolve("https://example.com/docs/", "image.png") is None


def test_http_loader_uses_deterministic_bfs_and_never_fetches_external(
    tmp_path: Path,
) -> None:
    start = "https://example.com/docs/"
    robots_url, robots = _robots("https://example.com")
    transport = FakeTransport(
        {
            robots_url: robots,
            start: _response(
                start,
                "<html><head><title>Start</title></head><body>"
                "<a href='b/'>B</a><a href='a/'>A</a>"
                "<a href='https://external.test/x/'>X</a></body></html>",
            ),
            f"{start}a/": _response(
                f"{start}a/", "<html><main><h1>A</h1><p>A page</p></main></html>"
            ),
            f"{start}b/": _response(
                f"{start}b/", "<html><main><h1>B</h1><p>B page</p></main></html>"
            ),
        }
    )
    loader = BoundedHttpHtmlLoader(
        _settings(),
        tmp_path / "raw",
        category="fixture",
        transport=transport,
        sleep=lambda value: None,
        now=lambda: datetime(2026, 8, 1, tzinfo=UTC),
    )

    documents = loader.load()

    assert [item.source_url for item in documents] == [
        start,
        f"{start}a/",
        f"{start}b/",
    ]
    assert transport.calls == [robots_url, start, f"{start}a/", f"{start}b/"]
    assert "https://external.test/x/" not in transport.calls
    assert loader.summary is not None
    assert loader.summary.excluded_external_url_count == 1


def test_http_loader_resume_reuses_valid_staged_page(
    tmp_path: Path,
) -> None:
    start = "https://example.com/docs/"
    next_url = f"{start}next/"
    staging = tmp_path / ".raw.staging"
    pages = staging / "pages"
    pages.mkdir(parents=True)
    start_html = "<html><main><a href='next/'>next</a></main></html>"
    cache_name = f"pages/{hashlib.sha256(start.encode()).hexdigest()}.html"
    (staging / cache_name).write_text(start_html, encoding="utf-8")
    record = {
        "source_url": start,
        "canonical_url": start,
        "logical_path": "example.com/docs/index.html",
        "category": "fixture",
        "fetched_at": "2026-08-01T00:00:00+00:00",
        "content_sha256": hashlib.sha256(start_html.encode()).hexdigest(),
        "cache_path": cache_name,
        "http_status": 200,
        "content_type": "text/html; charset=utf-8",
        "response_bytes": len(start_html.encode()),
    }
    (staging / "fetch_manifest.json").write_text(
        json.dumps(
            {
                "schema_revision": "bounded-http-fetch-v2",
                "complete": False,
                "source_config_sha256": source_config_sha256(
                    _settings(),
                    category="fixture",
                ),
                "start_urls": [start],
                "pages": [record],
                "failures": [],
            }
        ),
        encoding="utf-8",
    )
    robots_url, robots = _robots("https://example.com")
    transport = FakeTransport(
        {
            robots_url: robots,
            next_url: _response(
                next_url,
                "<html><main><h1>Next</h1><p>next page</p></main></html>",
            ),
        }
    )
    loader = BoundedHttpHtmlLoader(
        _settings(),
        tmp_path / "raw",
        category="fixture",
        transport=transport,
        sleep=lambda value: None,
        resume=True,
    )

    documents = loader.load()

    assert [item.source_url for item in documents] == [start, next_url]
    assert start not in transport.calls


def test_http_loader_resume_rejects_symlinked_staging_pages(
    tmp_path: Path,
) -> None:
    staging = tmp_path / ".raw.staging"
    outside = tmp_path / "outside"
    staging.mkdir()
    outside.mkdir()
    (staging / "pages").symlink_to(outside, target_is_directory=True)
    transport = FakeTransport({})

    with pytest.raises(RuntimeError, match="pages directory is unsafe"):
        BoundedHttpHtmlLoader(
            _settings(),
            tmp_path / "raw",
            category="fixture",
            transport=transport,
            resume=True,
        ).load()

    assert transport.calls == []
    assert list(outside.iterdir()) == []


def test_http_loader_rejects_native_junction_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = tmp_path / ".raw.staging"
    staging.mkdir()
    monkeypatch.setattr(
        Path,
        "is_junction",
        lambda path: path == staging,
        raising=False,
    )
    transport = FakeTransport({})

    with pytest.raises(RuntimeError, match="symlink/junction"):
        BoundedHttpHtmlLoader(
            _settings(),
            tmp_path / "raw",
            category="fixture",
            transport=transport,
            resume=True,
        ).load()

    assert transport.calls == []


def test_http_loader_respects_robots_deny_and_fail_closed(
    tmp_path: Path,
) -> None:
    robots_url, denied = _robots("https://example.com", allow=False)
    transport = FakeTransport({robots_url: denied})
    loader = BoundedHttpHtmlLoader(
        _settings(),
        tmp_path / "denied",
        category="fixture",
        transport=transport,
        sleep=lambda value: None,
    )

    assert loader.load() == ()
    assert transport.calls == [robots_url]
    assert loader.summary is not None
    assert loader.summary.failure_count == 1

    failed_transport = FakeTransport({robots_url: TimeoutError("robots timeout")})
    failed = BoundedHttpHtmlLoader(
        _settings(max_retries=0),
        tmp_path / "failed",
        category="fixture",
        transport=failed_transport,
        sleep=lambda value: None,
    )
    assert failed.load() == ()
    assert failed_transport.calls == [robots_url]


def test_http_loader_treats_missing_robots_as_empty_but_other_failures_closed(
    tmp_path: Path,
) -> None:
    start = "https://example.com/docs/"
    robots_url = "https://example.com/robots.txt"
    missing = urllib.error.HTTPError(robots_url, 404, "missing", {}, None)
    transport = FakeTransport(
        {
            robots_url: missing,
            start: _response(start, "<html><main><p>allowed</p></main></html>"),
        }
    )
    loader = BoundedHttpHtmlLoader(
        _settings(max_retries=0),
        tmp_path / "missing",
        category="fixture",
        transport=transport,
        sleep=lambda value: None,
    )
    assert len(loader.load()) == 1

    forbidden = urllib.error.HTTPError(robots_url, 403, "forbidden", {}, None)
    denied_transport = FakeTransport({robots_url: forbidden})
    denied = BoundedHttpHtmlLoader(
        _settings(max_retries=0),
        tmp_path / "forbidden",
        category="fixture",
        transport=denied_transport,
        sleep=lambda value: None,
    )
    assert denied.load() == ()


def test_http_loader_bounded_retry_and_page_failure_continuation(
    tmp_path: Path,
) -> None:
    start = "https://example.com/docs/"
    robots_url, robots = _robots("https://example.com")
    transport = FakeTransport({robots_url: robots, start: TimeoutError("timeout")})
    loader = BoundedHttpHtmlLoader(
        _settings(max_retries=2),
        tmp_path / "raw",
        category="fixture",
        transport=transport,
        sleep=lambda value: None,
    )

    assert loader.load() == ()
    assert Counter(transport.calls)[start] == 3
    assert loader.summary is not None
    assert loader.summary.failure_count == 1


@pytest.mark.parametrize(
    ("content_type", "reason"),
    (("application/pdf", "non-HTML"), ("application/json", "non-HTML")),
)
def test_http_loader_skips_non_html_content(
    tmp_path: Path,
    content_type: str,
    reason: str,
) -> None:
    start = "https://example.com/docs/"
    robots_url, robots = _robots("https://example.com")
    loader = BoundedHttpHtmlLoader(
        _settings(max_retries=0),
        tmp_path / content_type.replace("/", "-"),
        category="fixture",
        transport=FakeTransport(
            {
                robots_url: robots,
                start: _response(start, "x", content_type=content_type),
            }
        ),
        sleep=lambda value: None,
    )

    assert loader.load() == ()
    assert reason in loader.summary.failures[0].reason  # type: ignore[union-attr]


def test_http_loader_rejects_external_redirect(tmp_path: Path) -> None:
    start = "https://example.com/docs/"
    robots_url, robots = _robots("https://example.com")
    loader = BoundedHttpHtmlLoader(
        _settings(max_retries=0),
        tmp_path / "raw",
        category="fixture",
        transport=FakeTransport(
            {
                robots_url: robots,
                start: _response(
                    start, "<html></html>", final_url="https://evil.test/"
                ),
            }
        ),
        sleep=lambda value: None,
    )

    assert loader.load() == ()
    assert "redirect escaped" in loader.summary.failures[0].reason  # type: ignore[union-attr]


def test_http_loader_deduplicates_canonical_and_records_content_duplicates(
    tmp_path: Path,
) -> None:
    start = "https://example.com/docs/"
    a = f"{start}a/"
    b = f"{start}b/"
    robots_url, robots = _robots("https://example.com")
    common = (
        "<html><head><link rel='canonical' href='../a/'></head><body>same</body></html>"
    )
    loader = BoundedHttpHtmlLoader(
        _settings(),
        tmp_path / "raw",
        category="fixture",
        transport=FakeTransport(
            {
                robots_url: robots,
                start: _response(start, "<a href='a/'>a</a><a href='b/'>b</a>"),
                a: _response(a, common),
                b: _response(b, common),
            }
        ),
        sleep=lambda value: None,
    )

    documents = loader.load()

    assert len(documents) == 2
    assert loader.summary is not None
    assert loader.summary.duplicate_canonical_count == 1


def test_http_loader_reuses_cache_and_supports_offline_replay(tmp_path: Path) -> None:
    start = "https://example.com/docs/"
    robots_url, robots = _robots("https://example.com")
    cache = tmp_path / "raw"
    first_transport = FakeTransport(
        {
            robots_url: robots,
            start: _response(start, "<html><body>cached</body></html>"),
        }
    )
    first = BoundedHttpHtmlLoader(
        _settings(),
        cache,
        category="fixture",
        transport=first_transport,
        sleep=lambda value: None,
    ).load()
    second_transport = FakeTransport({})
    replay_loader = BoundedHttpHtmlLoader(
        _settings(),
        cache,
        category="fixture",
        offline=True,
        transport=second_transport,
    )

    replay = replay_loader.load()

    assert [item.to_dict() for item in replay] == [item.to_dict() for item in first]
    assert second_transport.calls == []
    assert replay_loader.summary is not None
    assert replay_loader.summary.cache_reused
    assert replay_loader.summary.offline
    manifest = json.loads((cache / "fetch_manifest.json").read_text())
    assert manifest["complete"] is True


def test_http_loader_source_mismatch_refuses_before_network(tmp_path: Path) -> None:
    start = "https://example.com/docs/"
    cache = tmp_path / "raw"
    first_transport = FakeTransport(
        {start: _response(start, "<html><body>cached</body></html>")}
    )
    BoundedHttpHtmlLoader(
        _settings(max_pages=1),
        cache,
        category="fixture",
        transport=first_transport,
        sleep=lambda value: None,
    ).load()
    second_transport = FakeTransport({})

    with pytest.raises(RuntimeError, match="does not match.*--refresh"):
        BoundedHttpHtmlLoader(
            _settings(max_pages=2),
            cache,
            category="fixture",
            transport=second_transport,
            sleep=lambda value: None,
        ).load()

    assert second_transport.calls == []
    assert not (tmp_path / ".raw.staging").exists()


def test_http_loader_legacy_cache_is_not_assumed_to_match_offline(
    tmp_path: Path,
) -> None:
    start = "https://example.com/docs/"
    cache = tmp_path / "raw"
    BoundedHttpHtmlLoader(
        _settings(max_pages=1),
        cache,
        category="fixture",
        transport=FakeTransport(
            {start: _response(start, "<html><body>legacy</body></html>")}
        ),
        sleep=lambda value: None,
    ).load()
    manifest_path = cache / "fetch_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_revision"] = "bounded-http-fetch-v1"
    manifest.pop("source_config_sha256")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    transport = FakeTransport({})

    with pytest.raises(RuntimeError, match="--refresh"):
        BoundedHttpHtmlLoader(
            _settings(max_pages=2),
            cache,
            category="fixture",
            offline=True,
            transport=transport,
        ).load()

    assert transport.calls == []


def test_http_refresh_crash_restores_backup_when_candidate_cache_is_corrupt(
    tmp_path: Path,
) -> None:
    start = "https://example.com/docs/"
    robots_url, robots = _robots("https://example.com")
    data_root = tmp_path / "dataset"
    cache = data_root / "data/raw"
    old = BoundedHttpHtmlLoader(
        _settings(max_pages=1),
        cache,
        category="fixture",
        transport=FakeTransport(
            {
                robots_url: robots,
                start: _response(start, "<html><body>old cache</body></html>"),
            }
        ),
        sleep=lambda value: None,
    ).load()
    refreshed = BoundedHttpHtmlLoader(
        _settings(max_pages=1),
        cache,
        category="fixture",
        refresh=True,
        transport=FakeTransport(
            {
                robots_url: robots,
                start: _response(start, "<html><body>new cache</body></html>"),
            }
        ),
        sleep=lambda value: None,
    ).load()
    snapshot = source_snapshot_sha256(refreshed)
    source_path = cache / "fetch_manifest.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source.update(
        {
            "processing_config_sha256": "e" * 64,
            "source_snapshot_sha256": snapshot,
            "source_page_count": len(refreshed),
        }
    )
    source_path.write_text(json.dumps(source), encoding="utf-8")
    (data_root / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "schema_revision": "dataset-artifact-layout-v2",
                "source_config_sha256": source["source_config_sha256"],
                "processing_config_sha256": "e" * 64,
                "source_snapshot_sha256": snapshot,
                "source_page_count": len(refreshed),
            }
        ),
        encoding="utf-8",
    )
    candidate_page = cache / source["pages"][0]["cache_path"]
    candidate_page.write_text("corrupt", encoding="utf-8")
    no_network = FakeTransport({})

    recovered = BoundedHttpHtmlLoader(
        _settings(max_pages=1),
        cache,
        category="fixture",
        offline=True,
        transport=no_network,
    ).load()

    assert [document.content for document in recovered] == [old[0].content]
    assert no_network.calls == []
    assert not (data_root / "data/.raw.backup").exists()


def test_http_refresh_swap_recovery_retires_complete_orphan_staging(
    tmp_path: Path,
) -> None:
    start = "https://example.com/docs/"
    robots_url, robots = _robots("https://example.com")
    cache = tmp_path / "data/raw"
    BoundedHttpHtmlLoader(
        _settings(max_pages=1),
        cache,
        category="fixture",
        transport=FakeTransport(
            {
                robots_url: robots,
                start: _response(start, "<html><body>old cache</body></html>"),
            }
        ),
        sleep=lambda value: None,
    ).load()
    staging = tmp_path / "data/.raw.staging"
    backup = tmp_path / "data/.raw.backup"
    shutil.copytree(cache, staging)
    os.rename(cache, backup)

    recovered = loading.recover_bounded_http_refresh(cache)

    assert recovered is False
    assert cache.is_dir()
    assert not backup.exists()
    assert not staging.exists()
    refreshed = BoundedHttpHtmlLoader(
        _settings(max_pages=1),
        cache,
        category="fixture",
        refresh=True,
        transport=FakeTransport(
            {
                robots_url: robots,
                start: _response(start, "<html><body>new cache</body></html>"),
            }
        ),
        sleep=lambda value: None,
    )
    assert refreshed.load()[0].content == "<html><body>new cache</body></html>"


def test_http_refresh_commit_cleanup_failure_keeps_new_cache_replayable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = "https://example.com/docs/"
    robots_url, robots = _robots("https://example.com")
    data_root = tmp_path / "dataset"
    cache = data_root / "data/raw"
    BoundedHttpHtmlLoader(
        _settings(max_pages=1),
        cache,
        category="fixture",
        transport=FakeTransport(
            {
                robots_url: robots,
                start: _response(start, "<html><body>old cache</body></html>"),
            }
        ),
        sleep=lambda value: None,
    ).load()
    refreshed_loader = BoundedHttpHtmlLoader(
        _settings(max_pages=1),
        cache,
        category="fixture",
        refresh=True,
        transport=FakeTransport(
            {
                robots_url: robots,
                start: _response(start, "<html><body>new cache</body></html>"),
            }
        ),
        sleep=lambda value: None,
    )
    refreshed = refreshed_loader.load()
    real_rmtree = loading.shutil.rmtree

    def fail_retired_cleanup(path: Path, *args: Any, **kwargs: Any) -> None:
        if ".raw.backup.retired-" in path.name:
            (path / "fetch_manifest.json").unlink()
            raise OSError("injected partial cleanup failure")
        real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(loading.shutil, "rmtree", fail_retired_cleanup)
    refreshed_loader.commit_source()
    no_network = FakeTransport({})

    replayed = BoundedHttpHtmlLoader(
        _settings(max_pages=1),
        cache,
        category="fixture",
        offline=True,
        transport=no_network,
    ).load()

    assert [document.content for document in replayed] == [refreshed[0].content]
    assert no_network.calls == []
    assert not (data_root / "data/.raw.backup").exists()


def test_generic_parser_selector_priority_exclusion_and_structure() -> None:
    html = """
    <html><head><title>Fallback title</title></head><body>
      <main><h1>Main only</h1><p>wrong root text</p></main>
      <article>
        <nav>Navigation contamination</nav>
        <h1 id="intro">Article title</h1>
        <p>Lead with <code>uv sync --frozen</code>.</p>
        <h2 id="usage">Usage</h2>
        <pre>uv run --\n  python</pre>
        <ul><li>First option</li><li>Second option</li></ul>
        <table><tr><th>Name</th><th>Value</th></tr><tr><td>A</td><td>B</td></tr></table>
        <footer>Footer contamination</footer>
      </article>
    </body></html>
    """

    result = GenericHtmlParser(_parser_settings()).parse(_document(html))

    assert [item.section_title for item in result.sections] == [
        "Article title",
        "Usage",
    ]
    assert result.sections[0].source_url.endswith("#intro")
    assert "uv sync --frozen" in result.sections[0].text
    assert "uv run --\n  python" in result.sections[1].text
    assert "- First option\n- Second option" in result.sections[1].text
    assert "Name | Value\nA | B" in result.sections[1].text
    combined = "\n".join(item.text for item in result.sections)
    assert "Navigation contamination" not in combined
    assert "Footer contamination" not in combined
    assert "wrong root text" not in combined
    assert result.code_block_count == 1
    assert result.table_count == 1
    assert result.excluded_node_count == 2


def test_generic_parser_headingless_fallback_and_duplicate_anchor() -> None:
    fallback_settings = _parser_settings(
        content_selectors=("article",),
        fallback_to_body=True,
    )
    headingless = "<html><body><p>Headingless page content.</p></body></html>"
    result = GenericHtmlParser(fallback_settings).parse(_document(headingless))
    assert len(result.sections) == 1
    assert result.used_fallback

    duplicate = (
        "<article><h1 id='same'>One</h1><p>First section text.</p>"
        "<h2 id='same'>Two</h2><p>Second section text.</p></article>"
    )
    parsed = GenericHtmlParser(_parser_settings()).parse(_document(duplicate))
    assert [item.anchor for item in parsed.sections] == ["same", "same-2"]
    assert GenericHtmlParser(_parser_settings()).parse(_document(duplicate)) == parsed


def test_generic_parser_reports_missing_or_empty_content() -> None:
    parser = GenericHtmlParser(_parser_settings(fallback_to_body=False))
    with pytest.raises(ValueError, match="selector"):
        parser.parse(_document("<html><body><div>nothing</div></body></html>"))


def test_generic_parser_contains_no_site_specific_selector_literal() -> None:
    source = Path("src/python_doc_rag/parsing.py").read_text(encoding="utf-8")
    assert "md-content" not in source
    assert "astral" not in source.casefold()
