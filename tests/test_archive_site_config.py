from __future__ import annotations

from pathlib import Path

import pytest

from python_doc_rag.site_config import (
    PinnedLocalArchiveLoaderSettings,
    PythonSphinxParserSettings,
    SnapshotHttpArchiveLoaderSettings,
    load_site_config,
)

SHA256 = "a" * 64


def _site_toml(loader: str, *, parser_extra: str = "") -> str:
    return f"""[dataset]
name = "Fixture documentation"
slug = "fixture-docs"
language = "ja"
description = "archive fixture"

[loader]
{loader}

[parser]
type = "python-sphinx"
python_version = "3.13"
minimum_section_text_length = 20
{parser_extra}

[chunking]
chunk_size = 1000
chunk_overlap = 150

[index]
embedding_model = "fixture/model"
embedding_batch_size = 8

[profile]
prepare = "none"
"""


def _pinned_loader() -> str:
    return f'''type = "pinned-local-archive"
archive_path = "../../resources/snapshot.zip"
archive_sha256 = "{SHA256}"
archive_format = "zip"
archive_root = "python-docs-html"
original_archive_url = "https://example.test/archive.zip"
source_base_url = "https://example.test/docs/"
include_path_prefixes = ["tutorial/", "library/"]
max_archive_bytes = 1000000
max_members = 100
max_member_bytes = 100000
max_extracted_bytes = 500000'''


def _snapshot_loader() -> str:
    return '''type = "snapshot-http-archive"
archive_url = "https://example.test/archive.zip"
archive_format = "zip"
archive_root = "python-docs-html"
source_base_url = "https://example.test/docs/"
include_path_prefixes = ["tutorial/", "library/"]
update_policy = "manual"
timeout_seconds = 60
max_retries = 2
max_archive_bytes = 1000000
max_members = 100
max_member_bytes = 100000
max_extracted_bytes = 500000
user_agent = "fixture/1.0"'''


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "site.toml"
    path.write_text(text, encoding="utf-8")
    return path


def _replace(text: str, old: str, new: str) -> str:
    assert old in text
    return text.replace(old, new)


def test_pinned_archive_settings_parse_as_strict_variant(tmp_path: Path) -> None:
    config = load_site_config(_write(tmp_path, _site_toml(_pinned_loader())))

    assert isinstance(config.loader, PinnedLocalArchiveLoaderSettings)
    assert config.loader.archive_path == "../../resources/snapshot.zip"
    assert config.loader.archive_sha256 == SHA256
    assert config.loader.archive_root == "python-docs-html"
    assert config.loader.include_path_prefixes == ("tutorial/", "library/")
    assert isinstance(config.parser, PythonSphinxParserSettings)
    assert config.parser.python_version == "3.13"


def test_snapshot_http_settings_parse_as_strict_variant(tmp_path: Path) -> None:
    config = load_site_config(_write(tmp_path, _site_toml(_snapshot_loader())))

    assert isinstance(config.loader, SnapshotHttpArchiveLoaderSettings)
    assert config.loader.archive_url == "https://example.test/archive.zip"
    assert config.loader.update_policy == "manual"
    assert config.loader.timeout_seconds == 60
    assert config.loader.max_retries == 2


@pytest.mark.parametrize("suffix", ("?token=secret", "#download"))
def test_snapshot_archive_url_rejects_persisted_query_or_fragment(
    tmp_path: Path,
    suffix: str,
) -> None:
    loader = _replace(
        _snapshot_loader(),
        'archive_url = "https://example.test/archive.zip"',
        f'archive_url = "https://example.test/archive.zip{suffix}"',
    )

    with pytest.raises(ValueError, match="query or fragment"):
        load_site_config(_write(tmp_path, _site_toml(loader)))


@pytest.mark.parametrize(
    "loader",
    (
        _pinned_loader() + "\nunexpected = true",
        _snapshot_loader() + "\nunexpected = true",
    ),
)
def test_archive_settings_reject_unknown_loader_keys(
    tmp_path: Path,
    loader: str,
) -> None:
    with pytest.raises(ValueError, match="unknown loader"):
        load_site_config(_write(tmp_path, _site_toml(loader)))


def test_archive_settings_reject_unknown_parser_keys(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown parser"):
        load_site_config(
            _write(
                tmp_path,
                _site_toml(_pinned_loader(), parser_extra='selector = "main"'),
            )
        )


@pytest.mark.parametrize(
    ("loader", "required_line", "message"),
    (
        (_pinned_loader(), f'archive_sha256 = "{SHA256}"\n', "archive_sha256"),
        (
            _snapshot_loader(),
            'archive_url = "https://example.test/archive.zip"\n',
            "archive_url",
        ),
        (
            _snapshot_loader(),
            'update_policy = "manual"\n',
            "update_policy",
        ),
    ),
)
def test_archive_settings_reject_missing_required_fields(
    tmp_path: Path,
    loader: str,
    required_line: str,
    message: str,
) -> None:
    with pytest.raises(TypeError, match=message):
        load_site_config(
            _write(tmp_path, _site_toml(_replace(loader, required_line, "")))
        )


@pytest.mark.parametrize(
    ("loader", "old", "new"),
    (
        (
            _snapshot_loader(),
            'archive_url = "https://example.test/archive.zip"',
            'archive_url = "http://example.test/archive.zip"',
        ),
        (
            _snapshot_loader(),
            'archive_url = "https://example.test/archive.zip"',
            'archive_url = "/local/archive.zip"',
        ),
        (
            _pinned_loader(),
            'original_archive_url = "https://example.test/archive.zip"',
            'original_archive_url = "http://example.test/archive.zip"',
        ),
    ),
)
def test_archive_source_urls_must_be_absolute_https(
    tmp_path: Path,
    loader: str,
    old: str,
    new: str,
) -> None:
    with pytest.raises(ValueError, match="HTTPS"):
        load_site_config(_write(tmp_path, _site_toml(_replace(loader, old, new))))


@pytest.mark.parametrize(
    "value",
    (
        "relative/docs/",
        "ftp://example.test/docs/",
        "https://example.test/docs",
        "https://example.test/docs/?query=1",
        "https://example.test/docs/#fragment",
        "https://user:password@example.test/docs/",
    ),
)
def test_source_base_url_is_absolute_query_free_directory(
    tmp_path: Path,
    value: str,
) -> None:
    loader = _replace(
        _pinned_loader(),
        'source_base_url = "https://example.test/docs/"',
        f'source_base_url = "{value}"',
    )

    with pytest.raises(ValueError, match="HTTP|query|fragment|end with|authentication"):
        load_site_config(_write(tmp_path, _site_toml(loader)))


def test_source_base_url_may_use_plain_http_for_citations(tmp_path: Path) -> None:
    loader = _replace(
        _pinned_loader(),
        'source_base_url = "https://example.test/docs/"',
        'source_base_url = "http://example.test/docs/"',
    )

    config = load_site_config(_write(tmp_path, _site_toml(loader)))

    assert config.loader.source_base_url == "http://example.test/docs/"


@pytest.mark.parametrize(
    "replacement",
    (
        'archive_sha256 = "ABCDEF"',
        f'archive_sha256 = "{SHA256.upper()}"',
        f'archive_sha256 = "{"g" * 64}"',
        "archive_sha256 = 123",
    ),
)
def test_pinned_archive_sha_is_required_lowercase_hex(
    tmp_path: Path,
    replacement: str,
) -> None:
    loader = _replace(
        _pinned_loader(),
        f'archive_sha256 = "{SHA256}"',
        replacement,
    )

    with pytest.raises((TypeError, ValueError), match="archive_sha256"):
        load_site_config(_write(tmp_path, _site_toml(loader)))


@pytest.mark.parametrize(
    "replacement",
    (
        'archive_root = "/absolute"',
        'archive_root = "../escape"',
        "archive_root = 'C:\\docs'",
        "archive_root = 'docs\\nested'",
        'archive_root = "docs/"',
        'archive_root = "docs//nested"',
        "archive_root = 123",
    ),
)
def test_archive_root_must_be_safe_normalized_relative_posix_path(
    tmp_path: Path,
    replacement: str,
) -> None:
    loader = _replace(
        _pinned_loader(),
        'archive_root = "python-docs-html"',
        replacement,
    )

    with pytest.raises((TypeError, ValueError), match="archive_root"):
        load_site_config(_write(tmp_path, _site_toml(loader)))


@pytest.mark.parametrize(
    "replacement",
    (
        "include_path_prefixes = []",
        'include_path_prefixes = ["tutorial/", "tutorial/"]',
        'include_path_prefixes = ["tutorial", "tutorial/"]',
        'include_path_prefixes = ["../tutorial/"]',
        'include_path_prefixes = ["/tutorial/"]',
        "include_path_prefixes = ['C:\\tutorial']",
        "include_path_prefixes = ['tutorial\\nested']",
        'include_path_prefixes = ["tutorial//nested/"]',
        'include_path_prefixes = ["tutorial/", 3]',
        'include_path_prefixes = "tutorial/"',
    ),
)
def test_include_prefixes_are_nonempty_unique_safe_posix_paths(
    tmp_path: Path,
    replacement: str,
) -> None:
    loader = _replace(
        _pinned_loader(),
        'include_path_prefixes = ["tutorial/", "library/"]',
        replacement,
    )

    with pytest.raises((TypeError, ValueError), match="include_path_prefixes"):
        load_site_config(_write(tmp_path, _site_toml(loader)))


@pytest.mark.parametrize(
    ("old", "new", "message"),
    (
        ("max_archive_bytes = 1000000", "max_archive_bytes = 0", "max_archive_bytes"),
        (
            "max_archive_bytes = 1000000",
            "max_archive_bytes = 2000000001",
            "must not exceed",
        ),
        ("max_members = 100", "max_members = true", "max_members"),
        ("max_members = 100", "max_members = 1000001", "must not exceed"),
        ("max_member_bytes = 100000", "max_member_bytes = 0", "max_member_bytes"),
        (
            "max_extracted_bytes = 500000",
            "max_extracted_bytes = 8000000001",
            "must not exceed",
        ),
        (
            "max_member_bytes = 100000",
            "max_member_bytes = 600000",
            "must not exceed max_extracted_bytes",
        ),
    ),
)
def test_archive_size_and_member_limits_are_strict(
    tmp_path: Path,
    old: str,
    new: str,
    message: str,
) -> None:
    loader = _replace(_pinned_loader(), old, new)

    with pytest.raises((TypeError, ValueError), match=message):
        load_site_config(_write(tmp_path, _site_toml(loader)))


@pytest.mark.parametrize(
    ("old", "new", "message"),
    (
        ("timeout_seconds = 60", "timeout_seconds = 0", "timeout_seconds"),
        ("timeout_seconds = 60", "timeout_seconds = 301", "timeout_seconds"),
        ("timeout_seconds = 60", "timeout_seconds = nan", "timeout_seconds"),
        ("max_retries = 2", "max_retries = -1", "max_retries"),
        ("max_retries = 2", "max_retries = 6", "max_retries"),
        ("max_retries = 2", "max_retries = 1.5", "max_retries"),
    ),
)
def test_snapshot_operational_limits_are_bounded_and_typed(
    tmp_path: Path,
    old: str,
    new: str,
    message: str,
) -> None:
    loader = _replace(_snapshot_loader(), old, new)

    with pytest.raises((TypeError, ValueError), match=message):
        load_site_config(_write(tmp_path, _site_toml(loader)))


@pytest.mark.parametrize(
    ("loader", "old", "new", "message"),
    (
        (
            _pinned_loader(),
            'archive_format = "zip"',
            'archive_format = "tar"',
            "archive_format",
        ),
        (
            _snapshot_loader(),
            'update_policy = "manual"',
            'update_policy = "automatic"',
            "update_policy",
        ),
    ),
)
def test_archive_format_and_update_policy_fail_closed(
    tmp_path: Path,
    loader: str,
    old: str,
    new: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        load_site_config(_write(tmp_path, _site_toml(_replace(loader, old, new))))


@pytest.mark.parametrize(
    "replacement",
    (
        'archive_path = "/absolute/snapshot.zip"',
        "archive_path = 'C:\\snapshot.zip'",
        "archive_path = 'resources\\snapshot.zip'",
        'archive_path = "resources/"',
    ),
)
def test_pinned_archive_path_is_config_relative_and_portable(
    tmp_path: Path,
    replacement: str,
) -> None:
    loader = _replace(
        _pinned_loader(),
        'archive_path = "../../resources/snapshot.zip"',
        replacement,
    )

    with pytest.raises(ValueError, match="archive_path"):
        load_site_config(_write(tmp_path, _site_toml(loader)))
