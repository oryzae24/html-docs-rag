from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from python_doc_rag.service_config import (
    SERVICE_CONFIG_REVISION,
    ServiceConfigError,
    load_service_config,
)


def _service_toml(
    *,
    revision: object = SERVICE_CONFIG_REVISION,
    profile: object = "recommended-v2",
    device: object = "cuda",
    knowledge_bases: str | None = None,
) -> str:
    rows = knowledge_bases or """
[[knowledge_bases]]
id = "python-docs"
display_name = "Python documentation"
data_root = "data/python"

[[knowledge_bases]]
id = "uv-docs"
display_name = "uv Documentation"
data_root = "data/uv"
"""
    return (
        f"revision = {_toml_value(revision)}\n"
        f"profile = {_toml_value(profile)}\n"
        f"device = {_toml_value(device)}\n"
        f"{rows.strip()}\n"
    )


def _toml_value(value: object) -> str:
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _write_config(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "config/service.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_loads_two_knowledge_bases_and_preserves_order(tmp_path: Path) -> None:
    path = _write_config(tmp_path, _service_toml())

    config = load_service_config(path)

    assert config.revision == SERVICE_CONFIG_REVISION
    assert config.profile.name == "recommended-v2"
    assert config.device == "cuda"
    assert [item.id for item in config.knowledge_bases] == [
        "python-docs",
        "uv-docs",
    ]
    assert [item.display_name for item in config.knowledge_bases] == [
        "Python documentation",
        "uv Documentation",
    ]


def test_resolves_relative_data_roots_from_config_directory(tmp_path: Path) -> None:
    path = _write_config(tmp_path, _service_toml())

    config = load_service_config(path)

    assert config.knowledge_bases[0].data_root == (
        path.parent / "data/python"
    ).resolve()
    assert config.knowledge_bases[1].data_root == (
        path.parent / "data/uv"
    ).resolve()


def test_accepts_absolute_root_and_expands_home(tmp_path: Path) -> None:
    absolute = (tmp_path / "absolute").resolve()
    rows = f'''
[[knowledge_bases]]
id = "absolute"
display_name = "Absolute"
data_root = "{absolute.as_posix()}"

[[knowledge_bases]]
id = "home"
display_name = "Home"
data_root = "~/knowledge-base-fixture"
'''

    config = load_service_config(
        _write_config(tmp_path, _service_toml(knowledge_bases=rows))
    )

    assert config.knowledge_bases[0].data_root == absolute
    assert config.knowledge_bases[1].data_root == Path(
        "~/knowledge-base-fixture"
    ).expanduser().resolve()


def test_accepts_every_existing_runtime_profile_and_device(tmp_path: Path) -> None:
    for profile in ("default", "recommended-v1", "recommended-v2", "recommended"):
        for device in ("auto", "cpu", "cuda"):
            path = _write_config(
                tmp_path,
                _service_toml(profile=profile, device=device),
            )
            config = load_service_config(path)
            assert config.profile.name == profile
            assert config.device == device


@pytest.mark.parametrize(
    "knowledge_bases",
    [
        """
[[knowledge_bases]]
id = "same"
display_name = "First"
data_root = "data/first"
[[knowledge_bases]]
id = "same"
display_name = "Second"
data_root = "data/second"
""",
        """
[[knowledge_bases]]
id = "first"
display_name = "First"
data_root = "data/shared"
[[knowledge_bases]]
id = "second"
display_name = "Second"
data_root = "data/other/../shared"
""",
    ],
    ids=("duplicate-id", "duplicate-resolved-root"),
)
def test_rejects_duplicate_identity_or_root(
    tmp_path: Path,
    knowledge_bases: str,
) -> None:
    path = _write_config(
        tmp_path,
        _service_toml(knowledge_bases=knowledge_bases),
    )

    with pytest.raises(ServiceConfigError, match="重複"):
        load_service_config(path)


@pytest.mark.parametrize(
    "knowledge_base_id",
    ["Upper", "-leading", "contains.dot", "with space", "a" * 65],
)
def test_rejects_invalid_knowledge_base_slug(
    tmp_path: Path,
    knowledge_base_id: str,
) -> None:
    rows = f"""
[[knowledge_bases]]
id = "{knowledge_base_id}"
display_name = "Display"
data_root = "data/root"
"""
    path = _write_config(tmp_path, _service_toml(knowledge_bases=rows))

    with pytest.raises(ServiceConfigError, match=r"\[a-z0-9\]"):
        load_service_config(path)


def test_rejects_empty_display_name(tmp_path: Path) -> None:
    rows = """
[[knowledge_bases]]
id = "docs"
display_name = "   "
data_root = "data/root"
"""
    path = _write_config(tmp_path, _service_toml(knowledge_bases=rows))

    with pytest.raises(ServiceConfigError, match="display_name.*空"):
        load_service_config(path)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (
            _service_toml() + "unknown = true\n",
            "unknown=\\['unknown'\\]",
        ),
        (
            _service_toml(
                knowledge_bases="""
[[knowledge_bases]]
id = "docs"
display_name = "Display"
data_root = "data/root"
secret = "not-accepted"
"""
            ),
            "unknown=\\['secret'\\]",
        ),
    ],
    ids=("top-level", "knowledge-base"),
)
def test_rejects_unknown_keys(
    tmp_path: Path,
    content: str,
    message: str,
) -> None:
    with pytest.raises(ServiceConfigError, match=message):
        load_service_config(_write_config(tmp_path, content))


def test_rejects_wrong_revision_unknown_profile_and_device(tmp_path: Path) -> None:
    cases = (
        (_service_toml(revision="future-v2"), "revision"),
        (_service_toml(profile="not-a-profile"), "unsupported runtime profile"),
        (_service_toml(device="mps"), "auto、cpu、cuda"),
    )
    for content, message in cases:
        with pytest.raises(ServiceConfigError, match=message):
            load_service_config(_write_config(tmp_path, content))


def test_rejects_zero_knowledge_bases(tmp_path: Path) -> None:
    content = (
        f'revision = "{SERVICE_CONFIG_REVISION}"\n'
        'profile = "recommended-v2"\n'
        'device = "cuda"\n'
        "knowledge_bases = []\n"
    )

    with pytest.raises(ServiceConfigError, match="1件以上"):
        load_service_config(_write_config(tmp_path, content))


@pytest.mark.parametrize(
    "content",
    [
        _service_toml(revision=True),
        _service_toml(profile=True),
        _service_toml(device=True),
        (
            f'revision = "{SERVICE_CONFIG_REVISION}"\n'
            'profile = "recommended-v2"\n'
            'device = "cuda"\n'
            "knowledge_bases = true\n"
        ),
        _service_toml(
            knowledge_bases="""
[[knowledge_bases]]
id = true
display_name = "Display"
data_root = "data/root"
"""
        ),
        _service_toml(
            knowledge_bases="""
[[knowledge_bases]]
id = "docs"
display_name = true
data_root = "data/root"
"""
        ),
        _service_toml(
            knowledge_bases="""
[[knowledge_bases]]
id = "docs"
display_name = "Display"
data_root = true
"""
        ),
    ],
)
def test_rejects_type_mismatches_including_booleans(
    tmp_path: Path,
    content: str,
) -> None:
    with pytest.raises(ServiceConfigError, match="必要があります"):
        load_service_config(_write_config(tmp_path, content))


def test_rejects_missing_keys_and_invalid_toml(tmp_path: Path) -> None:
    with pytest.raises(ServiceConfigError, match="missing"):
        load_service_config(
            _write_config(
                tmp_path,
                f'revision = "{SERVICE_CONFIG_REVISION}"\n',
            )
        )
    with pytest.raises(ServiceConfigError, match="TOMLが不正"):
        load_service_config(_write_config(tmp_path, "revision = [\n"))


def test_config_and_nested_records_are_frozen(tmp_path: Path) -> None:
    config = load_service_config(_write_config(tmp_path, _service_toml()))

    with pytest.raises(FrozenInstanceError):
        config.device = "cpu"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        config.knowledge_bases[0].display_name = "changed"  # type: ignore[misc]
