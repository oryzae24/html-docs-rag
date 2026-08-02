import ast
import subprocess
import sys
from pathlib import Path

from python_doc_rag.ingestion.registry import build_document_parser
from python_doc_rag.parsers.generic_html import GenericHtmlParser
from python_doc_rag.site_config import (
    GenericHtmlParserSettings,
    PythonSphinxParserSettings,
)
from python_doc_rag.sites.python_docs.parser import PythonSphinxHtmlParserAdapter


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules


def test_common_pipeline_loaders_and_generic_parser_do_not_import_python_site() -> None:
    paths = (
        Path("src/python_doc_rag/ingestion/protocols.py"),
        Path("src/python_doc_rag/ingestion/orchestration.py"),
        Path("src/python_doc_rag/ingestion/serialization.py"),
        Path("src/python_doc_rag/loading.py"),
        Path("src/python_doc_rag/loaders/urls.py"),
        Path("src/python_doc_rag/loaders/zip_archive.py"),
        Path("src/python_doc_rag/parsers/generic_html.py"),
        Path("src/python_doc_rag/preparation.py"),
    )

    for path in paths:
        assert not any(
            module.startswith("python_doc_rag.sites.python_docs")
            for module in _imports(path)
        ), path


def test_common_preparation_does_not_depend_on_legacy_python_corpus() -> None:
    imports = _imports(Path("src/python_doc_rag/preparation.py"))

    assert "python_doc_rag.corpus" not in imports
    assert "python_doc_rag.ingestion.serialization" in imports


def test_parser_registry_resolves_types_independently() -> None:
    python_parser = build_document_parser(
        PythonSphinxParserSettings(
            type="python-sphinx",
            python_version="3.13",
            minimum_section_text_length=20,
        )
    )
    generic_parser = build_document_parser(
        GenericHtmlParserSettings(
            type="generic-html",
            content_selectors=("main",),
            exclude_selectors=(),
            title_selectors=("title", "h1"),
            heading_levels=(1, 2, 3),
            minimum_section_text_length=20,
            fallback_to_body=False,
            include_lead_text=True,
        )
    )

    assert isinstance(python_parser, PythonSphinxHtmlParserAdapter)
    assert isinstance(generic_parser, GenericHtmlParser)


def test_registry_is_only_core_module_registering_python_parser() -> None:
    core_paths = (
        *Path("src/python_doc_rag/ingestion").glob("*.py"),
        *Path("src/python_doc_rag/parsers").glob("*.py"),
    )
    importing = {
        path
        for path in core_paths
        if any(
            module == "python_doc_rag.sites.python_docs.parser"
            for module in _imports(path)
        )
    }

    assert importing == {Path("src/python_doc_rag/ingestion/registry.py")}


def test_common_protocol_import_does_not_eagerly_load_python_site_parser() -> None:
    script = (
        "import sys; "
        "import python_doc_rag.ingestion.protocols; "
        "assert 'python_doc_rag.sites.python_docs.parser' not in sys.modules"
    )

    subprocess.run([sys.executable, "-c", script], check=True)
