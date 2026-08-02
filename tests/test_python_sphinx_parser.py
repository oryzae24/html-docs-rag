import hashlib
from pathlib import Path

from python_doc_rag.models import SourceDocument
from python_doc_rag.site_config import PythonSphinxParserSettings
from python_doc_rag.sites.python_docs.parser import PythonSphinxHtmlParserAdapter

FIXTURES = Path(__file__).parent / "fixtures"


def _document(content: str) -> SourceDocument:
    source_url = "https://trusted.example.test/frozen/page.html?view=full"
    return SourceDocument(
        source_url=source_url,
        canonical_url=source_url,
        content=content,
        content_sha256=hashlib.sha256(content.encode()).hexdigest(),
        source_kind="fixture-archive",
        logical_path="tutorial/example.html",
        category="tutorial",
    )


def test_python_adapter_uses_settings_and_loader_provided_source_url() -> None:
    content = (FIXTURES / "sample_python_doc.html").read_text(encoding="utf-8")
    settings = PythonSphinxParserSettings(
        type="python-sphinx",
        python_version="3.13-fixture",
        minimum_section_text_length=1,
    )

    parsed = PythonSphinxHtmlParserAdapter(settings).parse(_document(content))

    assert parsed.sections
    assert {section.python_version for section in parsed.sections} == {"3.13-fixture"}
    assert all(
        section.source_url.startswith(
            "https://trusted.example.test/frozen/page.html?view=full#"
        )
        for section in parsed.sections
    )


def test_python_adapter_uses_configured_minimum_section_length() -> None:
    content = (FIXTURES / "sample_python_doc.html").read_text(encoding="utf-8")
    settings = PythonSphinxParserSettings(
        type="python-sphinx",
        python_version="3.13",
        minimum_section_text_length=100_000,
    )

    parsed = PythonSphinxHtmlParserAdapter(settings).parse(_document(content))

    assert parsed.sections == ()
    assert parsed.excluded_section_count > 0
