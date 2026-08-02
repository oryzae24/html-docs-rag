"""Backward-compatible imports for the Python/Sphinx parser."""

from python_doc_rag.sites.python_docs.parser import (
    HtmlParseResult,
    build_python_doc_url,
    parse_python_doc_html,
    parse_python_doc_html_result,
)

__all__ = [
    "HtmlParseResult",
    "build_python_doc_url",
    "parse_python_doc_html",
    "parse_python_doc_html_result",
]
