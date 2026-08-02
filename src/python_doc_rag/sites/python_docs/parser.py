"""Highly tuned section extraction for Python's Sphinx documentation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from urllib.parse import quote, unquote, urljoin, urlsplit, urlunsplit

from bs4 import BeautifulSoup, Tag

from python_doc_rag.config import MIN_SECTION_TEXT_LENGTH
from python_doc_rag.ingestion.protocols import DocumentParseResult
from python_doc_rag.models import DocumentSection, SourceDocument
from python_doc_rag.site_config import PythonSphinxParserSettings
from python_doc_rag.sites.python_docs.constants import (
    PYTHON_DOC_BASE_URL,
    PYTHON_DOC_VERSION,
)

_HEADING_NAMES = {"h1", "h2", "h3"}
_CONTENT_NAMES = {"p", "pre", "ul", "ol", "dl", "table", "blockquote"}
_HEADING_LIKE_NAMES = {"h4", "h5", "h6", "dt"}
_HEADING_LIKE_CLASSES = {"question", "rubric", "topic-title"}
_NOISE_SELECTOR = ", ".join(
    (
        "script",
        "style",
        "nav",
        "footer",
        "header",
        "aside",
        "noscript",
        "form",
        ".related",
        ".related-pages",
        ".sphinxsidebar",
        ".search",
        ".searchbox",
        ".search-form",
        ".navigation",
        ".navbar",
        ".breadcrumbs",
        ".breadcrumb",
        ".wy-nav-side",
        ".wy-nav-top",
        ".rst-footer-buttons",
        ".prev-next-area",
        ".footnote-reference",
        ".footnote-backref",
        "[role='search']",
        "[role='navigation']",
        "[role='doc-noteref']",
    )
)


@dataclass(frozen=True, slots=True)
class HtmlParseResult:
    """Sections and exclusion counts produced while parsing one HTML page."""

    sections: tuple[DocumentSection, ...]
    excluded_section_count: int


class PythonSphinxHtmlParserAdapter:
    """Expose the tuned Python/Sphinx parser through the common protocol."""

    def __init__(self, settings: PythonSphinxParserSettings | None = None) -> None:
        """Use explicit site settings, retaining no-argument API compatibility."""
        if settings is not None and settings.type != "python-sphinx":
            raise ValueError(
                "PythonSphinxHtmlParserAdapter requires parser.type=python-sphinx"
            )
        self._python_version = settings.python_version if settings is not None else None
        self._minimum_section_text_length = (
            settings.minimum_section_text_length
            if settings is not None
            else MIN_SECTION_TEXT_LENGTH
        )

    def parse(self, document: SourceDocument) -> DocumentParseResult:
        """Parse one page with configured or historical citation semantics."""
        configured = self._python_version is not None
        python_version = self._python_version
        if python_version is None:
            python_version = document.metadata.get(
                "python_version",
                PYTHON_DOC_VERSION,
            )
            if not isinstance(python_version, str):
                raise TypeError("python_version metadata must be a string")
        parsed = parse_python_doc_html_result(
            document.content,
            source_path=document.logical_path,
            category=document.category,
            python_version=python_version,
            min_section_text_length=self._minimum_section_text_length,
            source_url=document.source_url if configured else None,
        )
        return DocumentParseResult(
            sections=parsed.sections,
            excluded_section_count=parsed.excluded_section_count,
        )


def build_python_doc_url(
    source_path: str | PurePosixPath,
    *,
    anchor: str = "",
    base_url: str = PYTHON_DOC_BASE_URL,
) -> str:
    """Join a relative archive path to the trusted Python docs base URL."""
    parsed_base = urlsplit(base_url)
    if parsed_base.scheme != "https" or parsed_base.hostname != "docs.python.org":
        raise ValueError("base_url must be an HTTPS URL on docs.python.org")

    raw_path = str(source_path).replace("\\", "/")
    parsed_path = urlsplit(raw_path)
    if (
        parsed_path.scheme
        or parsed_path.netloc
        or parsed_path.query
        or parsed_path.fragment
    ):
        raise ValueError("source_path must be a relative path without URL components")

    path = PurePosixPath(parsed_path.path)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("source_path must stay within the documentation archive")

    relative_path = quote(path.as_posix(), safe="/-._~")
    url = urljoin(base_url.rstrip("/") + "/", relative_path)
    parsed_url = urlsplit(url)
    if (
        parsed_url.scheme != parsed_base.scheme
        or parsed_url.netloc != parsed_base.netloc
    ):
        raise ValueError("generated URL escaped the Python docs origin")

    if anchor:
        url = f"{url}#{quote(anchor, safe='-._~')}"
    return url


def parse_python_doc_html(
    html: str,
    *,
    source_path: str | PurePosixPath,
    category: str | None = None,
    python_version: str = PYTHON_DOC_VERSION,
    base_url: str = PYTHON_DOC_BASE_URL,
    source_url: str | None = None,
) -> list[DocumentSection]:
    """Parse HTML into non-empty sections keyed by h1, h2, and h3 headings."""
    result = parse_python_doc_html_result(
        html,
        source_path=source_path,
        category=category,
        python_version=python_version,
        base_url=base_url,
        source_url=source_url,
    )
    return list(result.sections)


def parse_python_doc_html_result(
    html: str,
    *,
    source_path: str | PurePosixPath,
    category: str | None = None,
    python_version: str = PYTHON_DOC_VERSION,
    base_url: str = PYTHON_DOC_BASE_URL,
    min_section_text_length: int = MIN_SECTION_TEXT_LENGTH,
    source_url: str | None = None,
) -> HtmlParseResult:
    """Parse one page and report sections excluded for having too little text."""
    if min_section_text_length < 1:
        raise ValueError("min_section_text_length must be greater than zero")

    normalized_path = _normalize_source_path(source_path)
    trusted_source_url = (
        _validated_source_url(source_url) if source_url is not None else None
    )
    soup = BeautifulSoup(html, "lxml")
    anchor_titles = _anchor_titles(soup)

    for noise in list(soup.select(_NOISE_SELECTOR)):
        if noise.parent is not None:
            noise.decompose()

    page_title = _page_title(soup, normalized_path)
    content_root = _find_content_root(soup)
    if content_root is None:
        raise ValueError("could not locate a document content area")

    resolved_category = category or PurePosixPath(normalized_path).parts[0]
    sections: list[DocumentSection] = []
    excluded_section_count = 0
    heading: Tag | None = None
    blocks: list[str] = []

    for element in content_root.find_all(_HEADING_NAMES | _CONTENT_NAMES):
        if element.name in _HEADING_NAMES:
            excluded_section_count += _append_section(
                sections,
                heading,
                blocks,
                page_title=page_title,
                source_path=normalized_path,
                source_url=trusted_source_url,
                category=resolved_category,
                python_version=python_version,
                base_url=base_url,
                min_section_text_length=min_section_text_length,
                anchor_titles=anchor_titles,
            )
            heading = element
            blocks = []
        elif heading is not None and not element.find_parent(_CONTENT_NAMES):
            text = _block_text(element)
            if text:
                blocks.append(text)

    excluded_section_count += _append_section(
        sections,
        heading,
        blocks,
        page_title=page_title,
        source_path=normalized_path,
        source_url=trusted_source_url,
        category=resolved_category,
        python_version=python_version,
        base_url=base_url,
        min_section_text_length=min_section_text_length,
        anchor_titles=anchor_titles,
    )
    return HtmlParseResult(tuple(sections), excluded_section_count)


def _find_content_root(soup: BeautifulSoup) -> Tag | None:
    """Select the most specific known Python/Sphinx document body."""
    selectors = ("main", "[role='main']", "div.body")
    for selector in selectors:
        candidate = soup.select_one(selector)
        if isinstance(candidate, Tag):
            return candidate
    return soup.body if isinstance(soup.body, Tag) else None


def _normalize_source_path(source_path: str | PurePosixPath) -> str:
    raw_path = str(source_path).replace("\\", "/")
    parsed_path = urlsplit(raw_path)
    path = PurePosixPath(parsed_path.path)
    if (
        parsed_path.scheme
        or parsed_path.netloc
        or parsed_path.query
        or parsed_path.fragment
        or path.is_absolute()
        or ".." in path.parts
        or not path.parts
    ):
        raise ValueError("source_path must be a safe relative path")
    return path.as_posix()


def _validated_source_url(source_url: str) -> str:
    parsed = urlsplit(source_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("source_url must be an absolute HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("source_url must not contain authentication information")
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, parsed.query, ""))


def _source_url_with_anchor(source_url: str, anchor: str) -> str:
    parsed = urlsplit(source_url)
    return urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, parsed.query, quote(anchor, safe="-._~"))
    )


def _page_title(soup: BeautifulSoup, source_path: str) -> str:
    if soup.title:
        title = soup.title.get_text(" ", strip=True)
        if title:
            return title
    first_heading = soup.find(_HEADING_NAMES)
    if first_heading:
        return _heading_text(first_heading)
    return PurePosixPath(source_path).stem


def _heading_text(heading: Tag) -> str:
    return _text_without_permalink(heading)


def _text_without_permalink(element: Tag) -> str:
    """Extract text while dropping only Sphinx's standalone permalink marker."""
    clone = BeautifulSoup(str(element), "lxml")
    for permalink in clone.select(".headerlink"):
        if permalink.get_text(" ", strip=True) == "¶" and not permalink.find_parent(
            ("code", "pre")
        ):
            permalink.decompose()
    return " ".join(clone.get_text(" ", strip=True).split())


def _block_text(element: Tag) -> str:
    if element.name == "pre":
        lines = [line.rstrip() for line in element.get_text("", strip=False).splitlines()]
        return "\n".join(lines).strip()
    return _text_without_permalink(element)


def _anchor_titles(soup: BeautifulSoup) -> dict[str, str]:
    """Map local anchor targets to their visible labels before noise is removed."""
    titles: dict[str, str] = {}
    for link in soup.find_all("a", href=True):
        href = str(link.get("href", ""))
        if not href.startswith("#") or len(href) == 1:
            continue
        title = _text_without_permalink(link)
        if title and title != "¶":
            titles.setdefault(unquote(href[1:]), title)
    return titles


def _nearest_heading_like_text(heading: Tag) -> str:
    """Find a nearby semantic label when a heading has no visible text."""
    parent = heading.find_parent(("section", "div"))
    if isinstance(parent, Tag):
        descendants = parent.find_all(True)
        heading_index = next(
            index for index, candidate in enumerate(descendants) if candidate is heading
        )
        candidates: list[tuple[int, Tag]] = []
        for index, candidate in enumerate(descendants):
            if candidate is heading:
                continue
            classes = set(candidate.get("class", ()))
            is_heading_like = (
                candidate.name in _HEADING_NAMES | _HEADING_LIKE_NAMES
                or candidate.get("role") == "heading"
                or bool(classes & _HEADING_LIKE_CLASSES)
                or (
                    candidate.name == "p"
                    and candidate.find("strong", recursive=False) is not None
                )
            )
            if is_heading_like:
                candidates.append((abs(index - heading_index), candidate))

        for _, candidate in sorted(candidates, key=lambda item: item[0]):
            title = _text_without_permalink(candidate)
            if title:
                return title

    for previous_heading in heading.find_all_previous(
        _HEADING_NAMES | _HEADING_LIKE_NAMES
    ):
        title = _text_without_permalink(previous_heading)
        if title:
            return title
    return ""


def _section_title(
    heading: Tag,
    anchor: str,
    anchor_titles: dict[str, str],
) -> str:
    """Resolve a section title without changing normal h1-h3 handling."""
    return (
        _heading_text(heading)
        or anchor_titles.get(anchor, "")
        or _nearest_heading_like_text(heading)
    )


def _heading_anchor(heading: Tag) -> str:
    """Find an anchor on the heading or its nearest section-like parent."""
    heading_id = heading.get("id")
    if heading_id:
        return str(heading_id)

    parent = heading.find_parent(("section", "div"), id=True)
    if isinstance(parent, Tag):
        return str(parent.get("id", ""))
    return ""


def _append_section(
    sections: list[DocumentSection],
    heading: Tag | None,
    blocks: list[str],
    *,
    page_title: str,
    source_path: str,
    source_url: str | None,
    category: str,
    python_version: str,
    base_url: str,
    min_section_text_length: int,
    anchor_titles: dict[str, str],
) -> int:
    text = "\n\n".join(blocks).strip()
    if heading is None:
        return 0
    if len(text) < min_section_text_length:
        return 1

    anchor = _heading_anchor(heading)
    section_url = (
        build_python_doc_url(source_path, anchor=anchor, base_url=base_url)
        if source_url is None
        else _source_url_with_anchor(source_url, anchor)
    )
    sections.append(
        DocumentSection(
            text=text,
            page_title=page_title,
            section_title=_section_title(heading, anchor, anchor_titles),
            source_path=source_path,
            source_url=section_url,
            anchor=anchor,
            category=category,
            python_version=python_version,
        )
    )
    return 0


__all__ = [
    "HtmlParseResult",
    "PythonSphinxHtmlParserAdapter",
    "build_python_doc_url",
    "parse_python_doc_html",
    "parse_python_doc_html_result",
]
