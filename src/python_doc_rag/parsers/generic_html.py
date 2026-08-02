"""Configurable site-neutral HTML section extraction."""

from __future__ import annotations

from urllib.parse import quote, urlsplit, urlunsplit

from bs4 import BeautifulSoup, Tag

from python_doc_rag.ingestion.protocols import DocumentParseResult
from python_doc_rag.models import DocumentSection, SourceDocument
from python_doc_rag.site_config import GenericHtmlParserSettings

_GENERIC_BLOCK_NAMES = {
    "p",
    "pre",
    "ul",
    "ol",
    "dl",
    "table",
    "blockquote",
}
_ALWAYS_EXCLUDED_SELECTORS = ("script", "style", "noscript", "template")


class GenericHtmlParser:
    """Parse configured static article/main HTML without site-specific classes."""

    def __init__(self, settings: GenericHtmlParserSettings) -> None:
        if settings.type != "generic-html":
            raise ValueError("GenericHtmlParser requires parser.type=generic-html")
        self._settings = settings

    def parse(self, document: SourceDocument) -> DocumentParseResult:
        """Extract deterministic heading sections and configured diagnostics."""
        soup = BeautifulSoup(document.content, "lxml")
        page_title = _generic_page_title(soup, self._settings, document.logical_path)
        selected: Tag | None = None
        for selector in self._settings.content_selectors:
            candidate = soup.select_one(selector)
            if isinstance(candidate, Tag) and candidate.get_text(" ", strip=True):
                selected = candidate
                break
        used_fallback = False
        if selected is None and self._settings.fallback_to_body:
            selected = soup.body if isinstance(soup.body, Tag) else None
            used_fallback = selected is not None
        if selected is None:
            raise ValueError("no configured content selector matched")

        content = BeautifulSoup(str(selected), "lxml")
        root: Tag | None = None
        if content.body is not None:
            if selected.name == "body":
                root = content.body
            else:
                candidate = content.body.find(selected.name)
                root = candidate if isinstance(candidate, Tag) else content.body
        else:
            candidate = content.find()
            root = candidate if isinstance(candidate, Tag) else None
        if not isinstance(root, Tag):
            raise ValueError("selected content root was empty")
        excluded_count = 0
        selectors = (*_ALWAYS_EXCLUDED_SELECTORS, *self._settings.exclude_selectors)
        for selector in selectors:
            for node in list(root.select(selector)):
                if node.parent is not None:
                    node.decompose()
                    excluded_count += 1
        for node in list(root.select("[hidden], [aria-hidden='true']")):
            if node.parent is not None:
                node.decompose()
                excluded_count += 1

        heading_names = {f"h{level}" for level in self._settings.heading_levels}
        code_block_count = len(root.find_all("pre"))
        table_count = len(root.find_all("table"))
        sections: list[DocumentSection] = []
        excluded_sections = 0
        heading: Tag | None = None
        blocks: list[str] = []
        anchors: dict[str, int] = {}
        for element in root.find_all(heading_names | _GENERIC_BLOCK_NAMES):
            if element.name in heading_names:
                if heading is not None or self._settings.include_lead_text:
                    excluded_sections += _append_generic_section(
                        sections,
                        document=document,
                        page_title=page_title,
                        heading=heading,
                        blocks=blocks,
                        anchors=anchors,
                        minimum_length=self._settings.minimum_section_text_length,
                    )
                heading = element
                blocks = []
                continue
            if element.find_parent(_GENERIC_BLOCK_NAMES):
                continue
            text = _generic_block_text(element)
            if text:
                blocks.append(text)

        if heading is not None:
            excluded_sections += _append_generic_section(
                sections,
                document=document,
                page_title=page_title,
                heading=heading,
                blocks=blocks,
                anchors=anchors,
                minimum_length=self._settings.minimum_section_text_length,
            )
        elif self._settings.include_lead_text:
            excluded_sections += _append_generic_section(
                sections,
                document=document,
                page_title=page_title,
                heading=None,
                blocks=blocks,
                anchors=anchors,
                minimum_length=self._settings.minimum_section_text_length,
            )
        if not sections and not blocks:
            raise ValueError("configured content root contained no document text")
        return DocumentParseResult(
            sections=tuple(sections),
            excluded_section_count=excluded_sections,
            excluded_node_count=excluded_count,
            code_block_count=code_block_count,
            table_count=table_count,
            used_fallback=used_fallback,
        )


def _generic_page_title(
    soup: BeautifulSoup,
    settings: GenericHtmlParserSettings,
    logical_path: str,
) -> str:
    for selector in settings.title_selectors:
        node = soup.select_one(selector)
        if isinstance(node, Tag):
            text = _normalize_text(node.get_text(" ", strip=True))
            if text:
                return text
    return logical_path.rsplit("/", maxsplit=1)[-1]


def _generic_block_text(element: Tag) -> str:
    if element.name == "pre":
        lines = [line.rstrip() for line in element.get_text("", strip=False).splitlines()]
        return "\n".join(lines).strip()
    if element.name == "table":
        rows: list[str] = []
        for row in element.find_all("tr"):
            cells = [
                _normalize_text(cell.get_text(" ", strip=True))
                for cell in row.find_all(("th", "td"))
            ]
            if any(cells):
                rows.append(" | ".join(cells))
        return "\n".join(rows)
    if element.name in {"ul", "ol"}:
        items = [
            _normalize_text(item.get_text(" ", strip=True))
            for item in element.find_all("li", recursive=False)
        ]
        return "\n".join(f"- {item}" for item in items if item)
    if element.name == "dl":
        return "\n".join(
            _normalize_text(item.get_text(" ", strip=True))
            for item in element.find_all(("dt", "dd"), recursive=False)
            if item.get_text(" ", strip=True)
        )
    return _normalize_text(element.get_text(" ", strip=True))


def _append_generic_section(
    sections: list[DocumentSection],
    *,
    document: SourceDocument,
    page_title: str,
    heading: Tag | None,
    blocks: list[str],
    anchors: dict[str, int],
    minimum_length: int,
) -> int:
    text = "\n\n".join(blocks).strip()
    if len(text) < minimum_length:
        return int(bool(text) or heading is not None)
    title = page_title
    raw_anchor = ""
    if heading is not None:
        title = _normalize_text(heading.get_text(" ", strip=True)) or page_title
        raw_anchor = str(heading.get("id", ""))
        if not raw_anchor:
            parent = heading.find_parent(id=True)
            if isinstance(parent, Tag):
                raw_anchor = str(parent.get("id", ""))
    anchor = _deduplicate_anchor(raw_anchor, anchors)
    sections.append(
        DocumentSection(
            text=text,
            page_title=page_title,
            section_title=title,
            source_path=document.logical_path,
            source_url=_source_url_with_anchor(document.source_url, anchor),
            anchor=anchor,
            category=document.category,
            python_version="",
        )
    )
    return 0


def _deduplicate_anchor(value: str, anchors: dict[str, int]) -> str:
    if not value:
        return ""
    count = anchors.get(value, 0) + 1
    anchors[value] = count
    return value if count == 1 else f"{value}-{count}"


def _source_url_with_anchor(source_url: str, anchor: str) -> str:
    parsed = urlsplit(source_url)
    return urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, parsed.query, quote(anchor, safe="-._~"))
    )


def _normalize_text(value: str) -> str:
    return " ".join(value.split())
