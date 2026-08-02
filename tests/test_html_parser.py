from pathlib import Path

import pytest

from python_doc_rag.html_parser import build_python_doc_url, parse_python_doc_html

FIXTURE = Path(__file__).parent / "fixtures" / "sample_python_doc.html"
PARENT_ANCHOR_FIXTURE = (
    Path(__file__).parent / "fixtures" / "sample_python_doc_parent_anchor.html"
)
FAQ_FIXTURE = Path(__file__).parent / "fixtures" / "sample_python_doc_faq.html"


def test_parse_python_doc_html_extracts_heading_sections() -> None:
    sections = parse_python_doc_html(
        FIXTURE.read_text(encoding="utf-8"),
        source_path="tutorial/controlflow.html",
    )

    assert len(sections) == 3
    assert sections[0].page_title == "制御フローツール - Python 3.13.0 ドキュメント"
    assert sections[0].section_title == "その他の制御フローツール"
    assert sections[0].anchor == "controlflow"
    assert sections[0].category == "tutorial"
    assert sections[0].python_version == "3.13"
    assert sections[0].source_path == "tutorial/controlflow.html"
    assert sections[0].source_url == (
        "https://docs.python.org/ja/3.13/tutorial/controlflow.html#controlflow"
    )

    assert "if value:" in sections[1].text
    assert "前のトピック" not in " ".join(section.text for section in sections)
    assert "Copyright" not in " ".join(section.text for section in sections)
    assert all(section.section_title != "空の節" for section in sections)


@pytest.mark.parametrize(
    "source_path",
    [
        "https://example.com/page.html",
        "file:///tmp/page.html",
        "//example.com/page.html",
        "/tmp/page.html",
        "../secret.html",
    ],
)
def test_build_python_doc_url_rejects_unsafe_paths(source_path: str) -> None:
    with pytest.raises(ValueError):
        build_python_doc_url(source_path)


def test_parent_anchor_code_and_navigation_noise_are_handled() -> None:
    sections = parse_python_doc_html(
        PARENT_ANCHOR_FIXTURE.read_text(encoding="utf-8"),
        source_path="library/functions.html",
    )

    assert [section.anchor for section in sections] == ["built-in-functions", "details"]
    assert sections[0].source_url.endswith("functions.html#built-in-functions")
    assert "for item in items:\n    print(item)" in sections[0].text
    combined_text = " ".join(section.text for section in sections)
    assert "ドキュメント内を検索" not in combined_text
    assert "前のトピック" not in combined_text
    assert "Copyright" not in combined_text
    assert "[1]" not in combined_text


def test_faq_title_and_standalone_permalink_are_handled() -> None:
    sections = parse_python_doc_html(
        FAQ_FIXTURE.read_text(encoding="utf-8"),
        source_path="faq/programming.html",
    )

    faq_section = next(section for section in sections if section.anchor == "standalone-binary")
    assert faq_section.section_title == "スタンドアロンバイナリを作るには？"

    anchor_fallback = next(
        section for section in sections if section.anchor == "anchor-only-question"
    )
    assert anchor_fallback.section_title == "アンカーだけの質問は解決できますか？"

    nearby_fallback = next(
        section for section in sections if section.anchor == "nearby-label-question"
    )
    assert nearby_fallback.section_title == "近くの見出し相当要素を使えますか？"


def test_standalone_permalink_is_removed_without_changing_content() -> None:
    sections = parse_python_doc_html(
        FAQ_FIXTURE.read_text(encoding="utf-8"),
        source_path="faq/programming.html",
    )

    faq_section = next(section for section in sections if section.anchor == "standalone-binary")
    assert "この段落の はパーマリンクです。" in faq_section.text
    assert "段落記号 ¶ は内容として残します。" in faq_section.text
    assert 'inline_marker = "¶"' in faq_section.text
    assert 'block_marker = "¶"' in faq_section.text
