import json
from collections import defaultdict

import pytest
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

import app.ingest as ingest
from app.config import MIN_CHUNK_CHARS
from app.ingest import load_and_split_documents, load_html, load_markdown, resolve_markdown_links


def test_chunking_respects_size_limit():
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    long_text = "Kubernetes pod scheduling. " * 200
    doc = Document(page_content=long_text, metadata={"source": "k8s.pdf", "page": 1})

    chunks = splitter.split_documents([doc])

    assert len(chunks) > 1
    assert all(len(c.page_content) <= 850 for c in chunks)  # small buffer over 800
    assert all(c.metadata["source"] == "k8s.pdf" for c in chunks)  # metadata preserved


# --- resolve_markdown_links: every link in an answer must work in the UI -----------------------

SITE = "https://kubernetes.io"


def test_root_relative_link_gets_site_domain():
    # The original bug: (/docs/setup/) resolved against localhost:8000 and 404ed
    assert resolve_markdown_links("See [Setup](/docs/setup/).", SITE) == "See [Setup](https://kubernetes.io/docs/setup/)."


def test_site_with_trailing_slash_does_not_double_the_slash():
    assert resolve_markdown_links("[Setup](/docs/setup/)", "https://kubernetes.io/") == "[Setup](https://kubernetes.io/docs/setup/)"


def test_absolute_link_is_kept_unchanged():
    text = "[kind](https://kind.sigs.k8s.io/docs/)"
    assert resolve_markdown_links(text, SITE) == text


@pytest.mark.parametrize("url", ["build-artifacts.md", "../includes/version.md", "./customize-pipeline.md", "#working-with-pods"])
def test_repo_relative_links_and_anchors_become_plain_text(url):
    # No reliable public URL exists for these, so only the link text is kept
    assert resolve_markdown_links(f"See [the guide]({url}) for details.", SITE) == "See the guide for details."


def test_root_relative_link_without_site_becomes_plain_text():
    assert resolve_markdown_links("See [Setup](/docs/setup/).", None) == "See Setup."


def test_protocol_relative_link_is_not_prefixed_with_site():
    # //cdn.example.com/x is another host, not a path on SITE
    assert resolve_markdown_links("[lib](//cdn.example.com/x.js)", SITE) == "lib"


def test_images_are_removed():
    assert resolve_markdown_links("Before ![Screenshot of the UI](media/ui.png) after", SITE) == "Before  after"


def test_link_with_title_is_resolved():
    assert resolve_markdown_links('[Setup](/docs/setup/ "Setup guide")', SITE) == "[Setup](https://kubernetes.io/docs/setup/)"


def test_links_inside_code_blocks_are_left_alone():
    text = "See [Setup](/docs/setup/).\n```\necho '[not a link](/keep/me)'\n```\n"
    result = resolve_markdown_links(text, SITE)
    assert "[Setup](https://kubernetes.io/docs/setup/)" in result
    assert "echo '[not a link](/keep/me)'" in result


def test_load_markdown_uses_site_from_sites_json(tmp_path):
    (tmp_path / "sites.json").write_text(json.dumps({"pods.md": SITE}), encoding="utf-8")
    (tmp_path / "pods.md").write_text("# Pods\nSee [controllers](/docs/concepts/workloads/controllers/).", encoding="utf-8")
    (tmp_path / "unknown.md").write_text("See [controllers](/docs/concepts/workloads/controllers/).", encoding="utf-8")

    pods = load_markdown(str(tmp_path / "pods.md"), "pods.md")[0].page_content
    unknown = load_markdown(str(tmp_path / "unknown.md"), "unknown.md")[0].page_content

    assert "(https://kubernetes.io/docs/concepts/workloads/controllers/)" in pods
    assert unknown == "See controllers."  # no sites.json entry -> plain text, never a broken link


# --- load_html: code blocks and sentences must survive HTML cleanup ---------------------------

def write_html(tmp_path, body: str) -> str:
    path = tmp_path / "page.html"
    path.write_text(f"<html><body>{body}</body></html>", encoding="utf-8")
    return str(path)


def test_syntax_highlighted_code_block_stays_intact(tmp_path):
    # The original bug: every highlighted token is its own <span>, and splitting text at every tag
    # turned `services:\n  web:` into "services | : | web | :" one token per line
    html = (
        "<article><p>Create compose.yaml:</p><pre><code>"
        '<span class="k">services</span><span class="p">:</span>\n'
        '  <span class="k">web</span><span class="p">:</span>\n'
        '    <span class="k">image</span><span class="p">:</span> <span class="s">redis:alpine</span>'
        "</code></pre></article>"
    )
    text = load_html(write_html(tmp_path, html), "page.html")[0].page_content

    assert "```\nservices:\n  web:\n    image: redis:alpine\n```" in text


def test_inline_code_and_links_stay_inside_their_sentence(tmp_path):
    html = '<article><p>Run <code>docker compose up</code> and open <a href="/x">the app</a> in a browser.</p></article>'
    text = load_html(write_html(tmp_path, html), "page.html")[0].page_content

    assert "Run docker compose up and open the app in a browser." in text


def test_page_chrome_is_removed(tmp_path):
    html = (
        "<nav>Home | Guides | Reference</nav>"
        '<article><div class="toc">Table of contents: Step 1, Step 2</div>'
        "<button>Copy Markdown</button><script>trackPageView()</script>"
        "<p>The actual documentation text.</p></article>"
        "<footer>Privacy | Terms</footer>"
    )
    text = load_html(write_html(tmp_path, html), "page.html")[0].page_content

    assert text == "The actual documentation text."


def test_article_is_preferred_over_the_rest_of_the_page(tmp_path):
    html = "<main><div>Sidebar promo text</div><article><p>Article body.</p></article></main>"
    assert load_html(write_html(tmp_path, html), "page.html")[0].page_content == "Article body."


def test_page_without_text_returns_no_documents(tmp_path):
    assert load_html(write_html(tmp_path, "<nav>Only navigation</nav><script>x()</script>"), "page.html") == []


# --- near-empty chunk filter + chunk_index numbering ------------------------------------------

@pytest.fixture
def docs_dir(tmp_path, monkeypatch):
    """Point load_and_split_documents() at a temp folder instead of data/raw_docs."""
    monkeypatch.setattr(ingest, "RAW_DOCS_DIR", str(tmp_path))
    return tmp_path


# One paragraph just under CHUNK_SIZE: too long to share a chunk with a short title before it
LONG_PARAGRAPH = ("Pods are the smallest deployable units of computing in Kubernetes. " * 12)[:790].rstrip()
TITLE = "Dockerfile overview\n\nTable of contents"


def test_title_and_table_of_contents_chunk_is_dropped(docs_dir):
    # The real case: "Dockerfile overview\n\nTable of contents" became its own 38-char chunk
    assert len(LONG_PARAGRAPH) <= ingest.CHUNK_SIZE < len(TITLE) + 2 + len(LONG_PARAGRAPH), "test data assumes CHUNK_SIZE=800"
    (docs_dir / "overview.md").write_text(f"{TITLE}\n\n{LONG_PARAGRAPH}", encoding="utf-8")

    chunks = load_and_split_documents()

    assert [c.page_content for c in chunks] == [LONG_PARAGRAPH]


def test_chunk_size_boundary_follows_min_chunk_chars(docs_dir):
    (docs_dir / "too_short.txt").write_text("a" * (MIN_CHUNK_CHARS - 1), encoding="utf-8")
    (docs_dir / "just_enough.txt").write_text("b" * MIN_CHUNK_CHARS, encoding="utf-8")

    sources = [c.metadata["source"] for c in load_and_split_documents()]

    assert sources == ["just_enough.txt"]


def test_short_but_real_code_example_is_kept(docs_dir):
    # The shortest useful chunk in the real corpus: a 53-char Dockerfile example
    (docs_dir / "best_practices.md").write_text("```\n# syntax=docker/dockerfile:1\nFROM alpine:3.21\n```", encoding="utf-8")

    assert len(load_and_split_documents()) == 1


def test_chunk_index_counts_from_zero_per_file_without_gaps(docs_dir):
    # Neighbour expansion fetches chunk_index +-1, so numbering must be consecutive per file,
    # including after the filter drops a chunk (here: the title chunk of overview.md)
    (docs_dir / "overview.md").write_text(f"{TITLE}\n\n" + "\n\n".join([LONG_PARAGRAPH] * 3), encoding="utf-8")
    (docs_dir / "pods.md").write_text("\n\n".join([LONG_PARAGRAPH] * 2), encoding="utf-8")

    chunks = load_and_split_documents()

    by_source = defaultdict(list)
    for chunk in chunks:
        by_source[chunk.metadata["source"]].append(chunk.metadata["chunk_index"])
    assert by_source == {"overview.md": [0, 1, 2], "pods.md": [0, 1]}


def test_chunk_index_follows_reading_order(docs_dir):
    parts = [f"Part {n}. {LONG_PARAGRAPH}" for n in range(3)]
    (docs_dir / "guide.md").write_text("\n\n".join(parts), encoding="utf-8")

    chunks = sorted(load_and_split_documents(), key=lambda c: c.metadata["chunk_index"])

    assert [c.page_content[:7] for c in chunks] == ["Part 0.", "Part 1.", "Part 2."]
