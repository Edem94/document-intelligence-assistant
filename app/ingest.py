import json
import os
import re
import shutil
from collections import defaultdict
from pypdf import PdfReader
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from app.config import CHROMA_DIR, EMBEDDING_MODEL, RAW_DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_CHARS


def load_pdf(path: str, filename: str) -> list[Document]:
    docs = []
    reader = PdfReader(path)
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            docs.append(Document(page_content=text, metadata={"source": filename, "page": i + 1}))
    if not docs:
        print(f"  WARNING: {filename} has no extractable text (image-only PDF?) - skipped")
    return docs


BLOCK_TAGS = ["p", "div", "section", "li", "tr", "br", "h1", "h2", "h3", "h4", "h5", "h6", "table", "ul", "ol", "dt", "dd"]
# Docs-site chrome that isn't inside <nav>/<header>: tables of contents, breadcrumbs, copy/feedback buttons
CHROME_SELECTOR = "[class*=toc], [id*=toc], [aria-label*=breadcrumb], [role=navigation], button, form, svg"


def load_html(path: str, filename: str) -> list[Document]:
    with open(path, encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f, "html.parser")
    # Remove page chrome so only the actual article text gets embedded
    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()
    for tag in soup.select(CHROME_SELECTOR):
        tag.decompose()
    main = soup.find("article") or soup.find("main") or soup.body or soup

    # Keep code blocks intact: syntax highlighters wrap every token in its own <span>
    for pre in main.find_all("pre"):
        pre.replace_with(f"\n```\n{pre.get_text().strip()}\n```\n")
    # Newlines only between block elements, so inline <code>/<a>/<strong> stay inside their sentence
    for tag in main.find_all(BLOCK_TAGS):
        tag.insert_after("\n")
    text = re.sub(r"\n\s*\n+", "\n\n", main.get_text()).strip()
    return [Document(page_content=text, metadata={"source": filename, "page": 1})] if text else []


def load_text(path: str, filename: str) -> list[Document]:
    with open(path, encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return [Document(page_content=text, metadata={"source": filename, "page": 1})] if text.strip() else []


# [text](url), ![alt](img.png), with an optional "title" - links are resolved before chunking
MD_LINK_RE = re.compile(r'(!?)\[([^\]]*)\]\(\s*<?([^)\s>]*)>?(?:\s+"[^"]*")?\s*\)')


def resolve_markdown_links(text: str, site: str | None) -> str:
    """Make every link in the text one that works for the reader, or drop it to plain text.

    Docs written for a website use links like (/docs/setup/). Copied into an answer as-is, the UI
    resolves them against this app (localhost:8000/docs/setup/ -> 404). Root-relative links get the
    site's domain; repo-relative ones (build-artifacts.md, ../includes/x.md) and #anchors have no
    reliable public URL, so only their text is kept. Images are dropped. Code blocks are left alone.
    """
    def fix(match):
        is_image, label, url = match.groups()
        if is_image:
            return ""
        if url.startswith(("http://", "https://")):
            return match.group(0)
        if site and url.startswith("/") and not url.startswith("//"):
            return f"[{label}]({site.rstrip('/')}{url})"
        return label

    parts = re.split(r"(```.*?```)", text, flags=re.DOTALL)
    return "".join(part if part.startswith("```") else MD_LINK_RE.sub(fix, part) for part in parts)


def load_markdown(path: str, filename: str) -> list[Document]:
    sites_file = os.path.join(os.path.dirname(path), "sites.json")
    sites = {}
    if os.path.exists(sites_file):
        with open(sites_file, encoding="utf-8") as f:
            sites = json.load(f)
    if filename not in sites:
        print(f"  NOTE: {filename} has no entry in sites.json - its root-relative links become plain text")
    docs = load_text(path, filename)
    for doc in docs:
        doc.page_content = resolve_markdown_links(doc.page_content, sites.get(filename))
    return docs


LOADERS = {".pdf": load_pdf, ".html": load_html, ".htm": load_html, ".md": load_markdown, ".txt": load_text}


def load_documents(directory_path: str) -> list[Document]:
    documents = []
    for filename in sorted(os.listdir(directory_path)):
        ext = os.path.splitext(filename)[1].lower()
        loader = LOADERS.get(ext)
        if loader is None:
            continue
        docs = loader(os.path.join(directory_path, filename), filename)
        print(f"  {filename}: {len(docs)} section(s)")
        documents.extend(docs)
    return documents


def load_and_split_documents():
    documents = load_documents(RAW_DOCS_DIR)
    print(f"Loaded {len(documents)} pages/sections")

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(documents)
    # Skip near-empty chunks: a page title + "Table of contents", or a lone "### Options" heading
    chunks = [c for c in chunks if len(c.page_content.strip()) >= MIN_CHUNK_CHARS]
    # Position of each chunk within its file, so retrieval can pull in neighbouring chunks
    # (e.g. the "Step 2: ..." intro right before a YAML example, or the file an example includes)
    counters = defaultdict(int)
    for chunk in chunks:
        source = chunk.metadata["source"]
        chunk.metadata["chunk_index"] = counters[source]
        counters[source] += 1
    print(f"Split into {len(chunks)} chunks")
    return chunks


def build_vector_store():
    chunks = load_and_split_documents()

    # Fail loudly instead of writing an empty, broken collection to disk
    if not chunks:
        raise SystemExit("No text extracted from data/raw_docs - check your files. Vector store NOT built.")

    # Rebuild from scratch each run: avoids duplicate chunks and stale/corrupted collections
    # Without this check a locked folder (server still running on Windows) is silently kept and chunks get duplicated
    try:
        shutil.rmtree(CHROMA_DIR)
    except FileNotFoundError:
        pass
    except PermissionError:
        raise SystemExit(f"Cannot delete {CHROMA_DIR} - stop the running server (uvicorn) first, then re-run ingest.")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    print(f"Vector store built with {len(chunks)} chunks at {CHROMA_DIR}")
    return vector_store


if __name__ == "__main__":
    build_vector_store()
