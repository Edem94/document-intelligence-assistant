import os
import shutil
from pypdf import PdfReader
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from app.config import CHROMA_DIR, EMBEDDING_MODEL, RAW_DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP


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


def load_html(path: str, filename: str) -> list[Document]:
    with open(path, encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f, "html.parser")
    # Remove page chrome so only the actual article text gets embedded
    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()
    main = soup.find("main") or soup.find("article") or soup.body or soup
    text = main.get_text(separator="\n", strip=True)
    return [Document(page_content=text, metadata={"source": filename, "page": 1})] if text else []


def load_text(path: str, filename: str) -> list[Document]:
    with open(path, encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return [Document(page_content=text, metadata={"source": filename, "page": 1})] if text.strip() else []


LOADERS = {".pdf": load_pdf, ".html": load_html, ".htm": load_html, ".md": load_text, ".txt": load_text}


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
    print(f"Split into {len(chunks)} chunks")
    return chunks


def build_vector_store():
    chunks = load_and_split_documents()

    # Fail loudly instead of writing an empty, broken collection to disk
    if not chunks:
        raise SystemExit("No text extracted from data/raw_docs - check your files. Vector store NOT built.")

    # Rebuild from scratch each run: avoids duplicate chunks and stale/corrupted collections
    shutil.rmtree(CHROMA_DIR, ignore_errors=True)

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
