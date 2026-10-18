from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def test_chunking_respects_size_limit():
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    long_text = "Kubernetes pod scheduling. " * 200
    doc = Document(page_content=long_text, metadata={"source": "k8s.pdf", "page": 1})

    chunks = splitter.split_documents([doc])

    assert len(chunks) > 1
    assert all(len(c.page_content) <= 850 for c in chunks)  # small buffer over 800
    assert all(c.metadata["source"] == "k8s.pdf" for c in chunks)  # metadata preserved
