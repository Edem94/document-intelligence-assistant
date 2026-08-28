import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.config import CHROMA_DIR, RAW_DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP

def load_and_split_documents():
    loader = PyPDFDirectoryLoader(RAW_DOCS_DIR)
    documents = loader.load()  # each doc keeps metadata: source filename + page number
    print(f"Loaded {len(documents)} pages")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks

def build_vector_store():
    chunks = load_and_split_documents()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    vector_store.persist()
    print(f"Vector store built with {len(chunks)} chunks at {CHROMA_DIR}")
    return vector_store

if __name__ == "__main__":
    build_vector_store()
