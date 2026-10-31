from langchain_core.documents import Document

from app.retrieval import add_build_files

FENCE = "```"


class FakeStore:
    """Answers the two vector_store.get() shapes add_build_files uses - no embeddings needed."""

    def __init__(self, source: str, chunks: list[str]):
        self.chunks = [(text, {"source": source, "chunk_index": i}) for i, text in enumerate(chunks)]

    def get(self, where, where_document=None):
        if where_document:  # {"source": s} + {"$contains": text}, case-insensitive like Chroma
            needle = where_document["$contains"].lower()
            hits = [c for c in self.chunks if c[1]["source"] == where["source"] and needle in c[0].lower()]
        else:               # {"$and": [{"source": s}, {"chunk_index": {"$in": [...]}}]}
            source, indexes = where["$and"][0]["source"], where["$and"][1]["chunk_index"]["$in"]
            hits = [c for c in self.chunks if c[1]["source"] == source and c[1]["chunk_index"] in indexes]
        return {"ids": [str(m["chunk_index"]) for _, m in hits],
                "documents": [t for t, _ in hits], "metadatas": [m for _, m in hits]}


# The Compose quickstart's layout: app files and Dockerfile in step 1, compose.yaml much later
QUICKSTART = [
    "Create a directory for the project.",
    f"Create app.py:\n{FENCE}\nimport redis\nfrom flask import Flask\n{FENCE}",
    f"Create requirements.txt:\n{FENCE}\nflask\nredis\n{FENCE}",
    f"Create a Dockerfile:\n{FENCE}\nFROM python:3.12-alpine\nCOPY requirements.txt .",
    f"COPY . .\nCMD [\"flask\", \"run\", \"--debug\"]\n{FENCE}",
    "Docker sends everything in your project directory to the daemon.",
    f"Create compose.yaml:\n{FENCE}\nservices:\n  web:\n    build: .\n{FENCE}",
]


def chunk(store, index):
    text, meta = store.chunks[index]
    return Document(page_content=text, metadata=meta)


def test_compose_build_pulls_in_the_dockerfile_and_the_files_it_copies():
    store = FakeStore("quickstart.html", QUICKSTART)
    found = add_build_files(store, [chunk(store, 6)])
    indexes = sorted(i for _, i in found)
    # 2 = requirements.txt, 3 + 4 = the whole Dockerfile (split across chunks). Chunk 1 is
    # "from flask import" - must not be taken for the Dockerfile's FROM line
    assert {2, 3, 4} <= set(indexes)
    assert 1 not in indexes


def test_chunks_without_build_add_nothing():
    store = FakeStore("quickstart.html", QUICKSTART)
    assert add_build_files(store, [chunk(store, 5)]) == {}
