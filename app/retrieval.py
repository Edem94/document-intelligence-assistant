# Retrieval: query rewriting -> fusion -> LLM rerank -> neighbour expansion

import logging
import re
from collections import defaultdict

from langchain_core.documents import Document

logger = logging.getLogger("uvicorn.error")

# Query rewriting ("code HyDE"): chunks that are mostly YAML/commands embed far from a plain-English
# question, but close to a similar snippet. So we ask the LLM for the snippet the answer would
# contain and search with it alongside the original question.
REWRITE_TEMPLATE = """Write the minimal config file or shell commands that would appear in documentation answering this DevOps question. If it is a config file, also add the shell commands that build/run/use it. Output only the raw snippet, no explanation, no code fences. If the question isn't about config or commands, write one short sentence a doc page would use to answer it.

Question: {question}"""

# Embeddings rank "mentions the same words" above "actually answers it": a complete two-service
# compose.yaml ranked 9th behind CLI reference snippets. An LLM picks the useful excerpts reliably.
RERANK_TEMPLATE = """You are selecting documentation excerpts to answer a DevOps question.

Question: {question}

Candidates:
{candidates}

Pick up to {k} candidates that best help answer the question. Prefer:
- excerpts that directly answer it over loosely related background or command reference tables
- complete, self-contained examples (e.g. a full config file defining every service it uses) over fragments
- for a config file you pick, also the excerpt showing the commands that build/run/use it, if one is listed
Reply with only the candidate numbers, most useful first, comma-separated (e.g. 4, 1, 9)."""

TOP_K = 8             # chunks chosen by the reranker (before neighbour expansion)
FETCH_PER_QUERY = 25  # candidates per query before fusion (15 missed a key chunk ranked 16th)
RERANK_POOL = 30      # fused candidates shown to the reranker (20 dropped good examples once the corpus grew)
MAX_PER_SOURCE = 10   # per file in that pool, so one long page can't fill it (one page had 15 of 30)
MAX_PICKS_PER_SOURCE = 4  # per file among the reranker's TOP_K picks (it once gave 7 of 8 to one page)
PREVIEW_CHARS = 600   # how much of each candidate the reranker sees


class Retriever:
    def __init__(self, vector_store, llm):
        self.vector_store = vector_store
        self.llm = llm.bind(max_tokens=300)

    def __call__(self, question: str) -> list[Document]:
        queries = [question]
        try:
            queries.append(self.llm.invoke(REWRITE_TEMPLATE.format(question=question)).content.strip())
        except Exception:
            logger.warning("Query rewrite failed; searching with the original question only", exc_info=True)
        fused = fuse_results([self.vector_store.similarity_search(q, k=FETCH_PER_QUERY) for q in queries],
                             k=len(queries) * FETCH_PER_QUERY)
        candidates = limit_per_source(fused, MAX_PER_SOURCE, k=RERANK_POOL)
        return expand_neighbors(self.vector_store, self.rerank(question, candidates))

    def rerank(self, question: str, candidates: list[Document]) -> list[Document]:
        listing = "\n\n".join(
            f"[{i}] ({d.metadata.get('source', 'unknown')})\n{d.page_content[:PREVIEW_CHARS]}"
            for i, d in enumerate(candidates, 1)
        )
        try:
            # Ask for a longer ranked list than we keep, so the per-page cap below has others to fall back on
            reply = self.llm.invoke(RERANK_TEMPLATE.format(question=question, candidates=listing, k=2 * TOP_K)).content
        except Exception:
            logger.warning("Rerank failed; using fused order", exc_info=True)
            return limit_per_source(candidates, MAX_PICKS_PER_SOURCE, k=TOP_K)
        picks = [int(n) for n in re.findall(r"\d+", reply) if 1 <= int(n) <= len(candidates)]
        ranked = [candidates[i - 1] for i in dict.fromkeys(picks)] or candidates
        # Neighbour expansion adds the chunks around each pick, so a few picks still cover a whole section
        return limit_per_source(ranked, MAX_PICKS_PER_SOURCE, k=TOP_K)


def limit_per_source(docs: list[Document], max_per_source: int, k: int) -> list[Document]:
    """Keep the ranked order, but at most `max_per_source` chunks from any one file, and `k` in total.

    Long pages (Pod lifecycle, Deployments) produce many similar chunks; without a cap they filled the
    whole candidate pool and the one short note that answered the question never reached the reranker.
    """
    per_source, kept = defaultdict(int), []
    for doc in docs:
        source = doc.metadata.get("source")
        if per_source[source] < max_per_source:
            per_source[source] += 1
            kept.append(doc)
            if len(kept) == k:
                break
    return kept


def fuse_results(results, k):
    """Merge ranked result lists with Reciprocal Rank Fusion (RRF).

    Each query's #1 hit is always kept: the snippet query often finds the one
    chunk with the example (e.g. a compose.yaml) that the question alone ranks low,
    and plain RRF would let the question's many prose matches outvote it.
    """
    scores, docs = defaultdict(float), {}
    for ranked in results:
        for rank, doc in enumerate(ranked):
            scores[doc.page_content] += 1 / (60 + rank)
            docs[doc.page_content] = doc
    picked = list(dict.fromkeys(ranked[0].page_content for ranked in results if ranked))
    picked += [c for c in sorted(scores, key=scores.get, reverse=True) if c not in picked]
    return [docs[c] for c in picked[:k]]


def add_variable_definitions(vector_store, docs: list[Document]) -> dict:
    """For each ${VAR} used in a chunk, find the chunk in the same file that sets it (e.g. a .env with VAR=...).

    Without this the model sees `ports: - "${APP_PORT}:5000"` but not the .env defining APP_PORT,
    which sits a few chunks earlier in the tutorial - and then tells the user the docs don't define it.
    """
    found_defs = {}
    for doc in docs:
        source = doc.metadata["source"]
        for var in set(re.findall(r"\$\{(\w+)", doc.page_content)):
            # "$contains" narrows it down; the regex skips lines like REDIS_HOST=${REDIS_HOST} that only use it
            hits = vector_store.get(where={"source": source}, where_document={"$contains": f"{var}="})
            for text, meta in zip(hits["documents"], hits["metadatas"]):
                if re.search(rf"(?m)^\s*{var}=(?!\$\{{)", text):
                    found_defs[(source, meta["chunk_index"])] = Document(page_content=text, metadata=meta)
                    break
    return found_defs


PLATFORMS = ["Windows", "macOS", "Linux"]


def add_platform_variants(vector_store, docs: list[Document]) -> dict:
    """If a chunk gives install steps for one OS, fetch the install steps for the other OSes from the same file.

    Install pages list every OS one after another (minikube: ~12 chunks of Linux variants before
    macOS and Windows), so retrieval only ever sees the first OS and the answer silently assumes it.
    """
    found, done_sources = {}, set()
    for doc in docs:
        source, text = doc.metadata["source"], doc.page_content
        mentioned = [p for p in PLATFORMS if p in text]
        if source in done_sources or not mentioned or not re.search(r"\binstall", text, re.IGNORECASE):
            continue
        done_sources.add(source)
        for platform in PLATFORMS:
            if platform in mentioned:
                continue
            # The FIRST chunk mentioning the OS + "install": pages lead each OS with the simplest option
            # (a package manager like winget/brew); later chunks are alternatives and PATH fiddling
            hits = vector_store.get(where={"source": source},
                                    where_document={"$and": [{"$contains": platform}, {"$contains": "install"}]})
            if hits["ids"]:
                text, meta = min(zip(hits["documents"], hits["metadatas"]), key=lambda h: h[1]["chunk_index"])
                found[(source, meta["chunk_index"])] = Document(page_content=text, metadata=meta)
    return found


def expand_neighbors(vector_store, docs: list[Document]) -> list[Document]:
    """Add the chunk before and after each selected chunk, then put each file's chunks back in reading order.

    Tutorials split a step across chunks: the intro, the YAML, and the files that YAML depends on
    (.env, an included infra.yaml) end up next to each other, but not in the same chunk.
    """
    if any("chunk_index" not in d.metadata for d in docs):
        logger.warning("Index has no chunk_index (built by an older ingest) - re-run `python -m app.ingest`")
        return docs
    chosen = {(d.metadata["source"], d.metadata["chunk_index"]): d for d in docs}
    for source, index in list(chosen):
        found = vector_store.get(where={"$and": [{"source": source}, {"chunk_index": {"$in": [index - 1, index + 1]}}]})
        for text, meta in zip(found["documents"], found["metadatas"]):
            chosen.setdefault((meta["source"], meta["chunk_index"]), Document(page_content=text, metadata=meta))
    for key, doc in add_variable_definitions(vector_store, list(chosen.values())).items():
        chosen.setdefault(key, doc)
    for key, doc in add_platform_variants(vector_store, list(chosen.values())).items():
        chosen.setdefault(key, doc)
    # Files in order of their best-ranked chunk; chunks within a file in document order
    file_rank = {}
    for d in docs:
        file_rank.setdefault(d.metadata["source"], len(file_rank))
    return [chosen[key] for key in sorted(chosen, key=lambda key: (file_rank[key[0]], key[1]))]
