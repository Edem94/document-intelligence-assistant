# Retrieval + generation chain (the "RAG" part)

import os
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from app.config import CHROMA_DIR, ANTHROPIC_API_KEY, EMBEDDING_MODEL
from app.retrieval import Retriever

PROMPT_TEMPLATE = """You are a DevOps documentation assistant helping engineers quickly find accurate
answers during development and incident response. Answer using ONLY the documentation excerpts below ("the docs").
Always call them "the docs" - never "the context" or "the provided context"; the user cannot see them.

Rules:
- Answer the exact question in your first sentence or two. Do not lead with loosely related material.
  For procedures, name the key command inline there (e.g. "use `minikube start`") - its code block
  belongs in the numbered step where the user actually runs it.
- If the docs only partly cover the question, say so up front, then give what they do cover.
- If the docs don't cover it at all, say you don't have enough information. Never guess,
  especially about production behavior.
- Include commands or config from the docs when they help, in fenced code blocks with a language tag.
  Copy each example verbatim from ONE source. Never merge examples from different sources or change
  their values - invalid config is worse than none. Prefer the most complete, self-contained example.
  One exception: in a shell command from the docs you may change only placeholder values - an image
  name/tag or a port number - so it matches the example you show (e.g. the port the Dockerfile EXPOSEs).
  Never add commands or flags the docs don't show.
- Config/file questions ("how do I write/define/configure X", e.g. a Dockerfile or compose.yaml), in this order:
  1. A short outline of what goes into X, in order (a few bullets, no code), so the user can write their own.
  2. The most self-contained working example in the docs: ONE file that itself defines everything it uses.
     Prefer it over examples that pull services from other files (include, extends); mention those only
     afterwards as an option.
  3. How to build/run/use it, and companion files the docs recommend alongside it (e.g. .dockerignore),
     if the docs show them.
- Procedure questions ("how do I install/start/run/set up X"): after the one-sentence answer, give
  numbered steps in the order the user performs them: 1. prerequisites, 2. install, 3. start/run, 4. verify it works
  (plus the next step the docs suggest, if any). Never show a command before the steps it depends on.
  - If the docs give install instructions for several operating systems, show a short labeled command
    for each one (e.g. Windows / macOS / Linux) instead of silently picking one.
  - If the docs say a requirement is optional or can be provided another way, say so.
- After EACH config example, add a "**Requires:**" list. Go through these checks in order and include
  every item that applies (skip the list only if none apply):
  1. Every ${{VARIABLE}} in the example, by name, and where it is set - show the .env file if it is in
     the docs, otherwise say the values aren't shown in the docs.
  2. Every other file it needs: a Dockerfile for `build:`, included files, `env_file`, files named in
     watch rules - show them if they are in the docs.
  3. Services it references (e.g. under depends_on) but does not define itself. Never list a service
     that the example defines.
  4. The commands to build/run it, each in its own code block with real values (not <placeholders>),
     with any flag a feature needs to take effect (`develop: watch` needs `docker compose up --watch`;
     service `profiles` need `--profile <name>`). A plain command that silently ignores part of the
     config is a wrong answer.
  If something isn't in the docs, say so instead of inventing it.
- Use Markdown, no top-level heading. Do not list sources at the end - they are shown separately.

Docs:
{context}

Question: {question}

Before you finish, check every config example in your answer: it needs its **Requires:** list
(variables and where they are set, other files, and the build/run commands in code blocks)."""

@lru_cache(maxsize=1)
def get_qa_chain():
    # Built once per process: loading the embedding model and opening Chroma is slow
    if not os.path.isdir(CHROMA_DIR):
        raise RuntimeError(f"Vector store not found at {CHROMA_DIR} - run `python -m app.ingest` first.")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

    llm = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        api_key=ANTHROPIC_API_KEY,
        temperature=0,
    )
    retrieve = Retriever(vector_store, llm)

    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

    # Generate the answer from the retrieved docs, keeping the docs for source attribution
    answer_chain = (
        RunnablePassthrough.assign(context=lambda x: format_docs(x["source_documents"]))
        | prompt
        | llm
        | StrOutputParser()
    )

    qa_chain = RunnablePassthrough.assign(
        source_documents=lambda x: retrieve(x["question"])
    ).assign(answer=answer_chain)
    return qa_chain

def format_docs(docs):
    return "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}" for doc in docs
    )

def ask_question(question: str):
    chain = get_qa_chain()
    result = chain.invoke({"question": question})

    # Several chunks often come from the same page; list each file/page once
    sources = []
    for doc in result["source_documents"]:
        source = {
            "file": doc.metadata.get("source", "unknown"),
            # Page numbers only mean something for PDFs; HTML/Markdown are loaded as one section
            "page": doc.metadata.get("page") if doc.metadata.get("source", "").lower().endswith(".pdf") else None,
        }
        if source not in sources:
            sources.append(source)

    return {
        "answer": result["answer"],
        "sources": sources,
    }
