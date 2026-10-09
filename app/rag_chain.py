## 5. Retrieval + generation chain (the "RAG" part)

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
answers during development and incident response. Answer using ONLY the provided context.

Rules:
- Answer the exact question in your first sentence or two. Do not lead with loosely related material.
- If the context only partly covers the question, say so up front, then give what it does cover.
- If the context doesn't cover it at all, say you don't have enough information. Never guess,
  especially about production behavior.
- Include commands or config from the context when they help, in fenced code blocks with a language tag.
  Copy each example verbatim from ONE source. Never merge examples from different sources or change
  their values - invalid config is worse than none. Prefer the most complete, self-contained example.
- For "how do I define/create/set up" questions, lead with the most complete working example in the context.
- Dependencies: after EACH code example, check whether it (a) uses ${{VARIABLES}}, (b) points to another file
  (include, env_file, extends, a build context with a Dockerfile), or (c) references a service it doesn't define
  (e.g. under depends_on). If so, add a line starting with "**Requires:**" naming each one, and show its
  definition from the context (e.g. the .env file or the included file). If the definition isn't in the
  context, say so instead of inventing it.
- Use Markdown, no top-level heading. Do not list sources at the end - they are shown separately.

Context:
{context}

Question: {question}"""

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
