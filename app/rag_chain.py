## 5. Retrieval + generation chain (the "RAG" part)

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from app.config import CHROMA_DIR, ANTHROPIC_API_KEY, EMBEDDING_MODEL

PROMPT_TEMPLATE = """You are a DevOps documentation assistant 
helping engineers quickly find accurate answers during development and incident response. Answer using ONLY the provided 
context. If the answer isn't in the context, say you don't have enough information —
never guess, especially about production behavior.

Context:
{context}

Question: {question}

Answer clearly and concisely, the way you'd want an answer during an incident. After your
answer, list the source(s) you used."""

def get_qa_chain():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    llm = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        api_key=ANTHROPIC_API_KEY,
        temperature=0,
    )

    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

    # Generate the answer from the retrieved docs, keeping the docs for source attribution
    answer_chain = (
        RunnablePassthrough.assign(context=lambda x: format_docs(x["source_documents"]))
        | prompt
        | llm
        | StrOutputParser()
    )

    qa_chain = RunnablePassthrough.assign(
        source_documents=lambda x: retriever.invoke(x["question"])
    ).assign(answer=answer_chain)
    return qa_chain

def format_docs(docs):
    return "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}" for doc in docs
    )

def ask_question(question: str):
    chain = get_qa_chain()
    result = chain.invoke({"question": question})

    sources = []
    for doc in result["source_documents"]:
        sources.append({
            "file": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", "?"),
        })

    return {
        "answer": result["answer"],
        "sources": sources,
    }
