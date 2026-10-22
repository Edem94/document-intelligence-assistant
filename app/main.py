# FastAPI service: health check, demo page, and the /ask endpoint

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import ANTHROPIC_API_KEY, FRONTEND_DIR
from app.rag_chain import ask_question, get_qa_chain

logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the embedding model + Chroma once at startup (get_qa_chain is cached),
    # so the first request isn't slow. Stay up if it fails so /health can report why.
    app.state.startup_error = None
    if not ANTHROPIC_API_KEY:
        app.state.startup_error = "ANTHROPIC_API_KEY is not set"
    else:
        try:
            get_qa_chain()
            logger.info("Vector store and QA chain loaded")
        except Exception as e:
            app.state.startup_error = str(e)
    if app.state.startup_error:
        logger.error("Startup failed: %s", app.state.startup_error)
    yield


app = FastAPI(title="Document Intelligence Assistant", lifespan=lifespan)


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)


class Source(BaseModel):
    file: str
    page: int | None = None


class AnswerResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/health")
def health(request: Request):
    error = request.app.state.startup_error
    if error:
        return JSONResponse(status_code=503, content={"status": "unavailable", "detail": error})
    return {"status": "ok"}


@app.post("/ask", response_model=AnswerResponse)
def ask(body: QuestionRequest, request: Request):
    # Sync def: FastAPI runs it in a threadpool, so the blocking LLM call doesn't stall the event loop
    if request.app.state.startup_error:
        raise HTTPException(status_code=503, detail=request.app.state.startup_error)
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question must not be blank")
    try:
        return ask_question(question)
    except Exception:
        logger.exception("Failed to answer question")
        raise HTTPException(status_code=502, detail="Failed to generate an answer. Check server logs.")


# Static assets (style.css, app.js) are served from /static; the page itself from /
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/", include_in_schema=False)
def root():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
