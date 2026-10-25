import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from app.main import RATE_LIMITS, app
from app.rag_chain import build_answer_chain, stream_answer

DOCS = [
    Document(page_content="A Pod persists until it is deleted.", metadata={"source": "kubernetes_pods.md", "page": 1}),
    Document(page_content="Controllers replace Pods.", metadata={"source": "kubernetes_pods.md", "page": 1}),
    Document(page_content="restartPolicy: Always", metadata={"source": "k8s_pod_lifecycle.html", "page": 1}),
]


def fake_llm(text: str):
    # Streams word by word, like a real model streams tokens - no API call
    return GenericFakeChatModel(messages=iter([AIMessage(content=text)]))


# --- the real answer chain (prompt -> LLM -> parser) ------------------------------------------

def test_answer_chain_streams_in_pieces_not_all_at_once():
    chain = build_answer_chain(fake_llm("Pods are replaced by their controller, not restarted."))

    pieces = list(chain.stream({"question": "How do I restart a Pod?", "source_documents": DOCS}))

    assert len(pieces) > 5
    assert "".join(pieces) == "Pods are replaced by their controller, not restarted."


def test_stream_answer_sends_sources_first_then_text():
    retrieve = lambda question: DOCS
    with patch("app.rag_chain.get_pipeline", return_value=(retrieve, build_answer_chain(fake_llm("Delete the Pod.")))):
        events = list(stream_answer("How do I restart a Pod?"))

    kind, sources = events[0]
    assert kind == "sources"
    # Deduplicated per file, in retrieval order
    assert sources == [{"file": "kubernetes_pods.md", "page": None}, {"file": "k8s_pod_lifecycle.html", "page": None}]
    assert all(kind == "token" for kind, _ in events[1:])
    assert "".join(text for _, text in events[1:]) == "Delete the Pod."


# --- /ask/stream endpoint ---------------------------------------------------------------------

@pytest.fixture
def client():
    with patch("app.main.ANTHROPIC_API_KEY", "test-key"), patch("app.main.get_qa_chain"):
        with TestClient(app) as test_client:
            yield test_client


def read_events(response):
    return [json.loads(line) for line in response.text.splitlines() if line.strip()]


def fake_stream(question):
    yield "sources", [{"file": "kubernetes_pods.md", "page": None}]
    for word in ["A ", "Pod ", "persists."]:
        yield "token", word


@patch("app.main.stream_answer", side_effect=fake_stream)
def test_stream_endpoint_sends_ndjson_events_in_order(mock_stream, client):
    response = client.post("/ask/stream", json={"question": "How do I restart a Pod?"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    events = read_events(response)
    assert events[0] == {"type": "sources", "sources": [{"file": "kubernetes_pods.md", "page": None}]}
    assert [e["text"] for e in events[1:-1]] == ["A ", "Pod ", "persists."]
    assert events[-1] == {"type": "done"}
    mock_stream.assert_called_once_with("How do I restart a Pod?")


def failing_stream(question):
    yield "sources", []
    yield "token", "Partial "
    raise RuntimeError("Your credit balance is too low")


@patch("app.main.stream_answer", side_effect=failing_stream)
def test_error_mid_stream_becomes_an_error_event_without_leaking_details(mock_stream, client):
    response = client.post("/ask/stream", json={"question": "How do I restart a Pod?"})

    events = read_events(response)
    assert events[-1] == {"type": "error", "detail": "Failed to generate an answer. Please try again."}
    assert "credit" not in response.text  # internal errors stay in the server log
    assert {"type": "done"} not in events


@pytest.mark.parametrize("question, status", [("   ", 422), ("hi", 422)])
def test_stream_endpoint_rejects_bad_questions_before_streaming(client, question, status):
    response = client.post("/ask/stream", json={"question": question})
    assert response.status_code == status
    assert response.headers["content-type"].startswith("application/json")


@patch("app.main.stream_answer", side_effect=fake_stream)
def test_stream_endpoint_is_rate_limited(mock_stream, client, monkeypatch):
    monkeypatch.setattr(RATE_LIMITS[0][0], "limit", 1)
    assert client.post("/ask/stream", json={"question": "What is a Pod?"}).status_code == 200
    response = client.post("/ask/stream", json={"question": "What is a Pod?"})
    assert response.status_code == 429
    assert "Retry-After" in response.headers


@patch("app.main.stream_answer", side_effect=fake_stream)
def test_ask_and_ask_stream_share_one_limit(mock_stream, client, monkeypatch):
    # Otherwise switching endpoints would double a client's allowance
    monkeypatch.setattr(RATE_LIMITS[0][0], "limit", 1)
    with patch("app.main.ask_question", return_value={"answer": "ok", "sources": []}):
        assert client.post("/ask", json={"question": "What is a Pod?"}).status_code == 200
    assert client.post("/ask/stream", json={"question": "What is a Pod?"}).status_code == 429
