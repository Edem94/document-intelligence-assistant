from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    # `with TestClient(app)` runs the app's lifespan (startup), which sets app.state.startup_error.
    # A bare TestClient(app) skips it -> AttributeError: 'State' object has no attribute 'startup_error'.
    # The real startup loads the embedding model + Chroma and needs an API key, so stub it out.
    with patch("app.main.ANTHROPIC_API_KEY", "test-key"), patch("app.main.get_qa_chain"):
        with TestClient(app) as test_client:
            yield test_client


@pytest.fixture
def client_without_api_key():
    with patch("app.main.ANTHROPIC_API_KEY", None), patch("app.main.get_qa_chain"):
        with TestClient(app) as test_client:
            yield test_client


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@patch("app.main.ask_question")
def test_ask_endpoint_returns_answer_and_sources(mock_ask, client):
    mock_ask.return_value = {
        "answer": "HPA scales based on CPU utilization.",
        "sources": [{"file": "k8s-hpa.pdf", "page": 3}],
    }

    response = client.post("/ask", json={"question": "How does HPA scale pods?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "HPA scales based on CPU utilization."
    assert body["sources"][0]["file"] == "k8s-hpa.pdf"
    mock_ask.assert_called_once_with("How does HPA scale pods?")


@pytest.mark.parametrize("question", ["", "hi", "   "])
def test_ask_endpoint_rejects_empty_or_too_short_question(client, question):
    # "" and "hi" fail the min_length=3 validation; "   " passes it but is blank after strip()
    response = client.post("/ask", json={"question": question})
    assert response.status_code == 422


@patch("app.main.ask_question", side_effect=RuntimeError("LLM is down"))
def test_ask_endpoint_returns_502_when_answering_fails(mock_ask, client):
    response = client.post("/ask", json={"question": "How does HPA scale pods?"})
    assert response.status_code == 502


def test_health_and_ask_return_503_when_startup_failed(client_without_api_key):
    health = client_without_api_key.get("/health")
    assert health.status_code == 503
    assert "ANTHROPIC_API_KEY" in health.json()["detail"]

    ask = client_without_api_key.post("/ask", json={"question": "How does HPA scale pods?"})
    assert ask.status_code == 503
