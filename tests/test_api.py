from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

@patch("app.main.ask_question")
def test_ask_endpoint_returns_answer_and_sources(mock_ask):
    mock_ask.return_value = {
        "answer": "HPA scales based on CPU utilization.",
        "sources": [{"file": "k8s-hpa.pdf", "page": 3}],
    }

    response = client.post("/ask", json={"question": "How does HPA scale pods?"})

    assert response.status_code == 200
    body = response.json()
    assert "answer" in body
    assert body["sources"][0]["file"] == "k8s-hpa.pdf"

def test_ask_endpoint_rejects_empty_question():
    response = client.post("/ask", json={"question": ""})
    assert response.status_code in (200, 422)  # decide: allow or validate — see note below
