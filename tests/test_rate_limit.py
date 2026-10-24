from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

import app.rate_limit as rate_limit
from app.main import RATE_LIMITS, app
from app.rate_limit import RateLimiter, client_ip


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


# --- RateLimiter -------------------------------------------------------------------------------

def test_allows_up_to_the_limit_then_rejects():
    limiter = RateLimiter(limit=3, window=60, clock=FakeClock())
    assert [limiter.hit("a") for _ in range(3)] == [0, 0, 0]
    assert limiter.hit("a") > 0


def test_retry_after_is_time_until_oldest_request_leaves_the_window():
    clock = FakeClock()
    limiter = RateLimiter(limit=2, window=60, clock=clock)
    limiter.hit("a")             # t=1000
    clock.now += 10
    limiter.hit("a")             # t=1010
    clock.now += 5               # t=1015
    assert limiter.hit("a") == pytest.approx(45)   # the t=1000 request expires at t=1060


def test_window_slides_so_old_requests_stop_counting():
    clock = FakeClock()
    limiter = RateLimiter(limit=2, window=60, clock=clock)
    limiter.hit("a"); limiter.hit("a")
    assert limiter.hit("a") > 0
    clock.now += 60
    assert limiter.hit("a") == 0


def test_rejected_requests_are_not_counted():
    # Otherwise a client retrying while blocked would extend its own block forever
    clock = FakeClock()
    limiter = RateLimiter(limit=1, window=60, clock=clock)
    limiter.hit("a")
    for _ in range(5):
        clock.now += 10
        assert limiter.hit("a") > 0
    clock.now += 10              # 60s after the only counted request
    assert limiter.hit("a") == 0


def test_each_key_has_its_own_limit():
    limiter = RateLimiter(limit=1, window=60, clock=FakeClock())
    assert limiter.hit("1.1.1.1") == 0
    assert limiter.hit("1.1.1.1") > 0
    assert limiter.hit("2.2.2.2") == 0


def test_idle_keys_are_forgotten_so_memory_stays_bounded():
    clock = FakeClock()
    limiter = RateLimiter(limit=5, window=60, clock=clock)
    limiter.MAX_KEYS = 3
    for ip in ["a", "b", "c"]:
        limiter.hit(ip)
    clock.now += 61              # a, b, c are now idle
    limiter.hit("d"); limiter.hit("e")
    assert set(limiter._hits) == {"d", "e"}


# --- client_ip: who is the visitor? ------------------------------------------------------------

def make_request(socket_ip="10.0.0.1", forwarded_for=None) -> Request:
    headers = [(b"x-forwarded-for", forwarded_for.encode())] if forwarded_for else []
    return Request({"type": "http", "headers": headers, "client": (socket_ip, 1234)})


def test_without_proxy_the_socket_address_is_used_and_header_ignored(monkeypatch):
    monkeypatch.setattr(rate_limit, "TRUSTED_PROXY_HOPS", 0)
    # Locally anyone can send X-Forwarded-For, so it must not be trusted
    assert client_ip(make_request("10.0.0.1", forwarded_for="6.6.6.6")) == "10.0.0.1"


def test_behind_one_proxy_the_entry_it_appended_is_used(monkeypatch):
    monkeypatch.setattr(rate_limit, "TRUSTED_PROXY_HOPS", 1)
    assert client_ip(make_request("169.254.1.1", forwarded_for="203.0.113.7")) == "203.0.113.7"


def test_spoofed_entries_from_the_client_are_ignored(monkeypatch):
    monkeypatch.setattr(rate_limit, "TRUSTED_PROXY_HOPS", 1)
    # The client sent "X-Forwarded-For: 6.6.6.6"; the proxy appended the real address after it
    assert client_ip(make_request(forwarded_for="6.6.6.6, 203.0.113.7")) == "203.0.113.7"


def test_falls_back_to_socket_address_when_header_is_missing(monkeypatch):
    monkeypatch.setattr(rate_limit, "TRUSTED_PROXY_HOPS", 1)
    assert client_ip(make_request("10.0.0.1")) == "10.0.0.1"


# --- /ask endpoint -----------------------------------------------------------------------------

@pytest.fixture
def client():
    with patch("app.main.ANTHROPIC_API_KEY", "test-key"), patch("app.main.get_qa_chain"), \
            patch("app.main.ask_question", return_value={"answer": "ok", "sources": []}):
        with TestClient(app) as test_client:
            yield test_client


@pytest.fixture
def per_minute_limit(monkeypatch):
    def set_limit(n):
        monkeypatch.setattr(RATE_LIMITS[0][0], "limit", n)
    return set_limit


def test_ask_returns_429_with_retry_after_when_limit_is_exceeded(client, per_minute_limit):
    per_minute_limit(2)
    for _ in range(2):
        assert client.post("/ask", json={"question": "What is a Pod?"}).status_code == 200

    response = client.post("/ask", json={"question": "What is a Pod?"})

    assert response.status_code == 429
    assert 0 < int(response.headers["Retry-After"]) <= 60
    assert response.json()["detail"].startswith("Too many questions in a minute - please try again in")


def test_blank_questions_count_too(client, per_minute_limit):
    # The limit is checked before any validation, so junk requests can't be used to flood the API
    per_minute_limit(1)
    assert client.post("/ask", json={"question": "   "}).status_code == 422
    assert client.post("/ask", json={"question": "What is a Pod?"}).status_code == 429


def test_health_and_ui_are_not_rate_limited(client, per_minute_limit):
    per_minute_limit(1)
    client.post("/ask", json={"question": "What is a Pod?"})
    assert client.post("/ask", json={"question": "What is a Pod?"}).status_code == 429
    assert client.get("/health").status_code == 200
    assert client.get("/").status_code == 200
