# Rate limiting for /ask: every question is a paid Claude API call, and the service is public.

import threading
import time
from collections import defaultdict, deque

from fastapi import Request

from app.config import TRUSTED_PROXY_HOPS


class RateLimiter:
    """Sliding window: at most `limit` requests per `window` seconds for each key.

    In memory, so each server instance counts separately (Cloud Run runs at most 2 of them).
    Thread-safe, because FastAPI runs the sync /ask endpoint in a threadpool.
    """

    MAX_KEYS = 10_000  # forget idle clients once this many are tracked, so memory stays bounded

    def __init__(self, limit: int, window: float, clock=time.monotonic):
        self.limit = limit
        self.window = window
        self.clock = clock
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def hit(self, key: str) -> float:
        """Count a request for `key`. Returns 0 if allowed, else the seconds until it would be."""
        now = self.clock()
        with self._lock:
            hits = self._hits[key]
            while hits and hits[0] <= now - self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                return hits[0] + self.window - now
            hits.append(now)
            if len(self._hits) > self.MAX_KEYS:
                self._forget_idle(now)
            return 0.0

    def reset(self):
        with self._lock:
            self._hits.clear()

    def _forget_idle(self, now: float):
        for key in [k for k, hits in self._hits.items() if not hits or hits[-1] <= now - self.window]:
            del self._hits[key]


def client_ip(request: Request) -> str:
    """The visitor's IP address, as reported by a proxy we trust.

    Behind Cloud Run every request arrives from Google's front end, so the socket address is the
    same for everyone. The proxy appends the real client IP to X-Forwarded-For; entries to the
    LEFT of the ones our proxies added are written by the client and can be faked, so we count
    TRUSTED_PROXY_HOPS entries from the right. With 0 hops (local runs) the socket address is used.
    """
    if TRUSTED_PROXY_HOPS > 0:
        forwarded = [h.strip() for h in request.headers.get("x-forwarded-for", "").split(",") if h.strip()]
        if len(forwarded) >= TRUSTED_PROXY_HOPS:
            return forwarded[-TRUSTED_PROXY_HOPS]
    return request.client.host if request.client else "unknown"
