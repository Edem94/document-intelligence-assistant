import pytest

from app.main import RATE_LIMITS


@pytest.fixture(autouse=True)
def reset_rate_limits():
    # The limiters are module-level, so counts would leak from one test into the next
    for limiter, _, _ in RATE_LIMITS:
        limiter.reset()
    yield
