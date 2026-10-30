"""Login throttling: a sliding window of failed attempts per (email, client address).

In-process state is enough for the single-worker deployment in docker-compose; a multi-worker
deployment would move this to Redis or a table. After MAX_FAILURES failures inside WINDOW_S
seconds the pair is locked until the oldest failure leaves the window (HTTP 429 + Retry-After).
A successful login clears the pair.
"""
import threading
import time
from collections import defaultdict, deque

MAX_FAILURES = 5
WINDOW_S = 300.0

_lock = threading.Lock()
_fail: dict[tuple[str, str], deque] = defaultdict(deque)


def _prune(q: deque, now: float) -> None:
    while q and now - q[0] > WINDOW_S:
        q.popleft()


def retry_after(key: tuple[str, str], now: float | None = None) -> int:
    """Seconds until another attempt is allowed; 0 if not locked."""
    now = time.monotonic() if now is None else now
    with _lock:
        q = _fail.get(key)
        if not q:
            return 0
        _prune(q, now)
        return int(WINDOW_S - (now - q[0])) + 1 if len(q) >= MAX_FAILURES else 0


def failure(key: tuple[str, str], now: float | None = None) -> None:
    now = time.monotonic() if now is None else now
    with _lock:
        q = _fail[key]
        _prune(q, now)
        q.append(now)


def success(key: tuple[str, str]) -> None:
    with _lock:
        _fail.pop(key, None)


def reset() -> None:
    with _lock:
        _fail.clear()
