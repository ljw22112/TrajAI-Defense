"""Threat alerts: in-process fan-out to connected consoles (WebSocket) when an estimate reaches level >= ALERT_LEVEL.

One process per deployment (docker-compose runs a single worker), so a set of asyncio queues is enough; a
multi-worker deployment would publish through PostgreSQL LISTEN/NOTIFY or Redis instead.
"""
import asyncio

ALERT_LEVEL = 4
_subscribers: set[asyncio.Queue] = set()


def subscribe() -> asyncio.Queue:
    q: asyncio.Queue = asyncio.Queue(maxsize=50)
    _subscribers.add(q)
    return q


def unsubscribe(q: asyncio.Queue) -> None:
    _subscribers.discard(q)


def publish(msg: dict) -> int:
    n = 0
    for q in list(_subscribers):
        try:
            q.put_nowait(msg); n += 1
        except asyncio.QueueFull:          # a stalled client must not block the API
            pass
    return n
