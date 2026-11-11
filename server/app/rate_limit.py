"""Tiny in-memory rate limiter (sliding one-minute window per client IP).

Good enough for a single-instance demo server; a multi-server deployment would
keep these counters in Redis instead."""
from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import Request


class RateLimiter:
    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self.hits: dict[str, deque] = defaultdict(deque)

    @staticmethod
    def client_ip(request: Request) -> str:
        # Hugging Face Spaces sits behind a proxy, which puts the real IP first in this header.
        fwd = request.headers.get("x-forwarded-for")
        if fwd:
            return fwd.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def check(self, key: str) -> float:
        """Record a hit. Returns 0 if allowed, else seconds until the next request is allowed."""
        if self.per_minute <= 0:
            return 0.0
        now = time.monotonic()
        q = self.hits[key]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= self.per_minute:
            return 60 - (now - q[0])
        q.append(now)
        if len(self.hits) > 10_000:               # forget idle clients so memory can't grow forever
            for k in [k for k, v in self.hits.items() if not v or now - v[-1] > 60]:
                del self.hits[k]
        return 0.0
