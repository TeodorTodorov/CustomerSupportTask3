"""Small in-process security controls; deploy behind a real edge gateway.

Never log query bodies or API credentials. Do not trust X-Forwarded-For unless a
separately configured trusted reverse proxy has already sanitized it.
"""
from __future__ import annotations

import hmac
import os
import threading
import time
import unicodedata
from collections import OrderedDict, deque
from dataclasses import dataclass

from .inference import InvalidQuery


@dataclass(frozen=True, slots=True)
class SecuritySettings:
    api_key: str | None = None
    max_body_bytes: int = 32768
    max_batch_size: int = 32
    max_chars: int = 4000
    requests_per_minute: int = 60
    max_concurrent_inference: int = 2

    @classmethod
    def from_environment(cls) -> SecuritySettings:
        key = os.getenv("KLETT_API_KEY")
        if not key or len(key) < 24:
            raise RuntimeError("Set KLETT_API_KEY (at least 24 characters) before serving HTTP")
        return cls(api_key=key)

    def __post_init__(self):
        if self.api_key is not None and len(self.api_key) < 24:
            raise ValueError("API key must contain at least 24 characters")
        if self.max_body_bytes < 256 or not (1 <= self.max_batch_size <= 256):
            raise ValueError("Invalid request size limits")
        if self.max_chars < 1 or self.max_batch_size * self.max_chars * 4 > 100000000:
            raise ValueError("Invalid maximum query length")
        if self.requests_per_minute <= 0 or self.max_concurrent_inference <= 0:
            raise ValueError("Invalid resource limits")


BANNED_CONTROLS = {"\x00", "\u202e", "\u202d", "\u202a", "\u202b", "\u202c", "\u2066", "\u2067", "\u2068", "\u2069"}


def validate_query(text: object, max_chars: int = 4000) -> str:
    if not isinstance(text, str):
        raise InvalidQuery("Query must be a string")
    cleaned = text.strip()
    if not cleaned:
        raise InvalidQuery("Query must not be empty")
    if len(cleaned) > max_chars:
        raise InvalidQuery(f"Query exceeds {max_chars} characters")
    if any(ch in BANNED_CONTROLS or (unicodedata.category(ch) == "Cc" and ch not in "\n\r\t") for ch in cleaned):
        raise InvalidQuery("Query contains unsupported control characters")
    return cleaned


def is_authorized(header: str | None, secret: str | None) -> bool:
    if secret is None:
        return False
    if not header or not header.startswith("Bearer "):
        return False
    presented = header.removeprefix("Bearer ")
    return hmac.compare_digest(presented.encode("utf-8"), secret.encode("utf-8"))


class SlidingWindowLimiter:
    """Bounded-memory per-process limiter. Use gateway/Redis for multiple replicas."""

    def __init__(self, limit: int, max_clients: int = 4096):
        self.limit = limit
        self.max_clients = max_clients
        self._entries: OrderedDict[str, deque[float]] = OrderedDict()
        self._lock = threading.Lock()

    def allow(self, identity: str) -> bool:
        now = time.monotonic()
        with self._lock:
            bucket = self._entries.get(identity)
            if bucket is None:
                if len(self._entries) >= self.max_clients:
                    self._entries.popitem(last=False)
                bucket = deque()
                self._entries[identity] = bucket
            self._entries.move_to_end(identity)
            while bucket and now - bucket[0] >= 60:
                bucket.popleft()
            if len(bucket) >= self.limit:
                return False
            bucket.append(now)
            return True


class RequestBodyLimitMiddleware:
    """Pure ASGI body limit, enforced even when Content-Length is absent/false."""

    def __init__(self, app, max_bytes: int = 32768):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        if scope.get("method") in {"POST", "PUT", "PATCH"} and scope.get("path", "").startswith("/v1/"):
            media = headers.get(b"content-type", b"").split(b";", 1)[0].strip().lower()
            if media != b"application/json":
                await self._reject(send, 415, b'{"detail":"Content-Type must be application/json"}')
                return
        raw_length = headers.get(b"content-length", b"")
        if raw_length:
            try:
                declared = int(raw_length)
            except ValueError:
                declared = -1
            if declared < 0:
                await self._reject(send, 400, b'{"detail":"Invalid Content-Length"}')
                return
            if declared > self.max_bytes:
                await self._reject(send, 413, b'{"detail":"Request body too large"}')
                return

        # Buffer at most the configured limit BEFORE entering FastAPI. This
        # handles chunked uploads and guarantees that over-limit requests get
        # a 413 rather than a framework-specific ClientDisconnect exception.
        body = bytearray()
        while True:
            event = await receive()
            if event["type"] == "http.disconnect":
                return
            if event["type"] != "http.request":
                continue
            body.extend(event.get("body", b""))
            if len(body) > self.max_bytes:
                await self._reject(send, 413, b'{"detail":"Request body too large"}')
                return
            if not event.get("more_body", False):
                break
        delivered = False

        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return {"type": "http.disconnect"}

        async def send_headers(message):
            if message["type"] == "http.response.start":
                message["headers"].extend([
                    (b"x-content-type-options", b"nosniff"),
                    (b"cache-control", b"no-store"),
                    (b"referrer-policy", b"no-referrer"),
                    (b"x-frame-options", b"DENY"),
                ])
            await send(message)

        await self.app(scope, replay, send_headers)

    @staticmethod
    async def _reject(send, status, body):
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", b"application/json"), (b"cache-control", b"no-store")]})
        await send({"type": "http.response.body", "body": body})
