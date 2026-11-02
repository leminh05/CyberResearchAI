"""HTTP client with timeouts, rate limiting, retries and exponential backoff.

All three API clients share this class, so the retry policy is implemented
(and tested) in exactly one place.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any, Callable

import requests

from cyberresearch.config import HttpSettings

logger = logging.getLogger(__name__)

# 429 = rate limited; 5xx = temporary server problems. Other 4xx are our fault
# (bad query, not found) and retrying would not help.
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class HttpError(Exception):
    """Raised when a request fails permanently or retries are exhausted."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class RateLimiter:
    """Guarantees a minimum interval between consecutive requests."""

    def __init__(
        self,
        min_interval_seconds: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.min_interval = min_interval_seconds
        self._clock = clock
        self._sleep = sleep
        self._last_request: float | None = None

    def wait(self) -> None:
        """Sleep just long enough to respect the minimum interval."""
        now = self._clock()
        if self._last_request is not None:
            remaining = self.min_interval - (now - self._last_request)
            if remaining > 0:
                self._sleep(remaining)
                now = self._clock()
        self._last_request = now


class HttpClient:
    """Thin wrapper around ``requests.Session.get`` with a retry policy.

    Args:
        source: Name used in log messages (e.g. ``"arxiv"``).
        min_interval_seconds: Rate limit, minimum seconds between requests.
        settings: Timeout and retry configuration.
        headers: Extra headers (User-Agent, API key headers).
        session: Injected for tests; defaults to a real ``requests.Session``.
        sleep / clock / jitter: Injected for fast, deterministic tests.
    """

    def __init__(
        self,
        source: str,
        min_interval_seconds: float,
        settings: HttpSettings,
        headers: dict[str, str] | None = None,
        session: Any = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        self.source = source
        self.settings = settings
        self.session = session or requests.Session()
        self.headers = headers or {}
        self._sleep = sleep
        self._jitter = jitter
        self.rate_limiter = RateLimiter(min_interval_seconds, clock=clock, sleep=sleep)
        self.request_count = 0
        self.retry_count = 0

    def backoff_seconds(self, attempt: int, retry_after: str | None = None) -> float:
        """Exponential backoff: base * 2^attempt (+ up to 1s jitter), capped.

        If the server sent ``Retry-After`` (seconds), wait at least that long.
        """
        delay = min(self.settings.backoff_max_seconds, self.settings.backoff_base_seconds * (2**attempt))
        delay += self._jitter()
        if retry_after:
            try:
                delay = max(delay, float(retry_after))
            except ValueError:
                pass  # HTTP-date form is rare for these APIs; ignore it
        return min(delay, self.settings.backoff_max_seconds + 1.0)

    def get(self, url: str, params: dict[str, Any] | None = None) -> requests.Response:
        """GET ``url`` and return a successful response, retrying when sensible.

        Raises:
            HttpError: on a non-retryable status or after ``max_retries``.
        """
        max_retries = self.settings.max_retries
        for attempt in range(max_retries + 1):
            self.rate_limiter.wait()
            self.request_count += 1
            reason: str
            retry_after: str | None = None
            try:
                response = self.session.get(
                    url, params=params, headers=self.headers, timeout=self.settings.timeout_seconds
                )
            except (requests.Timeout, requests.ConnectionError) as exc:
                reason = f"{type(exc).__name__}"
                status: int | None = None
            else:
                status = response.status_code
                if status < 400:
                    return response
                if status not in RETRYABLE_STATUS:
                    raise HttpError(f"{self.source}: HTTP {status} (not retryable)", status)
                reason = f"HTTP {status}"
                retry_after = response.headers.get("Retry-After")

            if attempt == max_retries:
                raise HttpError(f"{self.source}: giving up after {attempt + 1} attempts ({reason})", status)
            delay = self.backoff_seconds(attempt, retry_after)
            self.retry_count += 1
            logger.warning(
                "request failed, retrying",
                extra={"source": self.source, "reason": reason, "attempt": attempt + 1, "delay_s": round(delay, 2)},
            )
            self._sleep(delay)
        raise AssertionError("unreachable")  # pragma: no cover
