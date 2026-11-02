"""Retry, backoff, timeout and rate-limit behaviour of HttpClient."""

import logging

import pytest
import requests

from cyberresearch.ingestion.http import HttpError, RateLimiter
from fakes import FakeResponse, FakeSession, fake_http


def test_success_on_first_try_passes_timeout() -> None:
    session = FakeSession(FakeResponse(200, "ok"))
    http = fake_http(session)
    assert http.get("https://x", {"q": 1}).text == "ok"
    assert session.calls[0]["timeout"] == 5
    assert http.retry_count == 0


def test_retries_server_error_with_exponential_backoff() -> None:
    sleeps: list[float] = []
    session = FakeSession(FakeResponse(503), FakeResponse(502), FakeResponse(200, "ok"))
    http = fake_http(session, sleeps=sleeps)
    assert http.get("https://x").text == "ok"
    assert len(session.calls) == 3
    assert sleeps == [1.0, 2.0]  # base * 2^0, base * 2^1 (jitter disabled)


def test_rate_limited_429_honours_retry_after() -> None:
    sleeps: list[float] = []
    session = FakeSession(FakeResponse(429, headers={"Retry-After": "7"}), FakeResponse(200, "ok"))
    fake_http(session, sleeps=sleeps).get("https://x")
    assert sleeps == [7.0]


def test_timeouts_and_connection_errors_are_retried() -> None:
    session = FakeSession(requests.Timeout(), requests.ConnectionError(), FakeResponse(200, "ok"))
    http = fake_http(session)
    assert http.get("https://x").text == "ok"
    assert http.retry_count == 2


def test_gives_up_after_max_retries() -> None:
    session = FakeSession(*[FakeResponse(500)] * 4)  # 1 try + 3 retries
    with pytest.raises(HttpError) as err:
        fake_http(session).get("https://x")
    assert err.value.status_code == 500
    assert len(session.calls) == 4


def test_client_errors_are_not_retried() -> None:
    session = FakeSession(FakeResponse(400), FakeResponse(200))
    with pytest.raises(HttpError) as err:
        fake_http(session).get("https://x")
    assert err.value.status_code == 400
    assert len(session.calls) == 1


def test_backoff_is_capped() -> None:
    http = fake_http(FakeSession())
    assert http.backoff_seconds(10) <= http.settings.backoff_max_seconds + 1.0


def test_rate_limiter_waits_for_min_interval() -> None:
    now = [100.0]
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    limiter = RateLimiter(3.0, clock=lambda: now[0], sleep=sleep)
    limiter.wait()          # first request: no wait
    now[0] += 1.0           # next request 1s later
    limiter.wait()
    assert sleeps == [pytest.approx(2.0)]
    now[0] += 5.0           # enough time has passed
    limiter.wait()
    assert len(sleeps) == 1


def test_retry_logs_do_not_contain_api_keys(caplog: pytest.LogCaptureFixture) -> None:
    session = FakeSession(FakeResponse(503), FakeResponse(200))
    http = fake_http(session)
    http.headers = {"x-api-key": "SUPER-SECRET"}
    with caplog.at_level(logging.WARNING):
        http.get("https://x", {"api_key": "SUPER-SECRET"})
    assert "SUPER-SECRET" not in caplog.text
    assert "retrying" in caplog.text
