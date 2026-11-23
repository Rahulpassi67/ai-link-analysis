import asyncio
import httpx
import pytest

from app.services import website_extractor
from app.utils.errors import WebsiteExtractionError


class FakeResponse:
    def __init__(self, status_code=200, body=b"<html><body><p>Useful site content</p></body></html>"):
        self.status_code = status_code
        self.url = httpx.URL("https://public.example/")
        self.headers = {"content-type": "text/html; charset=utf-8"}
        self.encoding = "utf-8"
        self._body = body
        self.is_redirect = status_code in {301, 302, 303, 307, 308}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def raise_for_status(self):
        return None

    async def aiter_bytes(self):
        yield self._body


class FakeClient:
    def __init__(self, response, **kwargs):
        self.response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def stream(self, *args, **kwargs):
        return self.response


def test_http_server_error_is_a_safe_upstream_error(monkeypatch):
    monkeypatch.setattr(website_extractor, "validate_url", lambda url: None)
    monkeypatch.setattr(
        website_extractor.httpx,
        "AsyncClient",
        lambda **kwargs: FakeClient(FakeResponse(status_code=503))
    )

    with pytest.raises(WebsiteExtractionError) as raised:
        asyncio.run(website_extractor.fetch_with_httpx("https://public.example/"))

    assert raised.value.status_code == 502
    assert raised.value.code == "website_http_error"
    assert "503" in raised.value.public_message


def test_http_body_size_is_enforced_while_streaming(monkeypatch):
    monkeypatch.setattr(website_extractor, "validate_url", lambda url: None)
    large_body = b"x" * (website_extractor.MAX_CONTENT_SIZE + 1)
    monkeypatch.setattr(
        website_extractor.httpx,
        "AsyncClient",
        lambda **kwargs: FakeClient(FakeResponse(body=large_body))
    )

    with pytest.raises(WebsiteExtractionError) as raised:
        asyncio.run(website_extractor.fetch_with_httpx("https://public.example/"))

    assert raised.value.code == "website_too_large"
    assert raised.value.status_code == 413


def test_static_and_browser_timeouts_map_to_gateway_timeout(monkeypatch):
    monkeypatch.setattr(website_extractor, "validate_url", lambda url: None)

    async def static_timeout(url):
        raise httpx.ReadTimeout("private detail")

    async def browser_timeout(url):
        raise TimeoutError("private detail")

    monkeypatch.setattr(website_extractor, "fetch_with_httpx", static_timeout)
    monkeypatch.setattr(website_extractor, "fetch_with_playwright", browser_timeout)

    with pytest.raises(WebsiteExtractionError) as raised:
        asyncio.run(website_extractor.extract_website_content("https://public.example/"))

    assert raised.value.status_code == 504
    assert raised.value.code == "website_timeout"
    assert "private detail" not in str(raised.value)


def test_private_target_addresses_remain_blocked():
    with pytest.raises(ValueError, match="Private or local"):
        website_extractor.validate_url("http://127.0.0.1/")
