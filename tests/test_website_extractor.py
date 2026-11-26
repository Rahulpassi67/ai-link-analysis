import asyncio
import httpx
import pytest

from app.services import website_extractor
from app.utils.errors import WebsiteExtractionError


def test_html_extraction_excludes_navigation_footer_and_cookie_ui():
    html = """
    <html><head><title>Acme CRM</title></head><body>
      <header><a>Home</a><a>Pricing</a></header>
      <nav><a>Unrelated Product</a></nav>
      <div class="cookie-banner">Accept all cookies</div>
      <main><h1>Acme CRM</h1><p>CRM tools help sales teams manage leads.</p></main>
      <footer>Privacy policy Contact us</footer>
    </body></html>
    """

    result = website_extractor.extract_html(html, "https://public.example/", "test")

    assert "Acme CRM" in result["content"]
    assert "sales teams" in result["content"]
    assert "Pricing" not in result["content"]
    assert "Unrelated Product" not in result["content"]
    assert "Accept all cookies" not in result["content"]
    assert "Privacy policy" not in result["content"]


def test_html_extraction_falls_back_to_text_rich_section():
    html = """
    <html><body><nav>Menu item</nav>
      <section><h2>Services</h2><p>We provide useful consulting services.</p></section>
      <footer>Legal links</footer>
    </body></html>
    """

    result = website_extractor.extract_html(html, "https://public.example/", "test")

    assert "useful consulting services" in result["content"]
    assert "Menu item" not in result["content"]
    assert "Legal links" not in result["content"]


def test_html_extraction_keeps_legitimate_cookie_and_social_product_content():
    html = """
    <html><body>
      <main><h1>Cookie Manager</h1><p>Manage cookie settings for visitors.</p>
        <section class="social-media-marketing">
          <h2>Social media marketing</h2><p>Plan campaigns across social channels.</p>
        </section>
      </main>
      <div class="cookie-consent-banner">Accept cookies</div>
      <div id="main-navigation"><a>Unrelated product</a></div>
    </body></html>
    """

    result = website_extractor.extract_html(html, "https://public.example/", "test")

    assert "Manage cookie settings" in result["content"]
    assert "Plan campaigns" in result["content"]
    assert "Accept cookies" not in result["content"]
    assert "Unrelated product" not in result["content"]


def test_html_extraction_keeps_sibling_article_and_section_content():
    html = """
    <html><body>
      <main><h1>Product overview</h1><p>Manage inventory.</p></main>
      <article><h2>Shipping tools</h2><p>Compare shipping rates.</p></article>
      <section><h2>Payments</h2><p>Accept online payments.</p></section>
      <footer>Legal links</footer>
    </body></html>
    """

    result = website_extractor.extract_html(html, "https://public.example/", "test")

    assert "Manage inventory" in result["content"]
    assert "Compare shipping rates" in result["content"]
    assert "Accept online payments" in result["content"]
    assert "Legal links" not in result["content"]


def test_html_body_fallback_drops_hidden_sidebar_and_navigation_text():
    html = """
    <html><body>
      <div role="navigation">Home Pricing Careers</div>
      <div hidden>Hidden audience claim</div>
      <aside class="sidebar">Related stories and partner logos</aside>
      <div>Acme serves independent retailers and small agencies.</div>
    </body></html>
    """

    result = website_extractor.extract_html(html, "https://public.example/", "test")

    assert result["content"] == "Acme serves independent retailers and small agencies."


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
