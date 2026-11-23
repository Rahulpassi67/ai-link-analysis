
import asyncio
import ipaddress
import logging
import socket
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from app.utils.errors import WebsiteContentError, WebsiteExtractionError


MAX_CONTENT_SIZE = 2_000_000
MIN_CONTENT_LENGTH = 300
logger = logging.getLogger(__name__)

TIMEOUT = 15
DNS_TIMEOUT = 5

CHALLENGE_PHRASES = [
    "your request has been blocked",
    "access denied",
    "verify you are human",
    "verify that you are human",
    "captcha",
    "robot check",
    "security check",
    "unusual traffic",
    "automated requests",
    "request blocked",
    "checking your browser",
    "attention required",
]


def validate_url(url: str):
    """Allow only public HTTP/HTTPS website addresses."""
    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        raise ValueError("Only HTTP and HTTPS URLs are allowed.")

    if not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Invalid website URL.")

    if parsed.port and parsed.port not in (80, 443):
        raise ValueError("Only standard HTTP/HTTPS ports are allowed.")

    hostname = parsed.hostname

    try:
        try:
            ip = ipaddress.ip_address(hostname)
            addresses = [ip]
        except ValueError:
            results = socket.getaddrinfo(
                hostname, None, type=socket.SOCK_STREAM
            )
            addresses = [
                ipaddress.ip_address(item[4][0])
                for item in results
            ]

        if not addresses:
            raise ValueError("Website domain could not be found.")

        if any(not address.is_global for address in addresses):
            raise ValueError("Private or local URLs are not allowed.")

    except socket.gaierror:
        raise ValueError("Website domain could not be found.")

    return parsed


def is_security_challenge(title: str, content: str) -> bool:
    text = f"{title} {content}".lower()
    return any(phrase in text for phrase in CHALLENGE_PHRASES)


def extract_html(html: str, page_url: str, method: str):
    """Extract useful text and metadata from HTML."""
    soup = BeautifulSoup(html, "html.parser")

    for element in soup([
        "script", "style", "noscript", "iframe",
        "svg", "canvas", "template"
    ]):
        element.decompose()

    title = soup.title.get_text(" ", strip=True) if soup.title else ""

    meta = soup.find(
        "meta",
        attrs={"name": lambda value: value and value.lower() == "description"}
    )

    if not meta:
        meta = soup.find(
            "meta",
            attrs={"property": "og:description"}
        )

    description = meta.get("content", "").strip() if meta else ""

    headings = [
        heading.get_text(" ", strip=True)
        for heading in soup.find_all(["h1", "h2", "h3"])
    ]

    main_content = soup.find("main") or soup.find("article") or soup.body or soup

    content = main_content.get_text(separator=" ", strip=True)
    content = " ".join(content.split())

    return {
        "url": page_url,
        "title": title[:300],
        "meta_description": description[:1000],
        "headings": headings[:30],
        "content": content[:12000],
        "extraction_method": method,
        "content_length": len(content),
    }


async def fetch_with_httpx(url: str):
    """Try normal HTTP extraction first."""
    current_url = url

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(TIMEOUT),
        follow_redirects=False,
        trust_env=False,
        limits=httpx.Limits(
            max_connections=5,
            max_keepalive_connections=2
        )
    ) as client:

        for redirect_count in range(6):
            await asyncio.to_thread(validate_url, current_url)

            async with client.stream(
                "GET",
                current_url,
                headers={
                    "User-Agent": "Mozilla/5.0",
                    "Accept": "text/html,application/xhtml+xml"
                }
            ) as response:
                if response.status_code in (301, 302, 303, 307, 308):
                    location = response.headers.get("location")

                    if not location:
                        raise WebsiteExtractionError(
                            "invalid_redirect",
                            "The website returned an invalid redirect.",
                            502
                        )

                    if redirect_count == 5:
                        raise WebsiteExtractionError(
                            "too_many_redirects",
                            "The website redirected too many times.",
                            502
                        )

                    current_url = urljoin(str(response.url), location)
                    continue

                if response.status_code >= 400:
                    raise WebsiteExtractionError(
                        "website_http_error",
                        f"The target website returned HTTP {response.status_code}.",
                        502
                    )

                content_type = response.headers.get("content-type", "")
                if "text/html" not in content_type.lower():
                    raise WebsiteExtractionError(
                        "not_html",
                        "The submitted URL did not return an HTML webpage.",
                        422
                    )

                declared_size = response.headers.get("content-length")
                if declared_size and declared_size.isdigit() and int(declared_size) > MAX_CONTENT_SIZE:
                    raise WebsiteExtractionError(
                        "website_too_large",
                        "The website response exceeds the allowed size.",
                        413
                    )

                content = bytearray()
                async for chunk in response.aiter_bytes():
                    content.extend(chunk)
                    if len(content) > MAX_CONTENT_SIZE:
                        raise WebsiteExtractionError(
                            "website_too_large",
                            "The website response exceeds the allowed size.",
                            413
                        )
                encoding = response.encoding or "utf-8"
                html = bytes(content).decode(encoding, errors="replace")
                return extract_html(html, str(response.url), "httpx")

        raise ValueError("Unable to follow website redirects.")


async def fetch_with_playwright(url: str):
    """Render JavaScript-heavy pages using Chromium."""
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        raise WebsiteExtractionError(
            "browser_fallback_unavailable",
            "The page needs browser rendering, but the browser fallback is unavailable.",
            503
        )

    await asyncio.to_thread(validate_url, url)

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)

        try:
            context = await browser.new_context(
                service_workers="block"
            )

            page = await context.new_page()
            page.set_default_navigation_timeout(TIMEOUT * 1000)

            # Validate every network request before allowing it.
            async def guard_request(route):
                try:
                    request_url = route.request.url

                    if request_url.startswith(("http://", "https://")):
                        await asyncio.to_thread(
                            validate_url, request_url
                        )

                        if route.request.resource_type in (
                            "image", "media", "font"
                        ):
                            await route.abort()
                            return

                        await route.continue_()
                    else:
                        await route.abort()

                except Exception:
                    await route.abort()

            await page.route("**/*", guard_request)

            response = await page.goto(
                url,
                wait_until="domcontentloaded"
            )

            if response and response.status >= 400:
                raise ValueError(
                    f"Browser returned HTTP {response.status}."
                )

            await page.wait_for_timeout(1500)

            final_url = page.url
            await asyncio.to_thread(validate_url, final_url)

            html = await page.content()

            if len(html.encode("utf-8")) > MAX_CONTENT_SIZE:
                raise ValueError("Rendered website content is too large.")

            result = extract_html(
                html,
                final_url,
                "playwright"
            )

            return result

        finally:
            await browser.close()


async def extract_website_content(url: str):
    """
    Extract website content using HTTPX first,
    then Playwright if JavaScript rendering is needed.
    """
    try:
        await asyncio.wait_for(
            asyncio.to_thread(validate_url, url),
            timeout=DNS_TIMEOUT
        )
    except asyncio.TimeoutError as error:
        raise WebsiteExtractionError(
            "website_dns_timeout",
            "The website domain lookup timed out.",
            504
        ) from error

    try:
        result = await asyncio.wait_for(
            fetch_with_httpx(url),
            timeout=TIMEOUT + 2
        )

        # Do not use browser rendering to bypass security challenges.
        if is_security_challenge(
            result["title"], result["content"]
        ):
            raise WebsiteExtractionError(
                "website_security_challenge",
                "Website returned a security challenge. "
                "Automated extraction is not permitted.",
                422
            )

        if len(result["content"]) >= MIN_CONTENT_LENGTH:
            return result

        logger.info("Static extraction returned limited content; using browser fallback.")

    except WebsiteExtractionError:
        raise

    except ValueError as error:
        if "security challenge" in str(error).lower():
            raise
        logger.warning("Website extraction validation failed (%s).", type(error).__name__)

    except httpx.RequestError as error:
        logger.warning("Static website request failed (%s).", type(error).__name__)

    except asyncio.TimeoutError:
        logger.warning("Static website request timed out.")
        # A single bounded browser fallback remains useful for JS-heavy pages.

    # Use Playwright for JavaScript-rendered websites.
    try:
        result = await asyncio.wait_for(
            fetch_with_playwright(url),
            timeout=TIMEOUT + 5
        )

        if is_security_challenge(
            result["title"], result["content"]
        ):
            raise WebsiteExtractionError(
                "website_security_challenge",
                "Website returned a security challenge. "
                "Automated extraction is not permitted.",
                422
            )

        if len(result["content"]) < MIN_CONTENT_LENGTH:
            raise WebsiteContentError(
                "Not enough website content could be extracted."
            )

        return result

    except WebsiteExtractionError:
        raise

    except ValueError:
        raise WebsiteContentError(
            "The website did not provide enough usable content for analysis."
        )

    except asyncio.TimeoutError as error:
        raise WebsiteExtractionError(
            "website_timeout",
            "The website extraction timed out.",
            504
        ) from error

    except Exception as error:
        logger.warning("Browser website extraction failed (%s).", type(error).__name__)
        raise WebsiteExtractionError(
            "website_extraction_failed",
            "The website could not be extracted. Check that it is publicly accessible and try again.",
            502
        ) from error
