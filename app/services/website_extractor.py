
import ipaddress
import socket
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup


def validate_url(url: str):
    """Validate URL and block private or local network addresses."""

    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        raise ValueError("Only HTTP and HTTPS URLs are allowed.")

    if not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Invalid website URL.")

    hostname = parsed.hostname

    try:
        addresses = socket.getaddrinfo(hostname, None)
        for address in addresses:
            ip = ipaddress.ip_address(address[4][0])
            if not ip.is_global:
                raise ValueError("Private or local URLs are not allowed.")
    except socket.gaierror:
        raise ValueError("Website domain could not be found.")

    return parsed


async def extract_website_content(url: str):
    """Fetch a website and extract useful text."""

    validate_url(url)

    try:
        async with httpx.AsyncClient(
            timeout=15,
            follow_redirects=False,
            trust_env=False
        ) as client:

            response = await client.get(
                url,
                headers={"User-Agent": "Mozilla/5.0 AI-Link-Analyzer"}
            )

            if response.status_code >= 300:
                raise ValueError(
                    f"Website returned status code {response.status_code}."
                )

            if len(response.content) > 2_000_000:
                raise ValueError("Website content is too large.")

            content_type = response.headers.get("content-type", "")

            if "text/html" not in content_type.lower():
                raise ValueError("The URL does not contain an HTML webpage.")

            soup = BeautifulSoup(response.text, "html.parser")

            for element in soup([
                "script", "style", "nav", "footer",
                "header", "noscript", "iframe"
            ]):
                element.decompose()

            title = soup.title.get_text(strip=True) if soup.title else ""

            meta = soup.find(
                "meta",
                attrs={"name": lambda value: value and value.lower() == "description"}
            )
            description = meta.get("content", "") if meta else ""

            headings = [
                heading.get_text(" ", strip=True)
                for heading in soup.find_all(["h1", "h2", "h3"])
            ]

            text = soup.get_text(separator=" ", strip=True)

            return {
                "url": url,
                "title": title,
                "meta_description": description,
                "headings": headings[:30],
                "content": text[:12000]
            }

    except httpx.RequestError as error:
        raise ValueError(f"Unable to access website: {error}")
