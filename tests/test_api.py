
import pytest
from fastapi.testclient import TestClient

import app.main as main


client = TestClient(main.app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Linkly AI" in response.text


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_health_available_without_provider_keys(monkeypatch):
    from app.config import ConfigurationError, require_api_key

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

    with pytest.raises(ConfigurationError, match="GEMINI_API_KEY"):
        require_api_key("GEMINI_API_KEY")

    with pytest.raises(ConfigurationError, match="TAVILY_API_KEY"):
        require_api_key("TAVILY_API_KEY")


def test_analysis_reports_missing_gemini_configuration(monkeypatch):
    async def mock_extract(url):
        return {
            "url": url,
            "title": "Example",
            "meta_description": "",
            "content": "Example business content. " * 20
        }

    async def missing_gemini(website_data):
        from app.config import ConfigurationError
        raise ConfigurationError("GEMINI_API_KEY is not configured.")

    monkeypatch.setattr(main, "extract_website_content", mock_extract)
    monkeypatch.setattr(main, "analyze_with_ai", missing_gemini)

    response = client.post(
        "/link-analysis",
        json={"url": "https://example.com"}
    )

    assert response.status_code == 503
    assert "GEMINI_API_KEY" in response.json()["detail"]


def test_analysis_reports_missing_tavily_configuration(monkeypatch):
    async def mock_extract(url):
        return {
            "url": url,
            "title": "Example",
            "meta_description": "",
            "content": "Example business content. " * 20
        }

    async def mock_analyze(website_data):
        return {
            "competitor_analysis": {"brand_name": "Example"},
            "target_search_queries": [],
            "seo_analysis": {},
            "top_services": [],
            "target_audience": {}
        }

    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setattr(main, "extract_website_content", mock_extract)
    monkeypatch.setattr(main, "analyze_with_ai", mock_analyze)

    response = client.post(
        "/link-analysis",
        json={"url": "https://example.com"}
    )

    assert response.status_code == 200
    assert response.json()["analysis"]["discovered_competitors"] == []
    assert response.json()["analysis"]["competitor_discovery_status"]["status"] == "unavailable"


def test_invalid_url():
    response = client.post(
        "/link-analysis",
        json={"url": "not-a-valid-url"}
    )

    assert response.status_code == 422
    assert response.json()["success"] is False
    assert response.json()["error"]["code"] == "invalid_request"


def test_missing_url_and_malformed_json_have_consistent_errors():
    missing = client.post("/link-analysis", json={})
    malformed = client.post(
        "/link-analysis",
        content="{",
        headers={"Content-Type": "application/json"}
    )

    for response in (missing, malformed):
        assert response.status_code == 422
        assert response.json()["success"] is False
        assert response.json()["error"]["code"] == "invalid_request"
        assert isinstance(response.json()["detail"], str)


def test_empty_extraction_returns_clear_error(monkeypatch):
    async def mock_extract(url):
        return {"url": url, "title": "Example", "meta_description": "", "content": " "}

    monkeypatch.setattr(main, "extract_website_content", mock_extract)
    response = client.post("/link-analysis", json={"url": "https://example.com"})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "insufficient_website_content"


def test_incomplete_analysis_response_is_rejected(monkeypatch):
    async def mock_extract(url):
        return {
            "url": url,
            "title": "Example",
            "meta_description": "",
            "content": "Example business content. " * 20
        }

    async def mock_analyze(website_data):
        return {"competitor_analysis": {}}

    monkeypatch.setattr(main, "extract_website_content", mock_extract)
    monkeypatch.setattr(main, "analyze_with_ai", mock_analyze)
    response = client.post("/link-analysis", json={"url": "https://example.com"})

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "invalid_analysis_response"


def test_unreachable_site_and_timeout_use_structured_errors(monkeypatch):
    from app.utils.errors import WebsiteExtractionError

    async def unreachable(url):
        raise WebsiteExtractionError(
            "website_http_error", "The target website returned HTTP 503.", 502
        )

    monkeypatch.setattr(main, "extract_website_content", unreachable)
    response = client.post("/link-analysis", json={"url": "https://example.com"})
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "website_http_error"

    async def timeout(url):
        raise WebsiteExtractionError(
            "website_timeout", "The website extraction timed out.", 504
        )

    monkeypatch.setattr(main, "extract_website_content", timeout)
    response = client.post("/link-analysis", json={"url": "https://example.com"})
    assert response.status_code == 504
    assert response.json()["error"]["code"] == "website_timeout"


def test_gemini_quota_error_is_reported_safely(monkeypatch):
    from app.utils.errors import ExternalServiceError

    async def mock_extract(url):
        return {
            "url": url,
            "title": "Example",
            "meta_description": "",
            "content": "Example business content. " * 20
        }

    async def rate_limited(website_data):
        raise ExternalServiceError(
            "gemini_rate_limited",
            "Gemini is temporarily rate limited. Please try again later.",
            503
        )

    monkeypatch.setattr(main, "extract_website_content", mock_extract)
    monkeypatch.setattr(main, "analyze_with_ai", rate_limited)
    response = client.post("/link-analysis", json={"url": "https://example.com"})

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "gemini_rate_limited"
    assert "API key" not in response.text


def test_shopify_analysis_survives_optional_competitor_discovery_failure(monkeypatch):
    from app.utils.errors import ExternalServiceError

    async def mock_extract(url):
        return {
            "url": url,
            "title": "Example",
            "meta_description": "",
            "content": "Example business content. " * 20
        }

    async def mock_analyze(website_data):
        return {
            "competitor_analysis": {
                "brand_name": "Shopify",
                "overview": "Commerce platform for businesses."
            },
            "target_search_queries": [],
            "seo_analysis": {},
            "top_services": ["Online store platform"],
            "target_audience": {"primary": "Merchants"}
        }

    async def fail_discovery(analysis, website_url):
        raise ExternalServiceError(
            "tavily_rate_limited",
            "Tavily is temporarily rate limited. Please try again later.",
            503
        )

    monkeypatch.setattr(main, "extract_website_content", mock_extract)
    monkeypatch.setattr(main, "analyze_with_ai", mock_analyze)
    monkeypatch.setattr(main, "discover_competitors", fail_discovery)

    response = client.post("/link-analysis", json={"url": "https://www.shopify.com"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["analysis"]["discovered_competitors"] == []
    assert payload["analysis"]["competitor_discovery_status"]["status"] == "unavailable"
    assert payload["analysis"]["competitor_discovery_status"]["message"]
    assert payload["analysis"]["competitor_analysis"]["brand_name"] == "Shopify"
    assert payload["analysis"]["top_services"] == ["Online store platform"]
    assert payload["analysis"]["target_audience"]["primary"] == "Merchants"


def test_unexpected_discovery_exception_preserves_main_analysis(monkeypatch):
    async def mock_extract(url):
        return {
            "url": url,
            "title": "Example",
            "meta_description": "",
            "content": "Example business content. " * 20
        }

    async def mock_analyze(_website_data):
        return {
            "competitor_analysis": {"brand_name": "Shopify"},
            "target_search_queries": [],
            "seo_analysis": {},
            "top_services": ["Online store platform"],
            "target_audience": {"primary": "Merchants"}
        }

    async def unexpected_discovery_failure(*_args, **_kwargs):
        raise RuntimeError("temporary discovery implementation failure")

    monkeypatch.setattr(main, "extract_website_content", mock_extract)
    monkeypatch.setattr(main, "analyze_with_ai", mock_analyze)
    monkeypatch.setattr(main, "discover_competitors", unexpected_discovery_failure)

    response = client.post("/link-analysis", json={"url": "https://www.shopify.com"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["analysis"]["discovered_competitors"] == []
    assert payload["analysis"]["competitor_discovery_status"]["status"] == "unavailable"
    assert payload["analysis"]["competitor_analysis"]["brand_name"] == "Shopify"
    assert payload["analysis"]["top_services"] == ["Online store platform"]


def test_total_request_deadline_returns_gateway_timeout(monkeypatch):
    import asyncio

    async def slow_extract(url):
        await asyncio.sleep(0.05)
        return {
            "url": url,
            "title": "Example",
            "meta_description": "",
            "content": "Example business content. " * 20
        }

    monkeypatch.setattr(main, "ANALYSIS_TIMEOUT_SECONDS", 0.005)
    monkeypatch.setattr(main, "extract_website_content", slow_extract)

    response = client.post("/link-analysis", json={"url": "https://example.com"})

    assert response.status_code == 504
    assert response.json()["error"]["code"] == "analysis_timeout"


def test_successful_link_analysis(monkeypatch):

    async def mock_extract(url):
        return {
            "url": url,
            "title": "Example Business",
            "meta_description": "Example business website",
            "content": "Example business website content. " * 20
        }

    async def mock_analyze(website_data):
        return {
            "competitor_analysis": {},
            "target_search_queries": [],
            "seo_analysis": {},
            "top_services": [],
            "target_audience": {}
        }

    async def mock_discover(analysis, website_url):
        return [{
            "name": "Northstar Systems",
            "evidence_details": [{
                "competitor_name": "Northstar Systems",
                "source_url": "https://research.example/compare",
                "accessible": True,
                "accessibility_status": "accessible",
                "classification": "company_mentioned",
                "excerpts": [{
                    "competitor_name": "Northstar Systems",
                    "source_url": "https://research.example/compare",
                    "matched_text": "Northstar Systems makes Orbit.",
                    "matching_terms": ["Northstar Systems"],
                    "accessible": True,
                    "accessibility_status": "accessible",
                    "classification": "company_mentioned"
                }]
            }]
        }]

    monkeypatch.setattr(
        main,
        "extract_website_content",
        mock_extract
    )

    monkeypatch.setattr(
        main,
        "analyze_with_ai",
        mock_analyze
    )
    monkeypatch.setattr(
        main,
        "discover_competitors",
        mock_discover
    )

    response = client.post(
        "/link-analysis",
        json={"url": "https://example.com"}
    )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert data["website"]["url"] == "https://example.com/"
    assert data["website"]["title"] == "Example Business"

    assert "analysis" in data
    assert "competitor_analysis" in data["analysis"]
    assert "target_search_queries" in data["analysis"]
    assert "seo_analysis" in data["analysis"]
    assert "top_services" in data["analysis"]
    assert "target_audience" in data["analysis"]
    excerpt = data["analysis"]["discovered_competitors"][0]["evidence_details"][0]["excerpts"][0]
    assert excerpt["matched_text"] == "Northstar Systems makes Orbit."
    assert excerpt["source_url"] == "https://research.example/compare"


def test_website_extraction_error(monkeypatch):

    async def mock_extract(url):
        raise ValueError("Website could not be accessed")

    monkeypatch.setattr(
        main,
        "extract_website_content",
        mock_extract
    )

    response = client.post(
        "/link-analysis",
        json={"url": "https://example.com"}
    )

    assert response.status_code == 422
    assert "Website could not be accessed" in response.json()["detail"]
