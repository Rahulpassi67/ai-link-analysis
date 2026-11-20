
import pytest
from fastapi.testclient import TestClient

import app.main as main


client = TestClient(main.app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "success"


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_invalid_url():
    response = client.post(
        "/link-analysis",
        json={"url": "not-a-valid-url"}
    )

    assert response.status_code == 422


def test_successful_link_analysis(monkeypatch):

    async def mock_extract(url):
        return {
            "url": url,
            "title": "Example Business",
            "meta_description": "Example business website"
        }

    async def mock_analyze(website_data):
        return {
            "competitor_analysis": {},
            "target_search_queries": [],
            "seo_analysis": {},
            "top_services": [],
            "target_audience": {}
        }

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

    assert response.status_code == 400
    assert "Website could not be accessed" in response.json()["detail"]
