import asyncio
import sys
import types

import pytest

from app.services import competitor_discovery, evidence_verifier, llm_service
from app.utils.errors import ExternalServiceError


VALID_ANALYSIS = {
    "competitor_analysis": {},
    "target_search_queries": [],
    "seo_analysis": {},
    "top_services": [],
    "target_audience": {}
}


def _gemini_company_extraction(monkeypatch, outcomes):
    calls = {"count": 0}

    class FakeModels:
        def generate_content(self, **kwargs):
            outcome = outcomes[calls["count"]]
            calls["count"] += 1
            if isinstance(outcome, Exception):
                raise outcome
            return type("Response", (), {"text": outcome})()

    monkeypatch.setattr(
        competitor_discovery,
        "gemini_client",
        type("Client", (), {"models": FakeModels()})()
    )
    monkeypatch.setattr(competitor_discovery, "GEMINI_DISCOVERY_TIMEOUT", 20)
    delays = []

    async def record_sleep(delay):
        delays.append(delay)

    monkeypatch.setattr(competitor_discovery.asyncio, "sleep", record_sleep)
    result = asyncio.run(competitor_discovery.extract_companies_with_gemini(
        {"brand_name": "Shopify", "category": "ecommerce", "products_services": []},
        []
    ))
    return result, calls["count"], delays


def test_company_extraction_succeeds_on_first_attempt(monkeypatch):
    result, calls, delays = _gemini_company_extraction(
        monkeypatch,
        ['{"companies":[{"name":"Northstar"}]}']
    )

    assert result == [{"name": "Northstar"}]
    assert calls == 1
    assert delays == []


def test_company_extraction_retries_temporary_failure_then_succeeds(monkeypatch):
    result, calls, delays = _gemini_company_extraction(
        monkeypatch,
        [RuntimeError("503 UNAVAILABLE"), '{"companies":[]}']
    )

    assert result == []
    assert calls == 2
    assert delays == [0.5]


def test_gemini_server_error_retries_and_can_succeed_on_third_attempt(monkeypatch):
    class ServerError(Exception):
        pass

    result, calls, delays = _gemini_company_extraction(
        monkeypatch,
        [ServerError("temporary server failure"), ServerError("temporary server failure"),
         '{"companies":[{"name":"Northstar"}]}']
    )

    assert result == [{"name": "Northstar"}]
    assert calls == 3
    assert delays == [0.5, 1.0]


def test_company_extraction_stops_after_three_temporary_failures(monkeypatch, caplog):
    calls = {"count": 0}

    class FakeModels:
        def generate_content(self, **kwargs):
            calls["count"] += 1
            raise RuntimeError("429 RESOURCE_EXHAUSTED")

    monkeypatch.setattr(
        competitor_discovery,
        "gemini_client",
        type("Client", (), {"models": FakeModels()})()
    )
    delays = []

    async def record_sleep(delay):
        delays.append(delay)

    monkeypatch.setattr(competitor_discovery.asyncio, "sleep", record_sleep)
    with caplog.at_level("WARNING", logger="app.services.competitor_discovery"):
        with pytest.raises(ExternalServiceError) as raised:
            asyncio.run(competitor_discovery.extract_companies_with_gemini(
                {"brand_name": "Shopify", "category": "ecommerce", "products_services": []},
                []
            ))

    assert calls["count"] == 3
    assert delays == [0.5, 1.0]
    assert raised.value.code == "gemini_rate_limited"
    attempt_records = [
        record for record in caplog.records
        if getattr(record, "diagnostic_event", None) == "competitor_discovery.gemini_extraction_failed"
    ]
    assert [record.attempt for record in attempt_records] == [1, 2, 3]
    assert attempt_records[-1].retrying is False


def test_gemini_exhaustion_falls_back_to_validated_tavily_candidates(monkeypatch):
    search_result = {
        "url": "https://bigcommerce.com/platform",
        "title": "BigCommerce Ecommerce Platform",
        "content": (
            "BigCommerce is an ecommerce platform for merchants. "
            "It provides storefront and checkout tools."
        ),
    }
    monkeypatch.setattr(competitor_discovery, "require_api_key", lambda _name: "test-key")
    monkeypatch.setattr(competitor_discovery, "gemini_client", object())
    monkeypatch.setattr(competitor_discovery, "search_tavily", lambda *_args, **_kwargs: [search_result])

    async def gemini_unavailable(*_args, **_kwargs):
        raise ExternalServiceError("gemini_unavailable", "Gemini is unavailable.", 503)

    verified_candidates = []

    async def verify_candidate(candidate, **_kwargs):
        verified_candidates.append(dict(candidate))
        return {
            "name": candidate["name"],
            "product_name": "",
            "evidence_status": "company_and_product_mentioned",
            "evidence_confidence": 0.60,
            "heuristic_evidence_score": 0.60,
            "evidence_strength": "partial",
            "evidence_pages_checked": 1,
            "evidence_pages_accessible": 1,
            "evidence_details": [{
                "source_url": "https://bigcommerce.com/",
                "accessible": True,
                "classification": "company_and_product_mentioned",
                "excerpts": [{
                    "source_url": "https://bigcommerce.com/",
                    "matched_text": "BigCommerce is an ecommerce platform for merchants.",
                    "classification": "company_and_product_mentioned",
                }],
            }],
        }

    monkeypatch.setattr(competitor_discovery, "extract_companies_with_gemini", gemini_unavailable)
    monkeypatch.setattr(evidence_verifier, "verify_competitor_evidence", verify_candidate)
    semantic_matcher = types.ModuleType("app.services.semantic_matcher")
    semantic_matcher.rank_similar_items = lambda _profile, descriptions: [
        {"description": item, "similarity_score": 0.5} for item in descriptions
    ]
    monkeypatch.setitem(sys.modules, "app.services.semantic_matcher", semantic_matcher)
    analysis = {"competitor_analysis": {
        "brand_name": "Shopify",
        "category": "ecommerce platform",
        "products_services": ["online store platform"],
    }}

    result = asyncio.run(competitor_discovery.discover_competitors(
        analysis,
        "https://shopify.example",
    ))

    assert len(verified_candidates) == 1
    assert verified_candidates[0]["name"] == "BigCommerce"
    assert verified_candidates[0]["official_website"] == "https://bigcommerce.com"
    assert result[0]["name"] == "BigCommerce"
    assert result[0]["competitor_type"] == "unclear"
    assert result[0]["verification_status"] == "official_website_validated"
    assert analysis["competitor_discovery_status"]["status"] == "partial"


def test_tavily_fallback_filters_excluded_domains_brand_and_duplicate_companies():
    fallback = competitor_discovery.fallback_candidates_from_search_results(
        {"brand_name": "Shopify"},
        [
            {
                "url": "https://bigcommerce.com/platform",
                "title": "BigCommerce Ecommerce Platform",
                "content": "BigCommerce provides ecommerce software.",
            },
            {
                "url": "https://www.bigcommerce.com/pricing",
                "title": "BigCommerce pricing",
                "content": "BigCommerce plans for merchants.",
            },
            {
                "url": "https://shopify.com/",
                "title": "Shopify ecommerce platform",
                "content": "Shopify offers an online store platform.",
            },
            {
                "url": "https://g2.com/bigcommerce",
                "title": "BigCommerce reviews",
                "content": "BigCommerce software reviews.",
            },
        ],
    )

    assert len(fallback) == 1
    assert fallback[0]["name"] == "BigCommerce"
    assert fallback[0]["competitor_type"] == "unclear"
    assert fallback[0]["evidence_urls"] == ["https://bigcommerce.com/platform"]


def test_company_extraction_does_not_retry_permanent_errors(monkeypatch):
    calls = {"count": 0}

    class FakeModels:
        def generate_content(self, **kwargs):
            calls["count"] += 1
            raise RuntimeError("400 INVALID_ARGUMENT")

    monkeypatch.setattr(
        competitor_discovery,
        "gemini_client",
        type("Client", (), {"models": FakeModels()})()
    )
    with pytest.raises(ExternalServiceError):
        asyncio.run(competitor_discovery.extract_companies_with_gemini(
            {"brand_name": "Shopify", "category": "ecommerce", "products_services": []},
            []
        ))

    assert calls["count"] == 1


def test_company_extraction_retries_timeouts(monkeypatch):
    result, calls, delays = _gemini_company_extraction(
        monkeypatch,
        [TimeoutError("request timed out"), '{"companies":[]}']
    )

    assert result == []
    assert calls == 2
    assert delays == [0.5]


def test_company_extraction_invalid_json_is_not_retried(monkeypatch):
    with pytest.raises(ExternalServiceError) as raised:
        _gemini_company_extraction(monkeypatch, ["not json"])

    assert raised.value.code == "gemini_discovery_invalid_response"


def test_gemini_retries_transient_errors_with_a_controlled_retry(monkeypatch):
    calls = {"count": 0}

    class FakeModels:
        def generate_content(self, **kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                raise RuntimeError("503 temporarily unavailable")
            return type("Response", (), {"text": "{}"})()

    monkeypatch.setattr(
        llm_service,
        "client",
        type("Client", (), {"models": FakeModels()})()
    )
    monkeypatch.setattr(llm_service, "validate_website_data", lambda data: None)
    monkeypatch.setattr(llm_service, "validate_analysis", lambda data: VALID_ANALYSIS)
    monkeypatch.setattr(llm_service, "add_semantic_competitor_ranking", lambda data: data)

    async def no_sleep(seconds):
        return None

    monkeypatch.setattr(llm_service.asyncio, "sleep", no_sleep)
    result = asyncio.run(llm_service.analyze_with_ai({"content": "ignored by mocked validation"}))

    assert result is VALID_ANALYSIS
    assert calls["count"] == 2


def test_gemini_quota_errors_are_sanitized_after_bounded_retries(monkeypatch):
    calls = {"count": 0}

    class FakeModels:
        def generate_content(self, **kwargs):
            calls["count"] += 1
            raise RuntimeError("429 quota key=DO_NOT_EXPOSE")

    monkeypatch.setattr(
        llm_service,
        "client",
        type("Client", (), {"models": FakeModels()})()
    )
    monkeypatch.setattr(llm_service, "validate_website_data", lambda data: None)

    async def no_sleep(seconds):
        return None

    monkeypatch.setattr(llm_service.asyncio, "sleep", no_sleep)
    with pytest.raises(ExternalServiceError) as raised:
        asyncio.run(llm_service.analyze_with_ai({"content": "ignored"}))

    assert calls["count"] == 3
    assert raised.value.status_code == 503
    assert "DO_NOT_EXPOSE" not in str(raised.value)


def test_all_tavily_queries_failing_becomes_a_safe_service_error(monkeypatch):
    monkeypatch.setattr(competitor_discovery, "require_api_key", lambda name: "test-key")
    monkeypatch.setattr(competitor_discovery, "gemini_client", object())

    def fail_search(query, api_key):
        raise RuntimeError("429 provider key=DO_NOT_EXPOSE")

    monkeypatch.setattr(competitor_discovery, "search_tavily", fail_search)
    analysis = {"competitor_analysis": {
        "brand_name": "Linkly AI",
        "category": "website intelligence",
        "products_services": ["competitor analysis software"]
    }}

    with pytest.raises(ExternalServiceError) as raised:
        asyncio.run(competitor_discovery.discover_competitors(analysis))

    assert raised.value.status_code == 503
    assert "DO_NOT_EXPOSE" not in str(raised.value)


def test_partial_tavily_search_failure_preserves_available_results(monkeypatch):
    monkeypatch.setattr(competitor_discovery, "require_api_key", lambda name: "test-key")
    monkeypatch.setattr(competitor_discovery, "gemini_client", object())

    def search(query, api_key):
        if "companies offering" in query:
            raise RuntimeError("temporary failure")
        return [{
            "url": "https://northstar-systems.com/about",
            "title": "Northstar Systems",
            "content": "Northstar Systems develops website intelligence software."
        }]

    async def no_candidates(profile, results):
        return []

    monkeypatch.setattr(competitor_discovery, "search_tavily", search)
    monkeypatch.setattr(competitor_discovery, "extract_companies_with_gemini", no_candidates)
    analysis = {"competitor_analysis": {
        "brand_name": "Linkly AI",
        "category": "website intelligence",
        "products_services": ["competitor analysis software"]
    }}

    result = asyncio.run(competitor_discovery.discover_competitors(analysis))

    assert result == []
    assert analysis["competitor_discovery_status"]["status"] == "partial"


def test_one_failed_candidate_does_not_stop_evidence_verification(monkeypatch):
    async def verifier(candidate, **kwargs):
        if candidate["name"] == "Broken Candidate":
            raise RuntimeError("private candidate detail")
        return {
            "name": candidate["name"],
            "evidence_status": "inaccessible",
            "evidence_classification": "page_inaccessible",
            "evidence_pages_checked": 1,
            "evidence_pages_accessible": 0,
            "evidence_details": []
        }

    monkeypatch.setattr(evidence_verifier, "verify_competitor_evidence", verifier)
    results = asyncio.run(evidence_verifier.verify_competitors([
        {"name": "Broken Candidate"},
        {"name": "Healthy Candidate"}
    ]))

    assert results[0]["evidence_status"] == "verification_error"
    assert results[0]["evidence_classification"] == "unverified"
    assert results[1]["name"] == "Healthy Candidate"
    assert results[1]["evidence_status"] == "inaccessible"


def test_duplicate_evidence_urls_are_fetched_and_returned_once(monkeypatch):
    calls = []

    async def page_fetch(client, url):
        calls.append(url)
        return {
            "url": url,
            "accessible": True,
            "reason": None,
            "text": "Northstar Systems provides Orbit."
        }

    monkeypatch.setattr(evidence_verifier, "fetch_evidence_page", page_fetch)
    candidate = {
        "name": "Northstar Systems",
        "product_name": "Orbit",
        "evidence_urls": [
            "https://source.example/page?source=one",
            "https://www.source.example/page?source=two",
            "https://source.example/page/"
        ]
    }

    result = asyncio.run(evidence_verifier.verify_competitor_evidence(candidate))

    assert len(calls) == 1
    assert result["evidence_pages_checked"] == 1
    assert len(result["evidence_details"]) == 1
    assert len(result["evidence_details"][0]["excerpts"]) == 1
