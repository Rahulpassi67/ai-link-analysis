import asyncio
import sys
import threading
import time
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
    monkeypatch.setattr(llm_service.random, "uniform", lambda low, high: (low + high) / 2)
    delays = []

    async def record_sleep(delay):
        delays.append(delay)

    monkeypatch.setattr(competitor_discovery.asyncio, "sleep", record_sleep)
    result = asyncio.run(competitor_discovery.extract_companies_with_gemini(
        {"brand_name": "Shopify", "category": "ecommerce", "products_services": []},
        []
    ))
    return result, calls["count"], delays


@pytest.fixture(autouse=True)
def _reset_gemini_runtime_state():
    with llm_service._GEMINI_STATE_LOCK:
        llm_service._ANALYSIS_CACHE.clear()
        llm_service._ANALYSIS_INFLIGHT.clear()
        llm_service._GEMINI_COOLDOWN_UNTIL = 0.0
        llm_service._GEMINI_COOLDOWN_REASON = ""
    with competitor_discovery._COMPETITOR_CACHE_LOCK:
        competitor_discovery._COMPETITOR_EXTRACTION_CACHE.clear()
    yield
    with llm_service._GEMINI_STATE_LOCK:
        llm_service._ANALYSIS_CACHE.clear()
        llm_service._ANALYSIS_INFLIGHT.clear()
        llm_service._GEMINI_COOLDOWN_UNTIL = 0.0
        llm_service._GEMINI_COOLDOWN_REASON = ""
    with competitor_discovery._COMPETITOR_CACHE_LOCK:
        competitor_discovery._COMPETITOR_EXTRACTION_CACHE.clear()


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


def test_gemini_server_error_uses_one_bounded_retry(monkeypatch):
    class ServerError(Exception):
        pass

    result, calls, delays = _gemini_company_extraction(
        monkeypatch,
        [ServerError("temporary server failure"), '{"companies":[{"name":"Northstar"}]}']
    )

    assert result == [{"name": "Northstar"}]
    assert calls == 2
    assert delays == [0.5]


def test_company_extraction_does_not_retry_quota_exhaustion(monkeypatch, caplog):
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

    assert calls["count"] == 1
    assert delays == []
    assert raised.value.code == "gemini_rate_limited"
    assert "temporarily rate limited" in raised.value.public_message
    attempt_records = [
        record for record in caplog.records
        if getattr(record, "diagnostic_event", None) == "competitor_discovery.gemini_extraction_failed"
    ]
    assert [record.attempt for record in attempt_records] == [1]
    assert attempt_records[-1].retrying is False


def test_deterministic_tavily_candidate_without_relationship_evidence_is_dropped(monkeypatch):
    search_result = {
        "url": "https://bigcommerce.com/platform",
        "title": "BigCommerce Ecommerce Platform",
        "content": (
            "BigCommerce is an ecommerce platform for merchants. "
            "It provides storefront and checkout tools."
        ),
    }
    monkeypatch.setattr(competitor_discovery, "require_api_key", lambda _name: "test-key")
    monkeypatch.setattr(competitor_discovery, "search_tavily", lambda *_args, **_kwargs: [search_result])

    verified_candidates = []

    async def verify_candidate(candidate, **_kwargs):
        verified_candidates.append(dict(candidate))
        return {
            "name": candidate["name"],
            "product_name": "Ecommerce platform",
            "evidence_status": "market_overlap_unverified",
            "evidence_confidence": 0.60,
            "heuristic_evidence_score": 0.60,
            "evidence_strength": "partial",
            "evidence_pages_checked": 1,
            "evidence_pages_accessible": 1,
            "evidence_details": [{
                "source_url": "https://bigcommerce.com/",
                "accessible": True,
                "classification": "market_overlap_unverified",
                "excerpts": [{
                    "source_url": "https://bigcommerce.com/",
                    "matched_text": "BigCommerce is an ecommerce platform for merchants.",
                    "classification": "market_overlap_unverified",
                }],
            }],
        }

    async def unexpected_gemini(*_args, **_kwargs):
        raise AssertionError("deterministic candidate extraction should avoid Gemini")

    monkeypatch.setattr(competitor_discovery, "extract_companies_with_gemini", unexpected_gemini)
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
    assert result == []
    assert "competitor_discovery_status" not in analysis


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
    models = []

    class FakeModels:
        def generate_content(self, **kwargs):
            calls["count"] += 1
            models.append(kwargs["model"])
            if calls["count"] == 1:
                raise RuntimeError("503 temporarily unavailable")
            return type("Response", (), {"text": "{}"})()

    monkeypatch.setattr(
        llm_service,
        "client",
        type("Client", (), {"models": FakeModels()})()
    )
    monkeypatch.setattr(llm_service, "validate_website_data", lambda data: None)
    monkeypatch.setattr(llm_service, "validate_analysis", lambda data, **_kwargs: VALID_ANALYSIS)
    monkeypatch.setattr(llm_service, "add_semantic_competitor_ranking", lambda data: data)

    async def no_sleep(seconds):
        return None

    monkeypatch.setattr(llm_service.asyncio, "sleep", no_sleep)
    result = asyncio.run(llm_service.analyze_with_ai({"content": "ignored by mocked validation"}))

    assert result["competitor_analysis"]["competitors"] == []
    assert calls["count"] == 2
    assert models == [llm_service.PRIMARY_GEMINI_MODEL] * 2


def test_identical_concurrent_website_analyses_share_and_cache_one_gemini_call(monkeypatch):
    calls = {"count": 0}
    call_lock = threading.Lock()

    class FakeModels:
        def generate_content(self, **_kwargs):
            with call_lock:
                calls["count"] += 1
            return type("Response", (), {"text": "{}"})()

    monkeypatch.setattr(
        llm_service,
        "client",
        type("Client", (), {"models": FakeModels()})(),
    )
    monkeypatch.setattr(llm_service, "validate_website_data", lambda _data: None)
    monkeypatch.setattr(llm_service, "validate_analysis", lambda _data, **_kwargs: VALID_ANALYSIS)
    monkeypatch.setattr(llm_service, "add_semantic_competitor_ranking", lambda data: data)
    website_data = {"url": "https://cache-singleflight.example/", "content": "same website"}

    async def run_concurrently():
        return await asyncio.gather(*[
            llm_service.analyze_with_ai(dict(website_data)) for _ in range(5)
        ])

    results = asyncio.run(run_concurrently())
    results[0]["marker"] = "caller mutation"
    cached_result = asyncio.run(llm_service.analyze_with_ai(dict(website_data)))

    assert calls["count"] == 1
    assert "marker" not in cached_result
    assert all(result["competitor_analysis"]["competitors"] == [] for result in results)


def test_competitor_discovery_skips_while_gemini_quota_cooldown_is_active(monkeypatch):
    llm_service.activate_gemini_cooldown(RuntimeError("429 RESOURCE_EXHAUSTED"))
    monkeypatch.setattr(
        competitor_discovery,
        "require_api_key",
        lambda _name: (_ for _ in ()).throw(AssertionError("Tavily should be skipped")),
    )
    analysis = {"competitor_analysis": {"brand_name": "Example"}}

    result = asyncio.run(competitor_discovery.discover_competitors(analysis))

    assert result == []
    assert analysis["competitor_discovery_status"]["status"] == "unavailable"


def test_gemini_falls_back_once_after_primary_transient_retries(monkeypatch, caplog):
    models = []
    delays = []

    class FakeServerError(Exception):
        status_code = 503
        message = "upstream unavailable"

    class FakeModels:
        def generate_content(self, **kwargs):
            models.append(kwargs["model"])
            if kwargs["model"] == llm_service.PRIMARY_GEMINI_MODEL:
                raise FakeServerError(FakeServerError.message)
            return type("Response", (), {"text": "{}"})()

    monkeypatch.setattr(
        llm_service,
        "client",
        type("Client", (), {"models": FakeModels()})()
    )
    monkeypatch.setattr(llm_service, "validate_website_data", lambda _data: None)
    monkeypatch.setattr(llm_service, "validate_analysis", lambda _data, **_kwargs: VALID_ANALYSIS)
    monkeypatch.setattr(llm_service, "add_semantic_competitor_ranking", lambda data: data)
    monkeypatch.setattr(llm_service.random, "uniform", lambda low, high: (low + high) / 2)

    async def record_sleep(delay):
        delays.append(delay)

    monkeypatch.setattr(llm_service.asyncio, "sleep", record_sleep)
    with caplog.at_level("INFO", logger="app.services.llm_service"):
        result = asyncio.run(llm_service.analyze_with_ai({"content": "ignored"}))

    assert result["competitor_analysis"]["competitors"] == []
    assert models == [llm_service.PRIMARY_GEMINI_MODEL] * 2 + [llm_service.FALLBACK_GEMINI_MODEL]
    assert delays == [0.5]
    assert any(
        getattr(record, "diagnostic_event", None) == "gemini.analysis_succeeded"
        and record.model == llm_service.FALLBACK_GEMINI_MODEL
        for record in caplog.records
    )


def test_gemini_does_not_fallback_for_permanent_or_validation_errors(monkeypatch):
    models = []

    class FakeModels:
        def generate_content(self, **kwargs):
            models.append(kwargs["model"])
            if "permanent" in kwargs["contents"]:
                raise RuntimeError("400 INVALID_ARGUMENT")
            return type("Response", (), {"text": '{"invalid": true}'})()

    monkeypatch.setattr(
        llm_service,
        "client",
        type("Client", (), {"models": FakeModels()})()
    )
    monkeypatch.setattr(llm_service, "validate_website_data", lambda _data: None)

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(llm_service.asyncio, "sleep", no_sleep)

    with pytest.raises(ExternalServiceError):
        asyncio.run(llm_service.analyze_with_ai({"content": "permanent failure"}))
    assert models == [llm_service.PRIMARY_GEMINI_MODEL]

    models.clear()
    with pytest.raises(ExternalServiceError):
        asyncio.run(llm_service.analyze_with_ai({"content": "validation failure"}))
    assert models == [llm_service.PRIMARY_GEMINI_MODEL] * 2


def test_gemini_quota_errors_stop_without_retry_or_fallback_and_log_safely(monkeypatch, caplog):
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
    with caplog.at_level("WARNING", logger="app.services.llm_service"):
        with pytest.raises(ExternalServiceError) as raised:
            asyncio.run(llm_service.analyze_with_ai({"content": "ignored"}))

    assert calls["count"] == 1
    assert raised.value.status_code == 503
    assert raised.value.code == "gemini_rate_limited"
    assert "temporarily rate limited" in raised.value.public_message
    assert "DO_NOT_EXPOSE" not in str(raised.value)
    request_failures = [
        record for record in caplog.records
        if getattr(record, "diagnostic_event", None) == "gemini.request_failed"
    ]
    assert len(request_failures) == 1
    assert request_failures[0].status_code == 429
    assert "quota key=[REDACTED]" in request_failures[0].error_message
    assert request_failures[-1].retrying is False
    assert request_failures[-1].model == llm_service.PRIMARY_GEMINI_MODEL
    assert "DO_NOT_EXPOSE" not in caplog.text


def test_exhausted_gemini_server_errors_return_a_clear_service_error(monkeypatch, caplog):
    calls = {"count": 0}

    class FakeServerError(Exception):
        status_code = 503
        message = "upstream temporarily unavailable"

    class FakeModels:
        def generate_content(self, **_kwargs):
            calls["count"] += 1
            raise FakeServerError(FakeServerError.message)

    monkeypatch.setattr(
        llm_service,
        "client",
        type("Client", (), {"models": FakeModels()})()
    )
    monkeypatch.setattr(llm_service, "validate_website_data", lambda _data: None)

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(llm_service.asyncio, "sleep", no_sleep)
    with caplog.at_level("WARNING", logger="app.services.llm_service"):
        with pytest.raises(ExternalServiceError) as raised:
            asyncio.run(llm_service.analyze_with_ai({"content": "ignored"}))

    assert calls["count"] == 3
    assert raised.value.code == "gemini_unavailable"
    assert raised.value.status_code == 503
    assert "temporarily unavailable after multiple attempts" in raised.value.public_message
    assert any(
        getattr(record, "diagnostic_event", None) == "gemini.request_failed"
        and record.status_code == 503
        and record.error_message == "upstream temporarily unavailable"
        and record.retrying is False
        for record in caplog.records
    )


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

    async def no_candidates(profile, results, website_url=""):
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


def test_competitor_extraction_cache_uses_normalized_inputs(monkeypatch, caplog):
    calls = {"count": 0}

    class FakeModels:
        def generate_content(self, **_kwargs):
            calls["count"] += 1
            return type("Response", (), {"text": '{"companies":[{"name":"Northstar"}]}'})()

    monkeypatch.setattr(
        competitor_discovery,
        "gemini_client",
        type("Client", (), {"models": FakeModels()})(),
    )
    first = asyncio.run(competitor_discovery.extract_companies_with_gemini(
        {"brand_name": "Example Brand", "category": "SaaS", "products_services": ["CRM"]},
        [{"url": "https://northstar.example/company/", "title": "Northstar CRM", "content": "CRM for growing teams"}],
        "https://www.example.com/path",
    ))
    second = asyncio.run(competitor_discovery.extract_companies_with_gemini(
        {"BRAND_NAME": "EXAMPLE BRAND", "CATEGORY": " saas ", "PRODUCTS_SERVICES": [" crm "]},
        [{"url": "https://northstar.example/company", "title": " Northstar   CRM ", "content": "CRM FOR GROWING TEAMS"}],
        "https://example.com/another-path",
    ))

    assert first == second == [{"name": "Northstar"}]
    assert calls["count"] == 1
    assert any(
        getattr(record, "diagnostic_event", None)
        == "competitor_discovery.gemini_extraction_cache_hit"
        for record in caplog.records
    )


def test_shared_gemini_request_limit_covers_concurrent_callers():
    active = {"now": 0, "max": 0}
    lock = threading.Lock()

    class FakeModels:
        def generate_content(self, **_kwargs):
            with lock:
                active["now"] += 1
                active["max"] = max(active["max"], active["now"])
            time.sleep(0.03)
            with lock:
                active["now"] -= 1
            return type("Response", (), {"text": "ok"})()

    client = type("Client", (), {"models": FakeModels()})()

    async def run_calls():
        return await asyncio.gather(*[
            llm_service.generate_gemini_content(client, 2, model="test")
            for _ in range(6)
        ])

    responses = asyncio.run(run_calls())

    assert len(responses) == 6
    assert 1 < active["max"] <= llm_service.GEMINI_MAX_CONCURRENT_REQUESTS


def test_retry_after_sets_quota_cooldown_without_retry_or_fallback(monkeypatch, caplog):
    models = []
    delays = []

    class RateLimitError(Exception):
        status_code = 429
        headers = {"Retry-After": "3"}

    class FakeModels:
        def generate_content(self, **kwargs):
            models.append(kwargs["model"])
            raise RateLimitError("429 RESOURCE_EXHAUSTED")

    monkeypatch.setattr(
        llm_service,
        "client",
        type("Client", (), {"models": FakeModels()})(),
    )
    monkeypatch.setattr(llm_service, "validate_website_data", lambda _data: None)

    async def record_sleep(delay):
        delays.append(delay)

    monkeypatch.setattr(llm_service.asyncio, "sleep", record_sleep)
    monkeypatch.setattr(llm_service.random, "uniform", lambda _low, _high: 0.4)
    analysis_token = llm_service.set_analysis_id("analysis-test-123")
    with caplog.at_level("WARNING", logger="app.services.llm_service"):
        with pytest.raises(ExternalServiceError):
            asyncio.run(llm_service.analyze_with_ai({"content": "retry-after test"}))
    llm_service.reset_analysis_id(analysis_token)

    assert models == [llm_service.PRIMARY_GEMINI_MODEL]
    assert delays == []
    remaining, reason = llm_service.gemini_cooldown_remaining()
    assert reason == "gemini_rate_limited"
    assert 0 < remaining <= 3.0
    failures = [
        record for record in caplog.records
        if getattr(record, "diagnostic_event", None) == "gemini.request_failed"
    ]
    assert len(failures) == 1
    assert all(record.analysis_id == "analysis-test-123" for record in failures)


def test_retry_after_is_not_capped_below_provider_request(monkeypatch):
    class RateLimitError(Exception):
        status_code = 429
        headers = {"Retry-After": "42"}

    assert llm_service._retry_after_seconds(RateLimitError()) == 42
    assert llm_service._retry_delay(0, RateLimitError()) == 42
