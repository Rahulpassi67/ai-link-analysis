import asyncio
import json
from enum import IntEnum
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import main
from app.services import llm_service
from app.utils.errors import ExternalServiceError


def _valid_analysis():
    return {
        "competitor_analysis": {
            "brand_name": "Example CRM",
            "overview": "Example company overview.",
            "category": "CRM software",
            "products_services": ["CRM software"],
            "competitors": [],
            "brand_advantages": [],
            "positioning_opportunities": [],
        },
        "target_search_queries": [{
            "query": "How Example CRM software helps teams",
            "platform": "Google",
            "platform_type": "search_engine",
            "intent": "informational",
            "relevance": "Explains how Example CRM software supports its business profile.",
            "target_audience": "Not available",
        }],
        "seo_analysis": {
            "primary_keywords": [],
            "secondary_keywords": [],
            "long_tail_keywords": [],
            "business_keywords": [],
            "branded_product_keywords": [],
            "seo_title": "Example SEO title",
            "meta_description": "Example meta description",
            "content_recommendations": [],
            "keyword_intents": {
                "primary": [],
                "secondary": [],
                "long_tail": [],
                "business": [],
                "branded_products": [],
            },
            "content_gaps": [],
            "prioritized_recommendations": [],
        },
        "top_services": [
            {
                "name": f"Service {index}",
                "description": "A service description.",
                "target_industry": "Not available",
                "target_customer": "Not available",
                "business_benefit": "A business benefit.",
                "inferred": False,
            }
            for index in range(5)
        ],
        "target_audience": {
            "primary_audience": "Business users",
            "secondary_audience": "Not available",
            "industry_segments": [],
            "business_types": [],
            "customer_profiles": [],
            "geographic_focus": [],
            "needs_and_challenges": [],
            "interests": [],
            "recommended_marketing_segments": [],
            "industry_segment_evidence": [],
            "audience_evidence": [],
        },
    }


def _mock_gemini(monkeypatch, response_text, metadata=None):
    responses = response_text if isinstance(response_text, list) else [response_text]

    class FakeModels:
        def __init__(self):
            self.calls = 0
            self.request_contents = []

        def generate_content(self, **_kwargs):
            index = min(self.calls, len(responses) - 1)
            self.calls += 1
            self.request_contents.append(_kwargs.get("contents"))
            return SimpleNamespace(text=responses[index], **(metadata or {}))

    fake_models = FakeModels()
    monkeypatch.setattr(
        llm_service,
        "client",
        type("Client", (), {"models": fake_models})(),
    )
    monkeypatch.setattr(llm_service, "validate_website_data", lambda _data: None)
    monkeypatch.setattr(llm_service, "add_semantic_competitor_ranking", lambda data: data)
    return fake_models


def _run_analysis(website_data=None):
    return asyncio.run(llm_service.analyze_with_ai(website_data or {
        "title": "Example",
        "content": "WEBSITE_CONTENT_SENTINEL",
    }))


def _failure_record(caplog):
    return next(
        record for record in caplog.records
        if getattr(record, "diagnostic_event", None)
        == "gemini.response_validation_failed"
    )


def test_missing_gemini_fields_are_logged_without_response_content(monkeypatch, caplog):
    analysis = _valid_analysis()
    del analysis["competitor_analysis"]["brand_name"]
    response_text = json.dumps(analysis)
    _mock_gemini(monkeypatch, response_text, metadata={
        "candidates": [SimpleNamespace(finish_reason="STOP")],
        "usage_metadata": SimpleNamespace(
            prompt_token_count=1200,
            candidates_token_count=2400,
            total_token_count=3600,
        ),
        "model_version": "gemini-test-version",
    })

    with caplog.at_level("WARNING", logger="app.services.llm_service"):
        with pytest.raises(ExternalServiceError) as raised:
            _run_analysis()

    record = _failure_record(caplog)
    assert record.failure_category == "missing_field"
    assert record.missing_fields == ["competitor_analysis.brand_name"]
    assert record.invalid_fields == []
    assert record.top_level_keys == sorted(analysis.keys())
    assert record.json_parse_failed is False
    assert record.finish_reason == "STOP"
    assert record.response_metadata["usage"]["candidates_token_count"] == 2400
    diagnostic = json.loads(record.getMessage().split(" ", 1)[1])
    assert diagnostic["missing_fields"] == ["competitor_analysis.brand_name"]
    assert diagnostic["top_level_keys"] == sorted(analysis.keys())
    assert diagnostic["json_parse_failed"] is False
    assert raised.value.code == "gemini_invalid_response"
    assert raised.value.status_code == 502
    assert raised.value.public_message == (
        "Gemini returned an incomplete analysis response. Please try again."
    )
    assert "WEBSITE_CONTENT_SENTINEL" not in caplog.text
    assert response_text not in caplog.text


def test_unexpected_top_level_key_text_is_redacted_from_diagnostics():
    key = "MODEL_KEY_SENTINEL with unsafe text"

    assert llm_service._top_level_keys({"competitor_analysis": {}, key: {}}) == [
        "<non_schema_key>", "competitor_analysis"
    ]


def test_incorrect_gemini_field_type_is_logged_by_field_name(monkeypatch, caplog):
    analysis = _valid_analysis()
    analysis["seo_analysis"]["seo_title"] = 27
    response_text = json.dumps(analysis)
    _mock_gemini(monkeypatch, response_text)

    with caplog.at_level("WARNING", logger="app.services.llm_service"):
        with pytest.raises(ExternalServiceError) as raised:
            _run_analysis()

    record = _failure_record(caplog)
    assert record.failure_category == "invalid_field_type"
    assert record.missing_fields == []
    assert record.invalid_fields == ["seo_analysis.seo_title"]
    assert record.expected_field_types == {"seo_analysis.seo_title": "string"}
    assert raised.value.code == "gemini_invalid_response"
    assert raised.value.status_code == 502
    assert response_text not in caplog.text


def test_malformed_gemini_json_is_logged_without_raw_response(monkeypatch, caplog):
    response_text = "RESPONSE_SENTINEL { malformed"
    _mock_gemini(monkeypatch, response_text)

    with caplog.at_level("WARNING", logger="app.services.llm_service"):
        with pytest.raises(ExternalServiceError) as raised:
            _run_analysis()

    record = _failure_record(caplog)
    assert record.failure_category == "malformed_json"
    assert record.missing_fields == []
    assert record.invalid_fields == []
    assert record.json_line == 1
    assert record.json_column > 0
    assert record.json_parse_failed is True
    assert record.top_level_keys == []
    assert raised.value.code == "gemini_invalid_response"
    assert raised.value.status_code == 502
    assert raised.value.public_message == (
        "Gemini returned an invalid analysis response. Please try again."
    )
    assert response_text not in caplog.text
    assert "WEBSITE_CONTENT_SENTINEL" not in caplog.text


def test_malformed_json_is_retried_and_valid_retry_succeeds(monkeypatch, caplog):
    analysis = _valid_analysis()
    fake_models = _mock_gemini(
        monkeypatch,
        ["{ malformed", json.dumps(analysis)],
    )

    with caplog.at_level("WARNING", logger="app.services.llm_service"):
        result = _run_analysis()

    assert result["competitor_analysis"]["brand_name"] == "Example CRM"
    assert fake_models.calls == 2
    assert any(
        getattr(record, "diagnostic_event", None) == "gemini.response_validation_failed"
        and getattr(record, "failure_category", None) == "malformed_json"
        for record in caplog.records
    )


def test_max_tokens_json_truncation_is_logged_and_not_retried(monkeypatch, caplog):
    fake_models = _mock_gemini(
        monkeypatch,
        "{\"competitor_analysis\":",
        metadata={
            "candidates": [SimpleNamespace(finish_reason="MAX_TOKENS")],
            "usage_metadata": SimpleNamespace(
                prompt_token_count=50000,
                candidates_token_count=8192,
                total_token_count=58192,
            ),
        },
    )

    with caplog.at_level("WARNING", logger="app.services.llm_service"):
        with pytest.raises(ExternalServiceError) as raised:
            _run_analysis()

    record = _failure_record(caplog)
    assert fake_models.calls == 1
    assert record.finish_reason == "MAX_TOKENS"
    assert record.json_parse_failed is True
    assert record.response_metadata["usage"]["candidates_token_count"] == 8192
    assert raised.value.code == "gemini_incomplete_output"
    assert "MAX_TOKENS" in caplog.text
    assert "competitor_analysis\":\"" not in caplog.text


def test_schema_failure_gets_one_field_level_correction_retry(monkeypatch):
    incomplete = _valid_analysis()
    del incomplete["competitor_analysis"]["brand_name"]
    fake_models = _mock_gemini(
        monkeypatch,
        [json.dumps(incomplete), json.dumps(_valid_analysis())],
    )

    result = _run_analysis()

    assert result["competitor_analysis"]["brand_name"] == "Example CRM"
    assert fake_models.calls == 2
    assert '"missing_fields":["competitor_analysis.brand_name"]' in fake_models.request_contents[1]


def test_finish_reason_int_enum_is_logged_by_name():
    class FinishReason(IntEnum):
        MAX_TOKENS = 2

    finish_reason, metadata = llm_service._gemini_response_diagnostics(SimpleNamespace(
        candidates=[SimpleNamespace(finish_reason=FinishReason.MAX_TOKENS)],
        usage_metadata=None,
    ))

    assert finish_reason == "MAX_TOKENS"
    assert metadata["candidate_count"] == 1


def test_valid_gemini_analysis_succeeds_without_failure_diagnostic(monkeypatch, caplog):
    analysis = _valid_analysis()
    response_text = json.dumps(analysis)
    fake_models = _mock_gemini(monkeypatch, response_text)

    with caplog.at_level("WARNING", logger="app.services.llm_service"):
        result = _run_analysis()

    assert result["competitor_analysis"]["brand_name"] == "Example CRM"
    assert len(result["target_search_queries"]) == 1
    assert len(result["top_services"]) == 5
    assert result["seo_analysis"]["metadata_assessment"]["title_tag"]["observed"] == "Example"
    assert result["seo_analysis"]["metadata_assessment"]["meta_description"]["status"] == "missing"
    assert result["seo_analysis"]["seo_score"]["max_score"] == 100
    prompt = fake_models.request_contents[0]
    assert "products, services, business types" in prompt
    assert "Mark directly stated" in prompt and "as observed" in prompt
    for field in (
        "competitor_analysis", "target_search_queries", "seo_analysis",
        "primary_keywords", "branded_product_keywords", "keyword_intents",
        "content_gaps", "prioritized_recommendations", "top_services",
        "target_audience", "industry_segment_evidence", "audience_evidence",
        "brand_name", "overview", "category", "products_services",
        "competitors", "brand_advantages", "positioning_opportunities",
        "query", "platform", "platform_type", "intent", "relevance",
        "target_audience", "seo_title", "meta_description",
        "content_recommendations", "name", "description", "target_industry",
        "target_customer", "business_benefit", "inferred", "industry_segments",
        "business_types", "customer_profiles", "geographic_focus",
        "needs_and_challenges", "interests", "recommended_marketing_segments",
    ):
        assert f'"{field}"' in prompt
    assert not any(
        getattr(record, "diagnostic_event", None)
        == "gemini.response_validation_failed"
        for record in caplog.records
    )


def test_valid_response_with_phase_five_fields_is_preserved(monkeypatch):
    analysis = _valid_analysis()
    analysis["seo_analysis"].update({
        "branded_product_keywords": ["CRM software"],
        "content_gaps": [],
        "prioritized_recommendations": [],
    })
    analysis["seo_analysis"]["keyword_intents"]["branded_products"] = [{
        "keyword": "CRM software",
        "intent": "navigational",
    }]
    _mock_gemini(monkeypatch, json.dumps(analysis))

    result = _run_analysis({
        "title": "Example CRM software",
        "content": "WEBSITE_CONTENT_SENTINEL",
    })

    assert result["seo_analysis"]["branded_product_keywords"] == ["CRM software"]
    assert result["seo_analysis"]["keyword_intents"]["branded_products"] == [{
        "keyword": "CRM software",
        "intent": "navigational",
    }]


def test_invalid_query_relevance_is_repaired_without_retry_or_analysis_failure(monkeypatch):
    analysis = _valid_analysis()
    analysis["target_search_queries"][0]["relevance"] = "Relevant to this business."
    fake_models = _mock_gemini(monkeypatch, json.dumps(analysis))

    result = _run_analysis()

    assert fake_models.calls == 1
    assert result["target_search_queries"][0]["relevance"] == (
        "The query mentions CRM, a topic listed in the business profile."
    )


def test_legacy_valid_response_defaults_missing_additive_fields(monkeypatch):
    analysis = _valid_analysis()
    seo = analysis["seo_analysis"]
    for field in ("branded_product_keywords", "content_gaps", "prioritized_recommendations", "keyword_intents"):
        seo.pop(field, None)
    analysis["target_search_queries"][0].pop("platform_type")
    analysis["target_audience"].pop("industry_segment_evidence")
    analysis["target_audience"].pop("audience_evidence")
    _mock_gemini(monkeypatch, json.dumps(analysis))

    result = _run_analysis()

    seo_result = result["seo_analysis"]
    assert seo_result["branded_product_keywords"] == []
    assert seo_result["content_gaps"] == []
    assert seo_result["prioritized_recommendations"] == []
    assert seo_result["keyword_intents"] == {
        "primary": [],
        "secondary": [],
        "long_tail": [],
        "business": [],
        "branded_products": [],
    }
    assert result["target_search_queries"][0]["platform_type"] == "search_engine"
    assert result["target_audience"]["industry_segment_evidence"] == []
    assert result["target_audience"]["audience_evidence"] == []


def test_legacy_valid_response_keeps_existing_api_contract(monkeypatch):
    analysis = _valid_analysis()
    seo = analysis["seo_analysis"]
    for field in ("branded_product_keywords", "content_gaps", "prioritized_recommendations", "keyword_intents"):
        seo.pop(field, None)
    analysis["target_search_queries"][0].pop("platform_type")
    analysis["target_audience"].pop("industry_segment_evidence")
    analysis["target_audience"].pop("audience_evidence")
    _mock_gemini(monkeypatch, json.dumps(analysis))

    async def extract(url):
        return {
            "url": url,
            "title": "Example",
            "meta_description": "",
            "content": "Example business content.",
        }

    async def no_competitors(*_args):
        return []

    monkeypatch.setattr(main, "extract_website_content", extract)
    monkeypatch.setattr(main, "discover_competitors", no_competitors)
    response = TestClient(main.app).post(
        "/link-analysis",
        json={"url": "https://example.com/"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert set(("competitor_analysis", "target_search_queries", "seo_analysis", "top_services", "target_audience")) <= set(body["analysis"])
    assert body["analysis"]["seo_analysis"]["branded_product_keywords"] == []


def test_present_optional_seo_fields_still_reject_incorrect_types():
    analysis = _valid_analysis()
    analysis["seo_analysis"]["branded_product_keywords"] = "Example CRM"
    with pytest.raises(ValueError, match="branded_product_keywords must be a list"):
        llm_service.validate_analysis(analysis, website_data={"title": "Example"})

    analysis = _valid_analysis()
    analysis["seo_analysis"]["content_gaps"] = {}
    with pytest.raises(ValueError, match="content_gaps must be a list"):
        llm_service.validate_analysis(analysis, website_data={"title": "Example"})

    analysis = _valid_analysis()
    analysis["seo_analysis"] = []
    with pytest.raises(ValueError, match="Invalid SEO analysis format"):
        llm_service.validate_analysis(analysis, website_data={"title": "Example"})


def test_gemini_validation_error_keeps_structured_api_response(monkeypatch):
    analysis = _valid_analysis()
    del analysis["target_audience"]
    response_text = json.dumps(analysis)
    _mock_gemini(monkeypatch, response_text)

    async def extract(_url):
        return {
            "url": "https://example.com/",
            "title": "Example",
            "meta_description": "",
            "content": "Example business content. " * 20,
        }

    monkeypatch.setattr(main, "extract_website_content", extract)
    monkeypatch.setattr(main, "analyze_with_ai", llm_service.analyze_with_ai)
    response = TestClient(main.app).post(
        "/link-analysis",
        json={"url": "https://example.com/"},
    )

    assert response.status_code == 502
    assert response.json() == {
        "success": False,
        "error": {
            "code": "gemini_invalid_response",
            "message": "Gemini returned an incomplete analysis response. Please try again.",
        },
        "detail": "Gemini returned an incomplete analysis response. Please try again.",
    }
