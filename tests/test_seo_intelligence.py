import pytest

from app.services.llm_service import (
    build_metadata_assessment,
    build_on_page_seo_score,
    filter_supported_industry_segments,
    ground_audience_claims,
    validate_search_queries,
    validate_seo_analysis,
)


def _seo_payload():
    return {
        "primary_keywords": [],
        "secondary_keywords": [],
        "long_tail_keywords": [],
        "business_keywords": [],
        "branded_product_keywords": [],
        "seo_title": "Suggested page title",
        "meta_description": "Suggested meta description",
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
    }


def test_keyword_analysis_removes_generic_and_duplicate_keywords():
    seo = _seo_payload()
    seo["primary_keywords"] = ["Software", "CRM platform", "CRM platform"]
    seo["secondary_keywords"] = ["crm-platform", "customer relationship management"]
    seo["keyword_intents"] = {
        "primary": [
            {"keyword": "Software", "intent": "informational"},
            {"keyword": "CRM platform", "intent": "commercial"},
        ],
        "secondary": [
            {"keyword": "crm-platform", "intent": "transactional"},
            {"keyword": "customer relationship management", "intent": "commercial"},
        ],
        "long_tail": [],
        "business": [],
        "branded_products": [],
    }

    result = validate_seo_analysis(seo)

    assert result["primary_keywords"] == ["CRM platform"]
    assert result["secondary_keywords"] == ["customer relationship management"]
    assert result["keyword_intents"]["primary"] == [
        {"keyword": "CRM platform", "intent": "commercial"}
    ]
    assert result["keyword_intents"]["secondary"] == [
        {"keyword": "customer relationship management", "intent": "commercial"}
    ]


def test_keyword_intent_requires_supported_enum_and_matching_keyword():
    seo = _seo_payload()
    seo["primary_keywords"] = ["inventory management software"]
    seo["keyword_intents"]["primary"] = [
        {"keyword": "inventory management software", "intent": "high volume"}
    ]

    with pytest.raises(ValueError, match="invalid search intent"):
        validate_seo_analysis(seo)

    seo["keyword_intents"]["primary"] = [
        {"keyword": "inventory management software", "intent": "commercial"}
    ]
    seo["keyword_intents"]["primary"] = []
    with pytest.raises(ValueError, match="missing an intent"):
        validate_seo_analysis(seo)


def test_shopify_named_products_are_separated_from_generic_keyword_groups():
    source = {
        "title": "Shopify ecommerce platform tools",
        "content": "Shopify Capital provides funding and Shop Pay helps customers check out.",
    }
    seo = _seo_payload()
    seo["primary_keywords"] = ["Shop Pay", "Shopify ecommerce platform"]
    seo["business_keywords"] = ["Shopify Capital", "ecommerce platform"]
    seo["keyword_intents"]["primary"] = [
        {"keyword": "Shop Pay", "intent": "navigational"},
        {"keyword": "Shopify ecommerce platform", "intent": "commercial"},
    ]
    seo["keyword_intents"]["business"] = [
        {"keyword": "Shopify Capital", "intent": "transactional"},
        {"keyword": "ecommerce platform", "intent": "commercial"},
    ]

    result = validate_seo_analysis(
        seo,
        brand_name="Shopify",
        business_context="ecommerce platform",
        product_names=["Shop Pay", "Shopify Capital", "Shopify Markets"],
        website_data=source,
    )

    assert result["primary_keywords"] == ["Shopify ecommerce platform"]
    assert result["business_keywords"] == ["ecommerce platform"]
    assert result["branded_product_keywords"] == ["Shop Pay", "Shopify Capital"]
    assert [item["keyword"] for item in result["keyword_intents"]["branded_products"]] == [
        "Shop Pay", "Shopify Capital"
    ]


def test_search_platforms_distinguish_social_from_search_engine_and_check_cues():
    profile = {
        "brand_name": "Shopify",
        "category": "ecommerce platform",
        "products_services": ["online store"],
    }
    source = {"title": "Shopify ecommerce platform social content"}
    valid = [{
        "query": "How Shopify ecommerce platform supports social content",
        "platform": "Instagram",
        "platform_type": "social_network",
        "intent": "informational",
        "relevance": "How Shopify supports ecommerce social content.",
        "target_audience": "online merchants",
    }]
    result = validate_search_queries(valid, profile, source)
    assert result[0]["platform_type"] == "social_network"

    invalid = [{**valid[0], "platform_type": "search_engine"}]
    with pytest.raises(ValueError, match="platform_type is invalid"):
        validate_search_queries(invalid, profile, source)

    invalid = [{**valid[0], "query": "How Shopify ecommerce platform works", "relevance": "How Shopify supports ecommerce."}]
    with pytest.raises(ValueError, match="platform-specific content cue"):
        validate_search_queries(invalid, profile, source)


def test_industry_segments_require_relevant_page_evidence():
    website = {"content": "Shopify ecommerce platform helps merchants create an online store and sell products."}
    industries, evidence = filter_supported_industry_segments(
        ["Ecommerce", "Healthcare"],
        [
            {"industry": "Ecommerce", "basis": "inferred", "supporting_text": "Shopify ecommerce platform helps merchants create an online store and sell products.", "reason": "The extracted text identifies the product as an ecommerce platform."},
            {"industry": "Healthcare", "basis": "inferred", "supporting_text": "Shopify ecommerce platform helps merchants create an online store and sell products.", "reason": "Unsupported broad guess."},
        ],
        website,
    )

    assert industries == ["Ecommerce"]
    assert evidence[0]["basis"] == "inferred"


def test_directly_observed_audience_claims_survive_missing_evidence_objects():
    website = {
        "content": (
            "Shopify helps entrepreneurs and small businesses start selling online. "
            "Merchants can manage orders and inventory from one place."
        )
    }
    audience = {
        "primary_audience": "entrepreneurs and small businesses",
        "secondary_audience": "Not available",
        "business_types": ["small businesses", "healthcare providers"],
        "customer_profiles": ["merchants", "Fortune 500 retailers"],
        "geographic_focus": [],
        "needs_and_challenges": ["manage orders and inventory", "reduce tax liability"],
        "interests": [],
        "recommended_marketing_segments": [],
        "audience_evidence": [],
    }

    result = ground_audience_claims(audience, website)

    assert result["primary_audience"] == "entrepreneurs and small businesses"
    assert result["business_types"] == ["small businesses"]
    assert result["customer_profiles"] == ["merchants"]
    assert result["needs_and_challenges"] == ["manage orders and inventory"]
    assert {item["basis"] for item in result["audience_evidence"]} == {"observed"}
    assert all(item["supporting_text"] in website["content"] for item in result["audience_evidence"])


def test_industry_segments_directly_named_on_page_need_no_model_evidence_object():
    industries, evidence = filter_supported_industry_segments(
        ["Retail", "Healthcare"],
        [],
        {"content": "Shopify supports retail businesses selling online."},
    )

    assert industries == ["Retail"]
    assert evidence == [{
        "industry": "Retail",
        "basis": "observed",
        "supporting_text": "retail",
        "reason": "",
    }]


def test_validated_audience_evidence_populates_empty_fields():
    website = {
        "content": (
            "Shopify supports retailers and enterprise brands. "
            "Merchants can manage orders from one place."
        )
    }
    audience = {
        "primary_audience": "Not available",
        "secondary_audience": "Not available",
        "business_types": [],
        "customer_profiles": [],
        "geographic_focus": [],
        "needs_and_challenges": [],
        "interests": [],
        "recommended_marketing_segments": [],
        "audience_evidence": [
            {
                "field": "primary_audience",
                "claim": "retailers",
                "basis": "observed",
                "supporting_text": "Shopify supports retailers",
                "reason": "",
            },
            {
                "field": "secondary_audience",
                "claim": "enterprise brands",
                "basis": "inferred",
                "supporting_text": "enterprise brands",
                "reason": "The page identifies enterprise brands as an audience segment.",
            },
            {
                "field": "needs_and_challenges",
                "claim": "manage orders",
                "basis": "observed",
                "supporting_text": "Merchants can manage orders",
                "reason": "",
            },
        ],
    }

    result = ground_audience_claims(audience, website)

    assert result["primary_audience"] == "retailers"
    assert result["secondary_audience"] == "enterprise brands"
    assert result["needs_and_challenges"] == ["manage orders"]
    assert [item["basis"] for item in result["audience_evidence"]] == [
        "observed", "inferred", "observed"
    ]


def test_industry_evidence_populates_empty_segment_field():
    industries, evidence = filter_supported_industry_segments(
        [],
        [{
            "industry": "Retail",
            "basis": "inferred",
            "supporting_text": "Retail merchants use Shopify",
            "reason": "The page identifies retail businesses as a use case.",
        }],
        {"content": "Retail merchants use Shopify."},
    )

    assert industries == ["Retail"]
    assert evidence[0]["basis"] == "inferred"


def test_seo_recommendations_must_reference_a_real_page_issue_or_gap():
    seo = _seo_payload()
    seo["content_gaps"] = [{
        "topic": "Shopify Markets setup",
        "why_it_matters": "Users may need help configuring international selling.",
        "content_opportunity": "Add a setup guide.",
    }]
    seo["prioritized_recommendations"] = [
        {
            "recommendation": "Improve the page title.",
            "priority": "High",
            "expected_benefit": "Clarifies page topic.",
            "implementation_effort": "Low",
            "related_to": "title_tag",
            "related_topic": "",
        },
        {
            "recommendation": "Write a Shopify Markets setup guide.",
            "priority": "Medium",
            "expected_benefit": "Addresses the setup information gap.",
            "implementation_effort": "Medium",
            "related_to": "content_gap",
            "related_topic": "Shopify Markets setup",
        },
        {
            "recommendation": "Add generic SEO content.",
            "priority": "Low",
            "expected_benefit": "Could help the site.",
            "implementation_effort": "Low",
            "related_to": "content_gap",
            "related_topic": "Unidentified topic",
        },
        {
            "recommendation": "Add generic SEO content.",
            "priority": "Low",
            "expected_benefit": "Could help the site.",
            "implementation_effort": "Low",
            "related_to": "content_gap",
            "related_topic": "Shopify Markets setup",
        },
    ]
    page = {"title": "Short", "meta_description": ""}

    result = validate_seo_analysis(seo, website_data=page)

    assert [item["related_to"] for item in result["prioritized_recommendations"]] == [
        "title_tag", "content_gap"
    ]
    assert result["content_recommendations"] == [
        "Improve the page title.", "Write a Shopify Markets setup guide."
    ]

def test_prioritized_recommendations_and_content_gaps_are_validated():
    seo = _seo_payload()
    seo["content_gaps"] = [{
        "topic": "Product onboarding",
        "why_it_matters": "It addresses a common decision-stage question.",
        "content_opportunity": "Publish a getting-started guide.",
    }]
    seo["prioritized_recommendations"] = [{
        "recommendation": "Add a descriptive page title.",
        "priority": "High",
        "expected_benefit": "Makes the page topic clearer to visitors and crawlers.",
        "implementation_effort": "Low",
        "related_to": "title_tag",
        "related_topic": "",
    }]

    result = validate_seo_analysis(seo)
    assert result["content_gaps"][0]["topic"] == "Product onboarding"
    assert result["prioritized_recommendations"][0]["priority"] == "High"

    seo["prioritized_recommendations"][0]["priority"] = "Urgent"
    with pytest.raises(ValueError, match="priority must be"):
        validate_seo_analysis(seo)


def test_missing_observed_metadata_is_reported_without_failing():
    metadata = build_metadata_assessment(
        {"title": "", "meta_description": None},
        {"seo_title": "Suggested title", "meta_description": "Suggested description"},
    )

    assert metadata["title_tag"] == {
        "observed": "",
        "status": "missing",
        "character_count": 0,
        "suggestion": "Suggested title",
    }
    assert metadata["meta_description"]["status"] == "missing"
    assert metadata["meta_description"]["suggestion"] == "Suggested description"


def test_on_page_score_is_deterministic_explainable_and_not_a_google_score():
    score = build_on_page_seo_score({
        "title": "A descriptive title that is exactly long enough for SEO",
        "meta_description": "A clear description of the product and the value it provides to potential customers across its main use cases.",
        "headings": ["Overview", "Features"],
        "content": "useful " * 500,
    })

    assert score["score"] == 100
    assert score["max_score"] == 100
    assert [part["score"] for part in score["components"]] == [25, 25, 20, 30]
    assert "not a Google score" in score["methodology"]


def test_on_page_score_handles_missing_metadata_and_content_safely():
    score = build_on_page_seo_score({"title": None, "meta_description": None})

    assert score["score"] == 0
    assert score["components"][0]["finding"] == "Missing"
    assert score["components"][1]["finding"] == "Missing"
    assert score["components"][2]["finding"] == "0 extracted heading(s) observed."
