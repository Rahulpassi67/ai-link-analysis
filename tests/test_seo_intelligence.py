import pytest

from app.services.llm_service import (
    build_metadata_assessment,
    build_on_page_seo_score,
    consolidate_service_audience_evidence,
    filter_supported_industry_segments,
    ground_audience_claims,
    validate_search_queries,
    validate_service_audience_claims,
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


def test_invalid_search_relevance_is_repaired_or_query_is_omitted():
    profile = {
        "brand_name": "Shopify",
        "category": "ecommerce platform",
        "products_services": ["online store"],
    }
    source = {
        "title": "Shopify ecommerce platform social content",
        "content": "Shopify supports social content for online stores.",
    }
    queries = [
        {
            "query": "How Shopify ecommerce platform supports social content",
            "platform": "Instagram",
            "platform_type": "social_network",
            "intent": "informational",
            "relevance": "Relevant to this business.",
            "target_audience": "online merchants",
        },
        {
            "query": "How Acme supports social content",
            "platform": "LinkedIn",
            "platform_type": "social_network",
            "intent": "informational",
            "relevance": "Relevant to this business.",
            "target_audience": "online merchants",
        },
    ]

    result = validate_search_queries(queries, profile, source)

    assert len(result) == 1
    assert result[0]["relevance"] == (
        "The query mentions ecommerce, a topic listed in the business profile."
    )


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
        "supporting_text": "Shopify supports retail businesses selling online.",
        "reason": "",
    }]


def test_industry_filter_rejects_departments_and_retains_real_industries():
    quote = "The Company OS serves engineering, sales, customer support and finance teams in healthcare organizations."
    labels = ("Engineering", "Sales", "Customer Support", "Finance", "Healthcare")
    industries, evidence = filter_supported_industry_segments(
        list(labels),
        [{
            "industry": label,
            "basis": "inferred",
            "supporting_text": quote,
            "reason": "The page identifies this market.",
        } for label in labels],
        {"content": quote},
    )

    assert industries == ["Healthcare"]
    assert [item["industry"] for item in evidence] == ["Healthcare"]


def test_industry_filter_does_not_restore_word_only_labels():
    quote = "The Company OS brings engineering, sales, customer support and finance teams together."
    industries, evidence = filter_supported_industry_segments(
        ["Engineering", "Sales", "Customer Support", "Finance"],
        [],
        {"content": quote},
    )

    assert industries == []
    assert evidence == []


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
                "supporting_text": "Shopify supports retailers and enterprise brands.",
                "reason": "The page explicitly connects the product to enterprise brands.",
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


def test_audience_range_is_preserved_without_creating_endpoint_profile():
    quote = "For anyone from entrepreneurs to enterprise."
    audience = {
        "primary_audience": "Enterprise",
        "secondary_audience": "Not available",
        "business_types": ["Enterprise"],
        "customer_profiles": ["Enterprise"],
        "geographic_focus": [],
        "needs_and_challenges": [],
        "interests": [],
        "recommended_marketing_segments": ["Enterprise"],
        "audience_evidence": [
            {
                "field": "primary_audience", "claim": "Enterprise",
                "basis": "observed", "supporting_text": quote, "reason": "",
            },
            {
                "field": "customer_profiles", "claim": "Enterprise",
                "basis": "observed", "supporting_text": quote, "reason": "",
            },
            {
                "field": "business_types", "claim": "Enterprise",
                "basis": "observed", "supporting_text": quote, "reason": "",
            },
            {
                "field": "recommended_marketing_segments", "claim": "Enterprise",
                "basis": "observed", "supporting_text": quote, "reason": "",
            },
        ],
    }

    result = ground_audience_claims(audience, {"content": quote})

    assert result["primary_audience"] == "entrepreneurs to enterprise"
    assert result["secondary_audience"] == "Not available"
    assert result["business_types"] == []
    assert result["customer_profiles"] == []
    assert result["recommended_marketing_segments"] == []
    assert len(result["audience_evidence"]) == 1
    assert result["audience_evidence"][0]["field"] == "primary_audience"


def test_inferred_audience_claim_needs_shared_terms_with_its_source_quote():
    website = {
        "content": "HubSpot marketing automation helps small marketing teams manage campaigns."
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
                "claim": "small marketing teams",
                "basis": "inferred",
                "supporting_text": "HubSpot marketing automation helps small marketing teams",
                "reason": "The page describes the product as helping these teams manage campaigns.",
            },
            {
                "field": "secondary_audience",
                "claim": "enterprise finance departments",
                "basis": "inferred",
                "supporting_text": "HubSpot marketing automation helps small marketing teams",
                "reason": "This audience may use business software.",
            },
        ],
    }

    result = ground_audience_claims(audience, website)

    assert result["primary_audience"] == "small marketing teams"
    assert result["secondary_audience"] == "Not available"
    assert [item["basis"] for item in result["audience_evidence"]] == ["inferred"]


def test_audience_claim_must_be_connected_to_relation_not_nearby_by_chance():
    website = {
        "content": (
            "Acme offers tools for small teams. Enterprise finance departments "
            "are mentioned elsewhere in the page."
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
        "audience_evidence": [{
            "field": "primary_audience",
            "claim": "enterprise finance departments",
            "basis": "observed",
            "supporting_text": "Acme offers tools for small teams. Enterprise finance departments are mentioned elsewhere in the page.",
            "reason": "",
        }],
    }

    result = ground_audience_claims(audience, website)

    assert result["primary_audience"] == "Not available"
    assert result["audience_evidence"] == []


def test_natural_language_audience_description_is_accepted():
    quote = "Independent retailers make up most of our customer base."
    audience = {
        "primary_audience": "independent retailers",
        "secondary_audience": "Not available",
        "business_types": [],
        "customer_profiles": [],
        "geographic_focus": [],
        "needs_and_challenges": [],
        "interests": [],
        "recommended_marketing_segments": [],
        "audience_evidence": [],
    }

    result = ground_audience_claims(audience, {"content": quote})

    assert result["primary_audience"] == "independent retailers"
    assert result["audience_evidence"][0]["supporting_text"] == quote


def test_short_audience_quotes_are_expanded_to_one_exact_source_sentence():
    source_sentence = "Acme serves independent retailers and small agencies."
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
                "claim": "independent retailers",
                "basis": "observed",
                "supporting_text": "independent retailers",
                "reason": "",
            },
            {
                "field": "business_types",
                "claim": "small agencies",
                "basis": "inferred",
                "supporting_text": "small agencies",
                "reason": "The page explicitly identifies these businesses as customers.",
            },
        ],
    }

    result = ground_audience_claims(audience, {"content": source_sentence})

    assert result["primary_audience"] == "independent retailers"
    assert result["business_types"] == ["small agencies"]
    assert [item["basis"] for item in result["audience_evidence"]] == ["observed", "inferred"]
    assert all(item["supporting_text"] == source_sentence for item in result["audience_evidence"])


def test_concatenated_multisentence_audience_evidence_is_replaced_with_clean_source_text():
    source_sentence = "Acme serves independent retailers and small agencies."
    audience = {
        "primary_audience": "independent retailers",
        "secondary_audience": "Not available",
        "business_types": [],
        "customer_profiles": [],
        "geographic_focus": [],
        "needs_and_challenges": [],
        "interests": [],
        "recommended_marketing_segments": [],
        "audience_evidence": [{
            "field": "primary_audience",
            "claim": "independent retailers",
            "basis": "observed",
            "supporting_text": (
                "Home Pricing Careers. " + source_sentence + " Privacy Contact."
            ),
            "reason": "",
        }],
    }

    result = ground_audience_claims(audience, {"content": source_sentence})

    assert result["primary_audience"] == "independent retailers"
    assert len(result["audience_evidence"]) == 1
    assert result["audience_evidence"][0]["supporting_text"] == source_sentence
    assert "Home Pricing" not in result["audience_evidence"][0]["supporting_text"]


def test_audience_claims_remain_unavailable_without_reliable_source_evidence():
    audience = {
        "primary_audience": "Retail teams",
        "secondary_audience": "Enterprise buyers",
        "business_types": ["Retail businesses"],
        "customer_profiles": ["Retail managers"],
        "geographic_focus": ["North America"],
        "needs_and_challenges": ["Reduce operating costs"],
        "interests": ["Automation"],
        "recommended_marketing_segments": ["Enterprise retailers"],
        "audience_evidence": [{
            "field": "primary_audience",
            "claim": "Retail teams",
            "basis": "inferred",
            "supporting_text": "Retail teams use our service.",
            "reason": "The model inferred this audience.",
        }],
    }
    website = {"title": "", "meta_description": "", "content": ""}

    result = ground_audience_claims(audience, website)
    industries, industry_evidence = filter_supported_industry_segments(
        ["Retail"],
        [{
            "industry": "Retail",
            "basis": "inferred",
            "supporting_text": "Retail is a relevant industry.",
            "reason": "The model inferred this industry.",
        }],
        website,
    )

    assert result["primary_audience"] == "Not available"
    assert result["secondary_audience"] == "Not available"
    for field in (
        "business_types", "customer_profiles", "geographic_focus",
        "needs_and_challenges", "interests", "recommended_marketing_segments",
    ):
        assert result[field] == []
    assert result["audience_evidence"] == []
    assert industries == []
    assert industry_evidence == []


def test_service_audience_claims_require_service_specific_source_text():
    services = [{
        "name": "Project Hub",
        "description": "Project planning software",
        "target_customer": "Finance teams",
        "target_industry": "Healthcare",
        "business_benefit": "Manage workflows",
        "inferred": False,
    }]
    website = {
        "content": (
            "Project Hub helps project managers organize work. "
            "The Company OS also discusses finance teams and healthcare trends."
        )
    }

    result = validate_service_audience_claims(services, website)

    assert result[0]["target_customer"] == "Not available"
    assert result[0]["target_industry"] == "Not available"


def test_service_claim_words_cannot_supply_their_own_context_overlap():
    services = [{
        "name": "Shopify Plus",
        "description": "Enterprise commerce platform",
        "target_customer": "Enterprise",
        "target_industry": "Enterprise",
        "business_benefit": "Scale operations",
        "inferred": False,
    }]
    website = {"content": "For anyone from entrepreneurs to enterprise."}

    result = validate_service_audience_claims(services, website)

    assert result[0]["target_customer"] == "Not available"
    assert result[0]["target_industry"] == "Not available"


def test_service_fields_keep_offering_specific_evidence_and_drop_unsupported_numbers():
    services = [{
        "name": "Shopify POS",
        "description": "Point of sale tools for in-person payments.",
        "target_customer": "retail businesses",
        "target_industry": "Retail",
        "business_benefit": "Sell in person 50% faster.",
        "inferred": False,
    }]
    website = {
        "content": (
            "Shopify POS provides point of sale tools for in-person payments for retail businesses."
        )
    }

    result = validate_service_audience_claims(services, website)

    assert result[0]["name"] == "Shopify POS"
    assert result[0]["description"] == "Point of sale tools for in-person payments."
    assert result[0]["target_customer"] == "retail businesses"
    assert result[0]["target_industry"] == "Retail"
    assert result[0]["business_benefit"] == "Not available"


def test_service_customer_and_industry_evidence_populates_audience_fields():
    website = {
        "content": (
            "Marketing Hub helps marketing professionals plan campaigns. "
            "Sales Hub supports sales teams. "
            "HubSpot serves B2B software companies. "
            "Startups, small businesses and enterprise teams use HubSpot."
        )
    }
    audience = {
        "primary_audience": "Not available",
        "secondary_audience": "Not available",
        "industry_segments": [],
        "industry_segment_evidence": [],
        "business_types": ["Startups", "small businesses", "enterprise teams"],
        "customer_profiles": [],
        "geographic_focus": [],
        "needs_and_challenges": [],
        "interests": [],
        "recommended_marketing_segments": [],
        "audience_evidence": [],
    }
    services = [
        {
            "name": "Marketing Hub",
            "description": "Campaign planning tools",
            "target_customer": "Marketing Professionals",
            "target_industry": "B2B software companies",
            "inferred": False,
        },
        {
            "name": "Sales Hub",
            "description": "Sales team tools",
            "target_customer": "Sales Teams",
            "target_industry": "B2B software companies",
            "inferred": False,
        },
        {
            "name": "Suggested Finance Hub",
            "description": "Finance tools",
            "target_customer": "Finance teams",
            "target_industry": "Financial services",
            "inferred": True,
        },
    ]

    consolidate_service_audience_evidence(audience, services, website)
    audience["industry_segments"], audience["industry_segment_evidence"] = (
        filter_supported_industry_segments(
            audience["industry_segments"],
            audience["industry_segment_evidence"],
            website,
        )
    )
    result = ground_audience_claims(audience, website)

    assert result["primary_audience"] == "Marketing Professionals"
    assert result["secondary_audience"] == "Sales Teams"
    assert result["industry_segments"] == ["B2B software companies"]
    assert "Marketing Professionals" in result["customer_profiles"]
    assert "Startups" in result["customer_profiles"]
    assert "Sales Teams" in result["recommended_marketing_segments"]
    assert "Finance teams" not in result["customer_profiles"]
    assert {item["basis"] for item in result["audience_evidence"]} == {"observed"}


def test_repeated_audience_evidence_is_merged_without_losing_fields_or_provenance():
    quote = "HubSpot supports marketing professionals."
    website = {"content": quote}
    audience = {
        "primary_audience": "marketing professionals",
        "secondary_audience": "Not available",
        "business_types": [],
        "customer_profiles": ["marketing professionals"],
        "geographic_focus": [],
        "needs_and_challenges": [],
        "interests": [],
        "recommended_marketing_segments": ["marketing professionals"],
        "audience_evidence": [
            {
                "field": field,
                "claim": "marketing professionals",
                "basis": "observed",
                "supporting_text": quote,
                "reason": "",
            }
            for field in (
                "primary_audience",
                "customer_profiles",
                "recommended_marketing_segments",
            )
        ] + [{
            "field": "recommended_marketing_segments",
            "claim": "marketing professionals",
            "basis": "inferred",
            "supporting_text": quote,
            "reason": "This is an AI inference from the described product use.",
        }],
    }

    result = ground_audience_claims(audience, website)

    assert len(result["audience_evidence"]) == 2
    observed = next(item for item in result["audience_evidence"] if item["basis"] == "observed")
    assert observed["field"] == (
        "primary_audience, customer_profiles, recommended_marketing_segments"
    )
    assert observed["supporting_text"] == quote
    assert any(item["basis"] == "inferred" for item in result["audience_evidence"])


def test_distinct_audience_quotes_are_preserved_while_exact_duplicates_merge():
    website = {
        "content": (
            "Acme supports retail teams. Acme also helps enterprise teams."
        )
    }
    audience = {
        "primary_audience": "retail teams",
        "secondary_audience": "enterprise teams",
        "business_types": [],
        "customer_profiles": [],
        "geographic_focus": [],
        "needs_and_challenges": [],
        "interests": [],
        "recommended_marketing_segments": [],
        "audience_evidence": [
            {
                "field": "primary_audience", "claim": "retail teams",
                "basis": "observed", "supporting_text": "Acme supports retail teams", "reason": "",
            },
            {
                "field": "customer_profiles", "claim": "retail teams",
                "basis": "observed", "supporting_text": "Acme supports retail teams", "reason": "",
            },
            {
                "field": "secondary_audience", "claim": "enterprise teams",
                "basis": "observed", "supporting_text": "Acme also helps enterprise teams", "reason": "",
            },
        ],
    }

    result = ground_audience_claims(audience, website)

    assert len(result["audience_evidence"]) == 2
    assert {item["supporting_text"] for item in result["audience_evidence"]} == {
        "Acme supports retail teams.", "Acme also helps enterprise teams."
    }


def test_unsupported_customer_needs_are_removed_without_inference():
    website = {"content": "Acme provides CRM software for sales teams."}
    audience = {
        "primary_audience": "sales teams",
        "secondary_audience": "Not available",
        "business_types": [],
        "customer_profiles": [],
        "geographic_focus": [],
        "needs_and_challenges": ["reduce tax liability"],
        "interests": [],
        "recommended_marketing_segments": [],
        "audience_evidence": [],
    }

    result = ground_audience_claims(audience, website)

    assert result["needs_and_challenges"] == []
    assert not any(item["field"] == "needs_and_challenges" for item in result["audience_evidence"])


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
