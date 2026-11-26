import asyncio

import app.services.competitor_discovery as discovery
from app.services.competitor_discovery import (
    classify_competitor_type_from_sources,
    deduplicate_companies,
    find_official_website,
    is_publisher_candidate,
    official_website_is_validated,
    rank_candidates_by_evidence,
    validate_company_evidence,
)


def _fetched_evidence_detail(name, classification="company_mentioned"):
    return [{
        "accessible": True,
        "classification": classification,
        "source_url": "https://evidence.example/page",
        "excerpts": [{
            "classification": classification,
            "source_url": "https://evidence.example/page",
            "matched_text": f"{name} was mentioned in fetched page text.",
        }],
    }]


PROFILE = {
    "brand_name": "Linkly AI",
    "category": "website intelligence",
    "products_services": ["competitor analysis software", "website intelligence"]
}


def test_zero_candidate_path_emits_visible_structured_summary(monkeypatch, caplog):
    monkeypatch.setattr(discovery, "require_api_key", lambda _name: "test-key")
    monkeypatch.setattr(discovery, "gemini_client", object())

    with caplog.at_level("INFO", logger="app.services.competitor_discovery"):
        result = asyncio.run(discovery.discover_competitors({"competitor_analysis": {}}))

    summaries = [
        record for record in caplog.records
        if getattr(record, "diagnostic_event", None) == "competitor_discovery.summary"
    ]
    assert result == []
    assert len(summaries) == 1
    assert summaries[0].discovered_competitor_count == 0
    assert summaries[0].zero_result_reason == "missing_profile"
    assert '"discovered_competitor_count":0' in summaries[0].message


def test_official_domain_requires_name_signal_in_registered_domain():
    results = [
        {
            "title": "Northstar Systems official site",
            "url": "https://northstar-systems.com/about",
            "content": "Northstar Systems builds analytics software."
        },
        {
            "title": "Northstar Systems review",
            "url": "https://reviews.example.com/northstar",
            "content": "A review of Northstar Systems."
        },
        {
            "title": "Northstar Systems company profile",
            "url": "https://www.crunchbase.com/organization/northstar-systems",
            "content": "Company profile."
        }
    ]

    assert find_official_website("Northstar Systems, Inc.", results) == "https://northstar-systems.com"


def test_official_domain_lookup_rejects_unrelated_host_with_matching_title():
    results = [{
        "title": "Northstar Systems official site",
        "url": "https://randompublisher.com/northstar-systems",
        "content": "Northstar Systems builds analytics software."
    }]

    assert find_official_website("Northstar Systems", results) is None


def test_official_domain_requires_accessible_page_that_mentions_company():
    details = [{
        "source_url": "https://northstar-systems.com/",
        "accessible": True,
        "company_mentioned": False,
        "classification": "no_relevant_evidence"
    }]
    assert not official_website_is_validated(
        "https://northstar-systems.com",
        details
    )
    details[0]["company_mentioned"] = True
    assert not official_website_is_validated(
        "https://northstar-systems.com",
        details
    )
    details[0]["classification"] = "company_and_product_mentioned"
    assert official_website_is_validated(
        "https://northstar-systems.com",
        details
    )
    details[0]["accessible"] = False
    assert not official_website_is_validated(
        "https://northstar-systems.com",
        details
    )


def test_company_name_validation_uses_boundaries_not_substrings():
    result = validate_company_evidence(
        {
            "name": "Ace",
            "product_name": "Ace CRM",
            "evidence_urls": ["https://example.org/story"]
        },
        [{
            "url": "https://example.org/story",
            "title": "A face recognition startup",
            "content": "The company sells security software."
        }],
        PROFILE
    )

    assert result["evidence_status"] == "unsupported"
    assert result["competitor_type"] == "unclear"


def test_unsupported_product_is_removed_and_type_remains_unclear():
    result = validate_company_evidence(
        {
            "name": "Northstar Systems",
            "product_name": "Orbit Pro",
            "description": "A fabricated product description.",
            "evidence_urls": ["https://northstar-systems.com/product"]
        },
        [{
            "url": "https://northstar-systems.com/product",
            "title": "Northstar Systems",
            "content": "Northstar Systems provides website intelligence and competitor analysis software."
        }],
        PROFILE
    )

    assert result["evidence_status"] == "company_and_relevant_market_supported"
    assert result["product_name"] == ""
    assert result["description"]
    assert result["competitor_type"] == "unclear"
    assert "do not establish" in result["competitor_type_reason"]


def test_unrelated_company_is_rejected_even_when_company_name_is_real():
    result = validate_company_evidence(
        {
            "name": "Northstar Systems",
            "product_name": "AeroNav",
            "evidence_urls": ["https://northstar-aero.example/product"]
        },
        [{
            "url": "https://northstar-aero.example/product",
            "title": "Northstar Systems AeroNav",
            "content": "Northstar Systems AeroNav provides aircraft navigation systems."
        }],
        PROFILE
    )

    assert result["evidence_status"] == "unrelated"
    assert result["competitor_type"] == "unclear"
    assert result["evidence_urls"] == []


def test_unrelated_topic_in_another_sentence_does_not_make_company_relevant():
    result = validate_company_evidence(
        {
            "name": "Northstar Systems",
            "product_name": "AeroNav",
            "evidence_urls": ["https://northstar-aero.example/about"]
        },
        [{
            "url": "https://northstar-aero.example/about",
            "title": "Northstar Systems",
            "content": (
                "Northstar Systems develops aircraft navigation products. "
                "This article compares several tools for website intelligence."
            )
        }],
        PROFILE
    )

    assert result["evidence_status"] == "unrelated"
    assert result["competitor_type"] == "unclear"


def test_explicit_comparison_is_retained_for_review_without_claiming_competitor_type():
    result = validate_company_evidence(
        {
            "name": "Northstar Systems",
            "product_name": "Orbit",
            "evidence_urls": ["https://research.example/compare"]
        },
        [{
            "url": "https://research.example/compare",
            "title": "Linkly AI vs Northstar Systems",
            "content": "A feature comparison of Linkly AI and Northstar Systems."
        }],
        PROFILE
    )

    assert result["evidence_status"] == "company_and_relevant_market_supported"
    assert result["competitor_type"] == "unclear"
    assert "explicitly compares" in result["relevance_reason"]
    assert "vs" in result["description"].lower()


def test_evidence_tier_ranks_above_semantic_similarity_and_drops_irrelevant_pages():
    ranked = rank_candidates_by_evidence([
        {
            "name": "Semantic Match",
            "evidence_status": "company_mentioned",
            "evidence_pages_accessible": 1,
            "heuristic_evidence_score": 0.30,
            "similarity_score": 0.99,
            "evidence_details": _fetched_evidence_detail("Semantic Match"),
        },
        {
            "name": "Direct Evidence",
            "evidence_status": "explicit_competitor_relationship_found",
            "evidence_pages_accessible": 1,
            "heuristic_evidence_score": 0.90,
            "similarity_score": 0.31,
            "evidence_details": _fetched_evidence_detail(
                "Direct Evidence", "explicit_competitor_relationship_found"
            ),
        },
        {
            "name": "Explicit Comparison",
            "evidence_status": "product_or_capability_comparison_found",
            "evidence_pages_accessible": 1,
            "heuristic_evidence_score": 0.75,
            "similarity_score": 0.50,
            "evidence_details": _fetched_evidence_detail(
                "Explicit Comparison", "product_or_capability_comparison_found"
            ),
        },
        {
            "name": "Irrelevant",
            "evidence_status": "no_relevant_evidence",
            "evidence_pages_accessible": 2,
            "similarity_score": 1.0
        },
        {
            "name": "Inaccessible",
            "evidence_status": "inaccessible",
            "evidence_pages_accessible": 0,
            "heuristic_evidence_score": 0.0,
            "similarity_score": 0.95
        },
        {
            "name": "Zero Score",
            "evidence_status": "explicit_competitor_relationship_found",
            "evidence_pages_accessible": 1,
            "heuristic_evidence_score": 0.0,
            "similarity_score": 1.0
        }
    ])

    assert [candidate["name"] for candidate in ranked] == [
        "Direct Evidence", "Explicit Comparison"
    ]


def test_weak_candidate_remains_unclear_and_is_not_marked_strong():
    ranked = rank_candidates_by_evidence([{
        "name": "Weak Candidate",
        "competitor_type": "unclear",
        "evidence_status": "company_mentioned",
        "evidence_strength": "weak",
        "evidence_pages_accessible": 1,
        "heuristic_evidence_score": 0.30,
        "similarity_score": 0.99,
        "evidence_details": _fetched_evidence_detail("Weak Candidate"),
    }])

    assert ranked == []


def test_generic_candidate_names_need_a_specific_product_to_be_considered():
    assert is_publisher_candidate({"name": "What"})
    assert is_publisher_candidate({"name": "market"})
    assert is_publisher_candidate({"name": "EIN Presswire"})
    assert is_publisher_candidate({"name": "EINPresswire"})
    assert is_publisher_candidate({"name": "EINPresswire.com"})
    assert not is_publisher_candidate({"name": "What", "product_name": "What CRM"})


def test_einpresswire_market_overlap_candidate_is_excluded_from_results():
    candidate = {
        "name": "EINPresswire",
        "product_name": "Press release distribution",
        "evidence_status": "market_overlap_unverified",
        "evidence_pages_accessible": 1,
        "heuristic_evidence_score": 0.8,
        "similarity_score": 0.99,
        "evidence_details": _fetched_evidence_detail(
            "EINPresswire", "market_overlap_unverified"
        ),
    }

    assert is_publisher_candidate(candidate)
    assert rank_candidates_by_evidence([candidate]) == []


def test_known_press_and_comparison_domains_are_excluded_as_candidate_sources():
    assert discovery.is_excluded_domain("www.einpresswire.com")
    assert discovery.is_excluded_domain("press.prnewswire.com")
    assert discovery.is_excluded_domain("www.trustradius.com")


def test_negated_competitor_wording_does_not_classify_direct():
    competitor_type, _ = classify_competitor_type_from_sources(
        "Northstar Systems", "Orbit", "Linkly AI", [{
            "url": "https://research.example/compare",
            "content": "Northstar Systems is not a direct competitor to Linkly AI."
        }]
    )

    assert competitor_type == "unclear"


def test_unqualified_competitor_claim_does_not_choose_direct_or_indirect():
    competitor_type, _ = classify_competitor_type_from_sources(
        "Northstar Systems", "Orbit", "Linkly AI", [{
            "url": "https://research.example/compare",
            "content": "Northstar Systems is a competitor to Linkly AI."
        }]
    )

    assert competitor_type == "unclear"


def test_credible_commerce_market_evidence_preserves_candidates():
    commerce_profile = {
        "brand_name": "Linkly AI",
        "category": "e-commerce platform",
        "products_services": ["online store platform"]
    }
    examples = [
        (
            "Squarespace",
            "",
            "Squarespace provides a website builder with online store tools for merchants."
        ),
        (
            "Salesforce Commerce Cloud",
            "Commerce Cloud",
            "Salesforce Commerce Cloud provides digital commerce tools for online retailers."
        )
    ]

    for index, (name, product_name, text) in enumerate(examples):
        result = validate_company_evidence(
            {
                "name": name,
                "product_name": product_name,
                "evidence_urls": [f"https://market-evidence.example/{index}"]
            },
            [{
                "url": f"https://market-evidence.example/{index}",
                "title": name,
                "content": text
            }],
            commerce_profile
        )

        assert result["evidence_status"] in {
            "company_and_product_supported",
            "company_and_relevant_market_supported"
        }
        assert result["competitor_type"] == "unclear"
        assert result["evidence_urls"]


def test_duplicate_companies_merge_across_name_formatting_and_domain():
    result = deduplicate_companies([
        {
            "name": "Northstar Systems, Inc.",
            "product_name": "Orbit",
            "evidence_urls": ["https://source.example/a"]
        },
        {
            "name": "NORTHSTAR SYSTEMS",
            "description": "Website intelligence software.",
            "official_website": "https://northstar-systems.com",
            "evidence_urls": ["https://source.example/b"]
        },
        {
            "name": "Northstar Systems LLC",
            "official_website": "https://www.northstar-systems.com",
            "evidence_urls": []
        }
    ])

    assert len(result) == 1
    assert result[0]["normalized_name"] == "northstar systems"
    assert result[0]["product_name"] == "Orbit"
    assert result[0]["description"] == "Website intelligence software."
    assert set(result[0]["evidence_urls"]) == {
        "https://source.example/a",
        "https://source.example/b"
    }


def test_classification_uses_explicit_page_evidence_and_otherwise_stays_unclear():
    direct, direct_reason = classify_competitor_type_from_sources(
        "Northstar Systems", "Orbit", "Linkly AI", [{
            "url": "https://research.example/compare",
            "content": "Northstar Systems is a direct competitor to Linkly AI."
        }]
    )
    indirect, _ = classify_competitor_type_from_sources(
        "Northstar Systems", "Orbit", "Linkly AI", [{
            "url": "https://research.example/compare",
            "content": "Northstar Systems is an indirect competitor to Linkly AI."
        }]
    )
    unclear, unclear_reason = classify_competitor_type_from_sources(
        "Northstar Systems", "Orbit", "Linkly AI", [{
            "url": "https://northstar-systems.com",
            "content": "Northstar Systems and Linkly AI both offer website intelligence software."
        }]
    )

    assert direct == "direct"
    assert "direct competitor" in direct_reason
    assert indirect == "indirect"
    assert unclear == "unclear"
    assert "do not establish" in unclear_reason
