import asyncio

from app.services.competitor_discovery import (
    classify_competitor_type_from_sources,
    rank_candidates_by_evidence,
)
from app.services.evidence_verifier import (
    classify_evidence,
    deduplicate_competitor_excerpts,
    evaluate_evidence,
    verify_competitor_evidence,
    verify_competitors,
)


def test_explicit_relationship_returns_exact_excerpt_and_metadata():
    result = evaluate_evidence(
        {
            "name": "Northstar Systems",
            "product_name": "Orbit",
            "official_website": "https://northstar.example"
        },
        [{
            "url": "https://research.example/compare",
            "accessible": True,
            "text": (
                "Northstar Systems makes Orbit. "
                "Orbit competes with Linkly AI for customer support teams."
            )
        }],
        analyzed_brand_name="Linkly AI"
    )

    assert result["evidence_status"] == "explicit_competitor_relationship_found"
    detail = result["evidence_details"][0]
    assert detail["classification"] == "explicit_competitor_relationship_found"
    excerpt = next(
        excerpt for excerpt in detail["excerpts"]
        if excerpt["classification"] == "explicit_competitor_relationship_found"
    )
    assert excerpt["competitor_name"] == "Northstar Systems"
    assert excerpt["source_url"] == "https://research.example/compare"
    assert excerpt["matched_text"] == (
        "Orbit competes with Linkly AI for customer support teams."
    )
    assert {"Orbit", "Linkly AI", "competes with"}.issubset(
        set(excerpt["matching_terms"])
    )
    assert excerpt["accessibility_status"] == "accessible"


def test_genuine_direct_competitor_phrase_is_captured_as_source_evidence():
    result = evaluate_evidence(
        {"name": "Northstar Systems", "product_name": "Orbit"},
        [{
            "url": "https://research.example/market-map",
            "accessible": True,
            "text": (
                "Northstar Systems and Linkly AI both sell software. "
                "Northstar Systems is a direct competitor to Linkly AI."
            )
        }],
        analyzed_brand_name="Linkly AI"
    )

    assert result["evidence_status"] == "explicit_competitor_relationship_found"
    excerpt = next(
        excerpt for excerpt in result["evidence_details"][0]["excerpts"]
        if excerpt["classification"] == "explicit_competitor_relationship_found"
    )
    assert excerpt["source_url"] == "https://research.example/market-map"
    assert excerpt["matched_text"] == "Northstar Systems is a direct competitor to Linkly AI."
    assert classify_evidence(result["evidence_status"])["evidence_strength"] == "strong"
    assert classify_evidence(result["evidence_status"])["heuristic_evidence_score"] == 0.90


def test_market_overlap_alone_remains_unclear_and_below_comparison_strength():
    result = evaluate_evidence(
        {"name": "Northstar Systems", "product_name": "Orbit"},
        [{
            "url": "https://northstar.example/about",
            "accessible": True,
            "text": (
                "Northstar Systems makes Orbit, an ecommerce platform. "
                "Linkly AI also works with ecommerce businesses."
            )
        }],
        analyzed_brand_name="Linkly AI"
    )

    assert result["evidence_status"] == "company_and_product_mentioned"
    assert result["evidence_strength"] == "partial"
    assert result["evidence_status"] != "explicit_competitor_relationship_found"


def test_explicit_product_comparison_has_intermediate_strength_but_no_type():
    result = evaluate_evidence(
        {"name": "Northstar Systems", "product_name": "Orbit"},
        [{
            "url": "https://research.example/feature-review",
            "accessible": True,
            "text": "Orbit is compared with Linkly AI for ecommerce reporting."
        }],
        analyzed_brand_name="Linkly AI"
    )

    assert result["evidence_status"] == "product_or_capability_comparison_found"
    assert result["evidence_strength"] == "moderate"
    score = classify_evidence(result["evidence_status"])["heuristic_evidence_score"]
    assert 0.60 < score < 0.90
    excerpt = result["evidence_details"][0]["excerpts"][0]
    assert excerpt["classification"] == "product_or_capability_comparison_found"
    assert excerpt["matched_text"] == "Orbit is compared with Linkly AI for ecommerce reporting."


def test_ecommerce_alternative_statement_is_kept_but_relationship_stays_unclear():
    statement = (
        "BigCommerce is an alternative to Shopify in the ecommerce platform market."
    )
    result = evaluate_evidence(
        {"name": "BigCommerce", "product_name": "BigCommerce"},
        [{
            "url": "https://comparison.example/bigcommerce-shopify",
            "accessible": True,
            "text": statement
        }],
        analyzed_brand_name="Shopify",
        relevance_anchors=["e-commerce platform"]
    )

    assert result["evidence_status"] == "product_or_capability_comparison_found"
    assert result["evidence_details"][0]["excerpts"][0]["matched_text"] == statement
    relationship, _ = classify_competitor_type_from_sources(
        "BigCommerce", "BigCommerce", "Shopify", [{
            "url": "https://comparison.example/bigcommerce-shopify",
            "content": statement
        }]
    )
    assert relationship == "unclear"


def test_negated_relationship_is_not_classified_as_competitor_evidence():
    result = evaluate_evidence(
        {"name": "Northstar Systems", "product_name": "Orbit"},
        [{
            "url": "https://research.example/market-map",
            "accessible": True,
            "text": "Northstar Systems is not a competitor to Linkly AI."
        }],
        analyzed_brand_name="Linkly AI"
    )

    assert result["evidence_status"] == "company_mentioned"
    assert all(
        excerpt["classification"] != "explicit_competitor_relationship_found"
        for excerpt in result["evidence_details"][0]["excerpts"]
    )


def test_own_company_and_product_mentions_do_not_verify_relationship():
    result = evaluate_evidence(
        {"name": "Northstar Systems", "product_name": "Orbit"},
        [{
            "url": "https://northstar.example/product",
            "accessible": True,
            "text": (
                "Northstar Systems builds Orbit. "
                "Orbit helps teams automate customer support."
            )
        }],
        analyzed_brand_name="Linkly AI"
    )

    assert result["evidence_status"] == "company_and_product_mentioned"
    assert result["evidence_status"] != "explicit_competitor_relationship_found"
    assert result["evidence_details"][0]["excerpts"]
    assert all(
        excerpt["classification"] != "explicit_competitor_relationship_found"
        for excerpt in result["evidence_details"][0]["excerpts"]
    )


def test_irrelevant_accessible_page_has_no_relevant_evidence():
    result = evaluate_evidence(
        {"name": "Northstar Systems", "product_name": "Orbit"},
        [{
            "url": "https://research.example/general",
            "accessible": True,
            "text": "Welcome to our research site. We publish industry news."
        }],
        analyzed_brand_name="Linkly AI"
    )

    assert result["evidence_status"] == "no_relevant_evidence"
    assert result["evidence_details"][0]["classification"] == "no_relevant_evidence"
    assert result["evidence_details"][0]["excerpts"] == []


def test_inaccessible_page_is_reported_without_claiming_relevance():
    result = evaluate_evidence(
        {"name": "Northstar Systems", "product_name": "Orbit"},
        [{
            "url": "https://northstar.example/product",
            "accessible": False,
            "reason": "HTTPStatusError: 403"
        }],
        analyzed_brand_name="Linkly AI"
    )

    assert result["evidence_status"] == "inaccessible"
    detail = result["evidence_details"][0]
    assert detail["accessibility_status"] == "inaccessible"
    assert detail["classification"] == "page_inaccessible"
    assert detail["reason"] == "HTTPStatusError: 403"
    assert detail["excerpts"] == []


def test_zero_accessible_pages_cannot_become_a_ranked_discovery(monkeypatch):
    async def inaccessible_page(client, url):
        return {
            "url": url,
            "accessible": False,
            "reason": "HTTP 403",
            "text": ""
        }

    monkeypatch.setattr("app.services.evidence_verifier.fetch_evidence_page", inaccessible_page)
    candidate = asyncio.run(verify_competitor_evidence({
        "name": "Northstar Systems",
        "evidence_urls": ["https://northstar.example/about"]
    }, analyzed_brand_name="Linkly AI"))

    assert candidate["evidence_pages_accessible"] == 0
    assert candidate["heuristic_evidence_score"] == 0.0
    assert candidate["evidence_strength"] == "none"
    assert candidate["evidence_status"] == "inaccessible"
    assert rank_candidates_by_evidence([candidate]) == []


def test_verification_batch_logs_accessible_and_inaccessible_page_counts(monkeypatch, caplog):
    async def fake_verify(company, **kwargs):
        return {
            "name": company["name"],
            "evidence_pages_checked": 2,
            "evidence_pages_accessible": 1,
            "evidence_details": [],
        }

    monkeypatch.setattr("app.services.evidence_verifier.verify_competitor_evidence", fake_verify)
    with caplog.at_level("INFO", logger="app.services.evidence_verifier"):
        results = asyncio.run(verify_competitors([{"name": "Candidate"}]))

    record = next(
        item for item in caplog.records
        if getattr(item, "diagnostic_event", None)
        == "competitor_evidence.verification_batch_completed"
    )
    assert len(results) == 1
    assert record.page_fetch_success_count == 1
    assert record.page_fetch_inaccessible_count == 1


def test_product_relevance_filters_unrelated_brand_mentions():
    result = evaluate_evidence(
        {"name": "Adobe", "product_name": "Adobe Commerce"},
        [
            {
                "url": "https://adobe.example/acrobat",
                "accessible": True,
                "text": "Adobe Acrobat edits PDF files for creative teams."
            },
            {
                "url": "https://adobe.example/news",
                "accessible": True,
                "text": "Adobe News announces the company's latest creative products."
            },
            {
                "url": "https://adobe.example/creative-cloud",
                "accessible": True,
                "text": "Adobe Creative Cloud helps designers edit video and images."
            },
            {
                "url": "https://adobe.example/commerce-features",
                "accessible": True,
                "text": "Adobe Commerce offers customizable checkout and catalog features."
            },
            {
                "url": "https://adobe.example/commerce",
                "accessible": True,
                "text": "Adobe Commerce is an ecommerce platform for merchants."
            }
        ],
        analyzed_brand_name="Shopify",
        relevance_anchors=["e-commerce platform", "online store platform"]
    )

    unrelated_pages = result["evidence_details"][:4]
    relevant = result["evidence_details"][4]
    assert all(page["classification"] == "no_relevant_evidence" for page in unrelated_pages)
    assert all(page["excerpts"] == [] for page in unrelated_pages)
    assert relevant["classification"] == "company_and_product_mentioned"
    assert len(relevant["excerpts"]) == 1
    assert "Adobe Commerce" in relevant["excerpts"][0]["matched_text"]


def _rankable_candidate(name, product_name, result):
    classification = classify_evidence(result["evidence_status"])
    accessible_count = sum(
        detail.get("accessible") is True for detail in result["evidence_details"]
    )
    return {
        "name": name,
        "product_name": product_name,
        "evidence_status": result["evidence_status"],
        "evidence_pages_accessible": accessible_count,
        "heuristic_evidence_score": classification["evidence_confidence"],
        "evidence_details": result["evidence_details"],
        "similarity_score": 1.0,
    }


def test_generic_adobe_and_salesforce_commerce_mentions_are_not_candidates():
    result = evaluate_evidence(
        {"name": "Adobe", "product_name": "Adobe Commerce"},
        [{
            "url": "https://industry.example/commerce-products",
            "accessible": True,
            "text": (
                "Adobe Commerce and Salesforce Commerce Cloud are enterprise "
                "products from major software companies."
            ),
        }],
        analyzed_brand_name="Shopify",
        relevance_anchors=["e-commerce platform", "online store platform"],
    )

    assert result["evidence_status"] == "no_relevant_evidence"
    assert result["evidence_details"][0]["excerpts"] == []
    assert rank_candidates_by_evidence([
        _rankable_candidate("Adobe", "Adobe Commerce", result)
    ]) == []


def test_inaccessible_market_report_does_not_verify_candidate():
    result = evaluate_evidence(
        {"name": "Adobe", "product_name": "Adobe Commerce"},
        [{
            "url": "https://reports.example/commerce-market-report",
            "accessible": False,
            "reason": "HTTP 403",
        }],
        analyzed_brand_name="Shopify",
        relevance_anchors=["e-commerce platform"],
    )

    candidate = _rankable_candidate("Adobe", "Adobe Commerce", result)
    assert result["evidence_status"] == "inaccessible"
    assert result["evidence_details"][0]["excerpts"] == []
    assert rank_candidates_by_evidence([candidate]) == []


def test_accessible_third_party_adobe_comparison_can_qualify_but_stays_unclear():
    statement = (
        "Adobe Commerce is an alternative to Shopify in the ecommerce "
        "platform market."
    )
    source_url = "https://review.example/shopify-vs-adobe-commerce"
    result = evaluate_evidence(
        {"name": "Adobe", "product_name": "Adobe Commerce"},
        [{"url": source_url, "accessible": True, "text": statement}],
        analyzed_brand_name="Shopify",
        relevance_anchors=["e-commerce platform"],
    )

    candidate = _rankable_candidate("Adobe", "Adobe Commerce", result)
    assert result["evidence_status"] == "product_or_capability_comparison_found"
    assert result["evidence_details"][0]["excerpts"][0]["matched_text"] == statement
    assert result["evidence_details"][0]["source_url"] == source_url
    assert rank_candidates_by_evidence([candidate]) == [candidate]
    assert classify_competitor_type_from_sources(
        "Adobe", "Adobe Commerce", "Shopify", [{"url": source_url, "content": statement}]
    )[0] == "unclear"


def test_relevant_market_context_without_competitor_wording_can_qualify_unclear():
    statement = "Adobe Commerce is an ecommerce platform for merchants."
    source_url = "https://industry.example/adobe-commerce-platforms"
    result = evaluate_evidence(
        {"name": "Adobe", "product_name": "Adobe Commerce"},
        [{"url": source_url, "accessible": True, "text": statement}],
        analyzed_brand_name="Shopify",
        relevance_anchors=["e-commerce platform"],
    )

    candidate = _rankable_candidate("Adobe", "Adobe Commerce", result)
    assert result["evidence_status"] == "company_and_product_mentioned"
    assert rank_candidates_by_evidence([candidate]) == [candidate]
    assert classify_competitor_type_from_sources(
        "Adobe", "Adobe Commerce", "Shopify", [{"url": source_url, "content": statement}]
    )[0] == "unclear"


def test_standalone_company_and_generic_product_mentions_are_filtered():
    result = evaluate_evidence(
        {"name": "BigCommerce", "product_name": "BigCommerce"},
        [
            {
                "url": "https://bigcommerce.example/about",
                "accessible": True,
                "text": "BigCommerce."
            },
            {
                "url": "https://bigcommerce.example/features",
                "accessible": True,
                "text": "BigCommerce offers flexible features for growing businesses."
            }
        ],
        analyzed_brand_name="Shopify",
        relevance_anchors=["e-commerce platform", "online store platform"]
    )

    assert result["evidence_status"] == "no_relevant_evidence"
    assert all(
        detail["classification"] == "no_relevant_evidence"
        and detail["excerpts"] == []
        for detail in result["evidence_details"]
    )


def test_official_self_competitor_claim_is_not_independent_relationship_evidence():
    statement = (
        "BigCommerce is a direct competitor to Shopify in the ecommerce platform market."
    )
    result = evaluate_evidence(
        {
            "name": "BigCommerce",
            "product_name": "BigCommerce",
            "official_website": "https://bigcommerce.example"
        },
        [{
            "url": "https://bigcommerce.example/about",
            "accessible": True,
            "text": statement
        }],
        analyzed_brand_name="Shopify",
        relevance_anchors=["e-commerce platform"]
    )

    detail = result["evidence_details"][0]
    assert result["evidence_status"] != "explicit_competitor_relationship_found"
    assert detail["classification"] != "explicit_competitor_relationship_found"
    assert detail["source_type"] == "official_website_candidate"
    assert detail["competitor_language_found"] is False
    assert detail["excerpts"][0]["matched_text"] == statement

    third_party_relationships = [
        {
            "url": excerpt["source_url"],
            "content": excerpt["matched_text"]
        }
        for excerpt in detail["excerpts"]
        if excerpt["classification"] == "explicit_competitor_relationship_found"
    ]
    relationship, _ = classify_competitor_type_from_sources(
        "BigCommerce", "BigCommerce", "Shopify", third_party_relationships
    )
    assert relationship == "unclear"


def test_market_overlap_is_required_when_candidate_product_is_unknown():
    result = evaluate_evidence(
        {"name": "Squarespace", "product_name": ""},
        [
            {
                "url": "https://squarespace.example/team",
                "accessible": True,
                "text": "Squarespace serves creative teams with design templates."
            },
            {
                "url": "https://squarespace.example/commerce",
                "accessible": True,
                "text": "Squarespace provides online store and ecommerce tools for merchants."
            }
        ],
        analyzed_brand_name="Shopify",
        relevance_anchors=["e-commerce platform", "online store platform"]
    )

    irrelevant, relevant = result["evidence_details"]
    assert irrelevant["classification"] == "no_relevant_evidence"
    assert irrelevant["excerpts"] == []
    assert relevant["classification"] == "company_mentioned"
    assert relevant["excerpts"]


def test_duplicate_and_near_duplicate_excerpts_are_omitted_across_pages():
    first = (
        "Northstar Systems Orbit generated 25% of the global ecommerce platform "
        "market share in 2024 according to the annual industry report."
    )
    near_duplicate = (
        "Northstar Systems Orbit generated 25% of global ecommerce platform market "
        "share in 2024, according to the annual industry report."
    )
    result = evaluate_evidence(
        {"name": "Northstar Systems", "product_name": "Orbit"},
        [
            {
                "url": "https://reports.example/market-share-1",
                "accessible": True,
                "text": first
            },
            {
                "url": "https://reports.example/market-share-2",
                "accessible": True,
                "text": near_duplicate
            },
            {
                "url": "https://northstar.example/product",
                "accessible": True,
                "text": "Northstar Systems builds Orbit, an ecommerce platform for enterprise merchants."
            }
        ],
        analyzed_brand_name="Shopify",
        relevance_anchors=["e-commerce platform"]
    )

    assert len(result["evidence_details"][0]["excerpts"]) == 1
    assert result["evidence_details"][1]["excerpts"] == []
    assert result["evidence_details"][1]["duplicate_excerpts_omitted"] == 1
    assert result["evidence_details"][2]["excerpts"]


def test_semantically_repetitive_excerpts_are_deduplicated_after_word_reordering():
    first = (
        "Northstar Systems offers merchants an ecommerce platform with catalog "
        "management and order tools for store operations."
    )
    reordered = (
        "For store operations, Northstar Systems provides merchants catalog "
        "management, order tools, and an ecommerce platform."
    )
    result = evaluate_evidence(
        {"name": "Northstar Systems", "product_name": "Northstar Systems"},
        [
            {"url": "https://reports.example/first", "accessible": True, "text": first},
            {"url": "https://reports.example/second", "accessible": True, "text": reordered}
        ],
        analyzed_brand_name="Shopify",
        relevance_anchors=["ecommerce platform"]
    )

    assert len(result["evidence_details"][0]["excerpts"]) == 1
    assert result["evidence_details"][1]["excerpts"] == []
    assert result["evidence_details"][1]["duplicate_excerpts_omitted"] == 1


def test_repeated_market_share_excerpt_is_not_repeated_across_competitor_cards():
    repeated = (
        "Northstar Systems Orbit generated 25% of the global ecommerce platform "
        "market share in 2024 according to the annual industry report."
    )
    near_duplicate = (
        "Northstar Systems Orbit generated 25% of global ecommerce platform market "
        "share in 2024, according to the annual industry report."
    )
    distinct = (
        "Northstar Systems Orbit generated 31% of the global ecommerce platform "
        "market share in 2025 according to the annual industry report."
    )
    candidates = [
        {"evidence_details": [{"excerpts": [{"matched_text": repeated}]}]},
        {"evidence_details": [{"excerpts": [
            {"matched_text": near_duplicate},
            {"matched_text": distinct}
        ]}]}
    ]

    deduplicate_competitor_excerpts(candidates)

    assert candidates[0]["evidence_details"][0]["excerpts"] == [
        {"matched_text": repeated}
    ]
    second_detail = candidates[1]["evidence_details"][0]
    assert [item["matched_text"] for item in second_detail["excerpts"]] == [distinct]
    assert second_detail["duplicate_excerpts_omitted"] == 1


def test_paraphrased_market_share_finding_is_deduplicated_but_aggregate_is_kept():
    source_url = "https://industry.example/us-ecommerce-share"
    market_lead = (
        "Shopify leads the US ecommerce platform market with 30% share, "
        "followed by Wix at 23%."
    )
    point_lead = "Shopify has a 7-point lead over Wix Stores at 23%."
    combined_share = "Shopify and Wix together control 53% of the market."
    candidates = [{"evidence_details": [{
        "source_url": source_url,
        "excerpts": [
            {"source_url": source_url, "matched_text": market_lead},
            {"source_url": source_url, "matched_text": point_lead},
            {"source_url": source_url, "matched_text": combined_share},
        ]
    }]}]

    deduplicate_competitor_excerpts(candidates)

    detail = candidates[0]["evidence_details"][0]
    assert [item["matched_text"] for item in detail["excerpts"]] == [
        market_lead,
        combined_share,
    ]
    assert all(item["source_url"] == source_url for item in detail["excerpts"])
    assert detail["duplicate_excerpts_omitted"] == 1


def test_official_and_third_party_sources_are_distinguished():
    result = evaluate_evidence(
        {
            "name": "Northstar Systems",
            "product_name": "Orbit",
            "official_website": "https://northstar.example"
        },
        [
            {
                "url": "https://northstar.example/product",
                "accessible": True,
                "text": "Northstar Systems builds Orbit for enterprise merchants."
            },
            {
                "url": "https://research.example/review",
                "accessible": True,
                "text": "Northstar Systems Orbit is used by enterprise merchants."
            }
        ],
        relevance_anchors=["enterprise merchants"]
    )

    assert result["evidence_details"][0]["source_type"] == "official_website_candidate"
    assert result["evidence_details"][1]["source_type"] == "third_party"
    assert result["evidence_details"][0]["excerpts"][0]["source_type"] == "official_website_candidate"
    assert result["evidence_details"][1]["excerpts"][0]["source_type"] == "third_party"


def test_official_and_comparison_sources_are_prioritized(monkeypatch):
    requested = []

    async def fake_fetch(client, url):
        requested.append(url)
        return {
            "url": url,
            "accessible": True,
            "reason": None,
            "text": "Northstar Systems Orbit builds ecommerce tools for merchants."
        }

    monkeypatch.setattr("app.services.evidence_verifier.fetch_evidence_page", fake_fetch)
    asyncio.run(verify_competitor_evidence({
        "name": "Northstar Systems",
        "product_name": "Orbit",
        "official_website": "https://northstar.example/product",
        "evidence_urls": [
            "https://general.example/story",
            "https://comparison.example/northstar-vs-shopify",
            "https://statista.com/reports/ecommerce-market"
        ]
    }, analyzed_brand_name="Shopify", relevance_anchors=["ecommerce tools"]))

    assert requested == [
        "https://northstar.example/product",
        "https://statista.com/reports/ecommerce-market",
        "https://comparison.example/northstar-vs-shopify"
    ]
