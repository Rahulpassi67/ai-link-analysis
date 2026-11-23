from app.services import llm_service, semantic_matcher


def test_get_embeddings_returns_tfidf_vectors_for_nonempty_text():
    vectors = semantic_matcher.get_embeddings([
        "online commerce platform",
        "online store platform",
    ])

    assert vectors.shape[0] == 2
    assert vectors.nnz > 0


def test_similarity_scores_and_candidate_ranking_are_preserved():
    assert semantic_matcher.calculate_similarity(
        "online commerce platform",
        "online commerce platform",
    ) == 1.0
    assert semantic_matcher.calculate_similarity("online commerce", "sailing boats") == 0.0

    ranked = semantic_matcher.rank_similar_items(
        "online commerce platform",
        ["sailing boats", "online commerce store platform"],
    )

    assert [item["description"] for item in ranked] == [
        "online commerce store platform",
        "sailing boats",
    ]
    assert all(set(item) == {"description", "similarity_score"} for item in ranked)
    assert all(0.0 <= item["similarity_score"] <= 1.0 for item in ranked)


def test_empty_or_unusable_text_returns_zero_similarity():
    assert semantic_matcher.get_embeddings([]) == []
    assert semantic_matcher.calculate_similarity("", "commerce") == 0.0
    assert semantic_matcher.calculate_similarity("the and a", "!!!") == 0.0
    assert semantic_matcher.rank_similar_items("", ["commerce"]) == []
    assert semantic_matcher.rank_similar_items("commerce", [" ", None]) == []


def test_similarity_failure_preserves_gemini_analysis(monkeypatch):
    def fail_ranking(*_args, **_kwargs):
        raise RuntimeError("TF-IDF scoring unavailable")

    monkeypatch.setattr(semantic_matcher, "rank_similar_items", fail_ranking)
    analysis = {
        "competitor_analysis": {
            "brand_name": "Shopify",
            "category": "ecommerce",
            "overview": "Online commerce platform",
            "products_services": [],
            "competitors": [{"name": "BigCommerce", "strengths": [], "weaknesses": []}],
        }
    }

    result = llm_service.add_semantic_competitor_ranking(analysis)

    assert result is analysis
    assert result["competitor_analysis"]["competitors"][0]["name"] == "BigCommerce"
