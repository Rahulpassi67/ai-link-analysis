import json

import pytest

from app.services import llm_service, semantic_matcher


def test_semantic_model_loads_only_the_preloaded_local_directory(monkeypatch, tmp_path):
    (tmp_path / "modules.json").write_text(json.dumps([]), encoding="utf-8")
    calls = {}
    sentinel = object()

    def fake_sentence_transformer(path, **kwargs):
        calls["path"] = path
        calls.update(kwargs)
        return sentinel

    monkeypatch.setattr(semantic_matcher, "MODEL_PATH", tmp_path)
    monkeypatch.setattr(semantic_matcher, "SentenceTransformer", fake_sentence_transformer)
    monkeypatch.setattr(semantic_matcher, "_model", None)

    assert semantic_matcher._load_model() is sentinel
    assert calls == {"path": str(tmp_path), "local_files_only": True}


def test_missing_preloaded_model_fails_without_remote_download(monkeypatch, tmp_path):
    def unexpected_download(*_args, **_kwargs):
        raise AssertionError("Model loading must not access the remote hub at request time.")

    monkeypatch.setattr(semantic_matcher, "MODEL_PATH", tmp_path / "missing-model")
    monkeypatch.setattr(semantic_matcher, "SentenceTransformer", unexpected_download)
    monkeypatch.setattr(semantic_matcher, "_model", None)

    with pytest.raises(FileNotFoundError, match="preloaded"):
        semantic_matcher._load_model()


def test_model_loading_failure_preserves_gemini_analysis(monkeypatch, tmp_path):
    monkeypatch.setattr(semantic_matcher, "MODEL_PATH", tmp_path / "missing-model")
    monkeypatch.setattr(semantic_matcher, "_model", None)
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
