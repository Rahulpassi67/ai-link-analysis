from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_render_blueprint_uses_lightweight_similarity_and_keeps_tavily_optional():
    render = (ROOT / "render.yaml").read_text(encoding="utf-8")

    assert "plan: 1c-2g" in render
    assert "buildCommand: pip install -r requirements.txt" in render
    assert "preload_semantic_model" not in render
    assert "healthCheckPath: /health" in render
    assert "startCommand: uvicorn app.main:app --host 0.0.0.0 --port $PORT" in render
    assert "- key: GEMINI_API_KEY\n        sync: false" in render
    assert "- key: TAVILY_API_KEY" not in render


def test_production_and_development_requirements_are_separated_and_pinned():
    production = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    development = (ROOT / "requirements-dev.txt").read_text(encoding="utf-8").splitlines()

    assert production
    assert all("==" in requirement for requirement in production)
    assert not any(requirement.lower().startswith("pytest") for requirement in production)
    assert "-r requirements.txt" in development
    assert "pytest==9.2.0" in development
