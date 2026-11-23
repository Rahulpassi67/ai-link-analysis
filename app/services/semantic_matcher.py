from pathlib import Path
from threading import Lock

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / ".model_cache"
    / "all-MiniLM-L6-v2"
)
_model = None
_model_lock = Lock()


def _load_model():
    """Load only the model preloaded into the deployment artifact."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                if not (MODEL_PATH / "modules.json").is_file():
                    raise FileNotFoundError(
                        "The semantic model is not preloaded. Run "
                        "scripts/preload_semantic_model.py during the build."
                    )
                _model = SentenceTransformer(
                    str(MODEL_PATH),
                    local_files_only=True
                )
    return _model


def get_embeddings(texts: list[str]):
    """Convert sentences to vectors using the preloaded local model."""
    cleaned_texts = [
        text.strip() for text in texts
        if isinstance(text, str) and text.strip()
    ]
    if not cleaned_texts:
        return []
    return _load_model().encode(
        cleaned_texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )


def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate cosine similarity between two sentences."""
    if not text1.strip() or not text2.strip():
        return 0.0
    embeddings = get_embeddings([text1, text2])
    score = cosine_similarity(
        [embeddings[0]],
        [embeddings[1]]
    )[0][0]
    return round(float(score), 4)


def rank_similar_items(query: str, candidates: list[str]):
    """Rank text using the pinned, preloaded sentence embedding model."""
    if not query.strip():
        return []
    valid_candidates = [
        item for item in candidates
        if isinstance(item, str) and item.strip()
    ]
    if not valid_candidates:
        return []
    embeddings = get_embeddings([query] + valid_candidates)
    scores = cosine_similarity(
        [embeddings[0]],
        embeddings[1:]
    )[0]
    results = [
        {
            "description": candidate,
            "similarity_score": round(float(score), 4)
        }
        for candidate, score in zip(valid_candidates, scores)
    ]
    return sorted(results, key=lambda item: item["similarity_score"], reverse=True)
