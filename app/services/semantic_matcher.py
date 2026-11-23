from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def get_embeddings(texts: list[str]):
    """Convert text into TF-IDF vectors for local similarity comparisons."""
    cleaned_texts = [
        text.strip() for text in texts
        if isinstance(text, str) and text.strip()
    ]
    if not cleaned_texts:
        return []
    try:
        return TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
        ).fit_transform(cleaned_texts)
    except ValueError as error:
        # TfidfVectorizer rejects text with no usable tokens; represent those
        # texts as zero vectors so callers receive a zero similarity score.
        if "empty vocabulary" not in str(error).casefold():
            raise
        return csr_matrix((len(cleaned_texts), 1), dtype=float)


def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate cosine similarity between two sentences."""
    if not text1.strip() or not text2.strip():
        return 0.0
    vectors = get_embeddings([text1, text2])
    score = cosine_similarity(vectors[0], vectors[1])[0][0]
    return round(float(score), 4)


def rank_similar_items(query: str, candidates: list[str]):
    """Rank candidate text by TF-IDF cosine similarity to a query."""
    if not query.strip():
        return []
    valid_candidates = [
        item for item in candidates
        if isinstance(item, str) and item.strip()
    ]
    if not valid_candidates:
        return []
    vectors = get_embeddings([query] + valid_candidates)
    scores = cosine_similarity(vectors[0], vectors[1:])[0]
    results = [
        {
            "description": candidate,
            "similarity_score": round(float(score), 4)
        }
        for candidate, score in zip(valid_candidates, scores)
    ]
    return sorted(results, key=lambda item: item["similarity_score"], reverse=True)
