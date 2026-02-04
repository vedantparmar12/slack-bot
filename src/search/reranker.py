from __future__ import annotations

import structlog
from sentence_transformers import CrossEncoder

from src.config import settings
from src.search.hybrid import SearchResult

logger = structlog.get_logger()

_reranker_model: CrossEncoder | None = None


def _get_reranker() -> CrossEncoder:
    global _reranker_model
    if _reranker_model is None:
        _reranker_model = CrossEncoder(settings.reranker_model)
        logger.info("Loaded reranker model", model=settings.reranker_model)
    return _reranker_model


class Reranker:
    def __init__(self):
        self.top_k = settings.reranker_top_k

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Rerank results using cross-encoder and apply feedback boost."""
        top_k = top_k or self.top_k

        if not results:
            return []

        model = _get_reranker()

        # Build query-document pairs
        pairs = [(query, r.text) for r in results]

        # Score with cross-encoder
        scores = model.predict(pairs)

        # Apply feedback boost: final = rerank_score * (1 + 0.1 * feedback_score)
        for i, result in enumerate(results):
            base_score = float(scores[i])
            feedback_boost = 1.0 + 0.1 * result.feedback_score
            result.score = base_score * feedback_boost

        # Sort by final score descending
        results.sort(key=lambda r: r.score, reverse=True)

        return results[:top_k]

    @staticmethod
    def get_confidence(top_score: float) -> str:
        """Classify confidence based on top reranker score."""
        if top_score >= 0.7:
            return "high"
        elif top_score >= 0.3:
            return "medium"
        else:
            return "low"
