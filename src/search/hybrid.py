from __future__ import annotations

import asyncio
from dataclasses import dataclass

import structlog
from qdrant_client.models import (
    Filter,
    FusionQuery,
    Prefetch,
    QueryRequest,
    SparseVector,
)

from src.config import settings
from src.dependencies import get_qdrant_client
from src.ingestion.embedder import Embedder, _tokenize_for_sparse

logger = structlog.get_logger()


@dataclass
class SearchResult:
    point_id: str
    score: float
    text: str
    title: str
    url: str
    source_type: str
    document_id: str
    chunk_index: int
    feedback_score: float
    metadata: dict


class HybridSearch:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.qdrant = get_qdrant_client()
        self.collection = settings.qdrant_collection
        self.top_k = settings.search_top_k

    async def search(
        self,
        queries: list[str],
        qdrant_filter: Filter | None = None,
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Run hybrid search across multiple query variations and fuse results."""
        top_k = top_k or self.top_k

        # Run all query variations in parallel
        tasks = [
            self._search_single(query, qdrant_filter, top_k) for query in queries
        ]
        all_results = await asyncio.gather(*tasks)

        # Fuse results from all variations using RRF
        fused = self._reciprocal_rank_fusion(all_results, k=60)

        # Return top results
        return fused[:top_k]

    async def _search_single(
        self,
        query: str,
        qdrant_filter: Filter | None,
        top_k: int,
    ) -> list[SearchResult]:
        """Hybrid search: dense + sparse with RRF fusion in single Qdrant call."""
        # Embed query for dense search
        query_embedding = await self.embedder.embed_query(query)

        # Tokenize for sparse search
        sparse_data = _tokenize_for_sparse(query)

        # Use Qdrant Query API with prefetch for hybrid search
        results = await self.qdrant.query_points(
            collection_name=self.collection,
            prefetch=[
                Prefetch(
                    query=query_embedding,
                    using="dense",
                    limit=top_k,
                    filter=qdrant_filter,
                ),
                Prefetch(
                    query=SparseVector(
                        indices=list(sparse_data.keys()),
                        values=list(sparse_data.values()),
                    ),
                    using="sparse",
                    limit=top_k,
                    filter=qdrant_filter,
                ),
            ],
            query=FusionQuery(fusion="rrf"),
            limit=top_k,
        )

        return [
            SearchResult(
                point_id=str(point.id),
                score=point.score,
                text=point.payload.get("text", ""),
                title=point.payload.get("title", ""),
                url=point.payload.get("url", ""),
                source_type=point.payload.get("source_type", ""),
                document_id=point.payload.get("document_id", ""),
                chunk_index=point.payload.get("chunk_index", 0),
                feedback_score=point.payload.get("feedback_score", 0.0),
                metadata={
                    k: v
                    for k, v in point.payload.items()
                    if k
                    not in (
                        "text",
                        "title",
                        "url",
                        "source_type",
                        "document_id",
                        "chunk_index",
                        "feedback_score",
                    )
                },
            )
            for point in results.points
        ]

    @staticmethod
    def _reciprocal_rank_fusion(
        result_sets: list[list[SearchResult]],
        k: int = 60,
    ) -> list[SearchResult]:
        """Fuse multiple ranked lists using Reciprocal Rank Fusion."""
        scores: dict[str, float] = {}
        results_map: dict[str, SearchResult] = {}

        for result_list in result_sets:
            for rank, result in enumerate(result_list):
                pid = result.point_id
                scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank + 1)
                # Keep the result with the best original score
                if pid not in results_map or result.score > results_map[pid].score:
                    results_map[pid] = result

        # Sort by fused score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        fused = []
        for pid in sorted_ids:
            result = results_map[pid]
            result.score = scores[pid]
            fused.append(result)

        return fused
