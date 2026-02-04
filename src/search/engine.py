from __future__ import annotations

import hashlib
import time
import uuid

import structlog

from src.api.schemas import SearchFilters, SearchResponse, SourceResult
from src.cache.manager import CacheManager
from src.config import settings
from src.dependencies import get_openai_client
from src.search.filters import build_qdrant_filter
from src.search.hybrid import HybridSearch
from src.search.query_expander import QueryExpander
from src.search.reranker import Reranker
from src.ingestion.embedder import Embedder

logger = structlog.get_logger()

ANSWER_SYSTEM_PROMPT = """You are a helpful support assistant. Answer the user's question using ONLY the provided context. Follow these rules:

1. Use the context to provide an accurate, concise answer
2. Cite your sources using [1], [2], etc. matching the source numbers
3. If the context doesn't contain enough information to answer, say so clearly
4. Keep the answer focused and well-structured
5. Use markdown formatting for readability"""

ANSWER_USER_PROMPT = """Context:
{context}

Question: {query}

Provide a helpful answer based on the context above. Cite sources as [1], [2], etc."""


class SearchEngine:
    def __init__(self):
        self.embedder = Embedder()
        self.query_expander = QueryExpander()
        self.hybrid_search = HybridSearch(self.embedder)
        self.reranker = Reranker()
        self.cache = CacheManager(self.embedder)
        self.openai = get_openai_client()

    async def search(
        self,
        query: str,
        filters: SearchFilters | None = None,
        limit: int = 5,
        source: str = "api",
    ) -> SearchResponse:
        """Full search pipeline: cache → expand → search → rerank → generate."""
        start = time.time()
        query_id = str(uuid.uuid4())
        query_hash = hashlib.sha256(query.lower().strip().encode()).hexdigest()

        # Step 1: Check cache
        cached = await self.cache.get(query, filters)
        if cached:
            cached.query_id = query_id
            cached.cached = True
            latency = (time.time() - start) * 1000
            cached.latency_ms = latency
            logger.info("Cache hit", query=query, latency_ms=latency)
            await self._log_query(query, query_hash, [], latency, True, source)
            return cached

        # Step 2: Query expansion
        expanded_queries = await self.query_expander.expand(query)

        # Step 3: Hybrid search
        qdrant_filter = build_qdrant_filter(filters)
        search_results = await self.hybrid_search.search(
            queries=expanded_queries,
            qdrant_filter=qdrant_filter,
            top_k=settings.search_rerank_top_k,
        )

        if not search_results:
            response = SearchResponse(
                answer="I couldn't find any relevant information for your query.",
                sources=[],
                query_id=query_id,
                cached=False,
                latency_ms=(time.time() - start) * 1000,
                confidence="low",
            )
            return response

        # Step 4: Rerank
        reranked = self.reranker.rerank(query, search_results, top_k=limit)
        confidence = self.reranker.get_confidence(reranked[0].score if reranked else 0)

        # Step 5: Generate answer
        answer = await self._generate_answer(query, reranked)

        # Build response
        sources = [
            SourceResult(
                chunk_id=r.point_id,
                title=r.title,
                url=r.url,
                snippet=r.text[:300] + "..." if len(r.text) > 300 else r.text,
                score=round(r.score, 4),
                source_type=r.source_type,
                feedback_score=r.feedback_score,
            )
            for r in reranked
        ]

        latency = (time.time() - start) * 1000
        response = SearchResponse(
            answer=answer,
            sources=sources,
            query_id=query_id,
            cached=False,
            latency_ms=round(latency, 2),
            confidence=confidence,
        )

        # Step 6: Cache the result
        await self.cache.set(query, filters, response)

        # Log query
        chunk_ids = [r.point_id for r in reranked]
        await self._log_query(query, query_hash, chunk_ids, latency, False, source)

        logger.info(
            "Search completed",
            query=query,
            results=len(sources),
            confidence=confidence,
            latency_ms=round(latency, 2),
        )

        return response

    async def _generate_answer(self, query: str, results: list) -> str:
        """Generate an answer using GPT-4o with retrieved context."""
        # Build context with numbered sources
        context_parts = []
        for i, result in enumerate(results, 1):
            source_label = f"[{i}] ({result.source_type}) {result.title}"
            if result.url:
                source_label += f" - {result.url}"
            context_parts.append(f"{source_label}\n{result.text}")

        context = "\n\n---\n\n".join(context_parts)

        try:
            response = await self.openai.chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": ANSWER_USER_PROMPT.format(
                            context=context, query=query
                        ),
                    },
                ],
                temperature=0.3,
                max_tokens=1000,
            )
            return response.choices[0].message.content or "Unable to generate answer."
        except Exception as e:
            logger.error("Answer generation failed", error=str(e))
            return "I found relevant documents but couldn't generate a summary. Please review the sources below."

    async def _log_query(
        self,
        query: str,
        query_hash: str,
        chunk_ids: list[str],
        latency_ms: float,
        cache_hit: bool,
        source: str,
    ):
        """Log query to database for analytics and cache warming."""
        try:
            from src.db.models import QueryLog
            from src.db.session import async_session

            async with async_session() as session:
                log = QueryLog(
                    query_text=query,
                    query_hash=query_hash,
                    result_chunk_ids=chunk_ids,
                    latency_ms=latency_ms,
                    cache_hit=cache_hit,
                    source=source,
                )
                session.add(log)
                await session.commit()
        except Exception as e:
            logger.warning("Failed to log query", error=str(e))
