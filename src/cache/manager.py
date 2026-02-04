from __future__ import annotations

import hashlib
import uuid

import orjson
import structlog
from qdrant_client.models import PointStruct, VectorParams, Distance

from src.api.schemas import SearchFilters, SearchResponse
from src.config import settings
from src.dependencies import get_qdrant_client, get_redis_client
from src.ingestion.embedder import Embedder

logger = structlog.get_logger()


class CacheManager:
    """Two-layer cache: exact hash match (Redis) + semantic similarity (Qdrant)."""

    def __init__(self, embedder: Embedder):
        self.redis = get_redis_client()
        self.qdrant = get_qdrant_client()
        self.embedder = embedder
        self.cache_collection = settings.qdrant_cache_collection
        self.exact_ttl = settings.cache_exact_ttl_seconds
        self.semantic_ttl = settings.cache_semantic_ttl_seconds
        self.semantic_threshold = settings.cache_semantic_threshold

    def _cache_key(self, query: str, filters: SearchFilters | None) -> str:
        """Generate deterministic cache key from query + filters."""
        normalized = query.lower().strip()
        filter_str = ""
        if filters:
            parts = []
            if filters.source_types:
                parts.append(f"src={','.join(sorted(filters.source_types))}")
            if filters.tags:
                parts.append(f"tags={','.join(sorted(filters.tags))}")
            if filters.date_from:
                parts.append(f"from={filters.date_from.isoformat()}")
            if filters.date_to:
                parts.append(f"to={filters.date_to.isoformat()}")
            filter_str = "|".join(parts)

        raw = f"{normalized}|{filter_str}"
        return hashlib.sha256(raw.encode()).hexdigest()

    async def get(
        self, query: str, filters: SearchFilters | None = None
    ) -> SearchResponse | None:
        """Try Layer 1 (exact), then Layer 2 (semantic)."""
        # Layer 1: Exact match
        cache_key = self._cache_key(query, filters)
        result = await self._get_exact(cache_key)
        if result:
            logger.debug("Cache L1 hit (exact)", query=query[:50])
            return result

        # Layer 2: Semantic match
        result = await self._get_semantic(query)
        if result:
            logger.debug("Cache L2 hit (semantic)", query=query[:50])
            return result

        return None

    async def set(
        self,
        query: str,
        filters: SearchFilters | None,
        response: SearchResponse,
    ):
        """Store in both cache layers."""
        cache_key = self._cache_key(query, filters)

        # Layer 1: Store in Redis
        await self._set_exact(cache_key, response)

        # Layer 2: Store embedding in Qdrant for semantic matching
        await self._set_semantic(query, cache_key, response)

    async def invalidate_by_document(self, document_id: str):
        """Invalidate all cache entries referencing a specific document."""
        # Get the set of cache keys that reference this document
        doc_key = f"doc:{document_id}:cache_keys"
        cache_keys = await self.redis.smembers(doc_key)

        if cache_keys:
            # Delete all referenced cache entries
            pipeline = self.redis.pipeline()
            for key in cache_keys:
                pipeline.delete(f"cache:{key.decode() if isinstance(key, bytes) else key}")
            pipeline.delete(doc_key)
            await pipeline.execute()
            logger.info("Invalidated cache entries", document_id=document_id, count=len(cache_keys))

    async def flush_all(self):
        """Clear all caches."""
        # Clear Redis cache entries (only cache: prefixed keys)
        cursor = 0
        while True:
            cursor, keys = await self.redis.scan(cursor, match="cache:*", count=100)
            if keys:
                await self.redis.delete(*keys)
            if cursor == 0:
                break

        # Clear Qdrant cache collection
        try:
            await self.qdrant.delete_collection(self.cache_collection)
            await self.qdrant.create_collection(
                collection_name=self.cache_collection,
                vectors_config=VectorParams(
                    size=settings.embedding_dimensions,
                    distance=Distance.COSINE,
                ),
            )
        except Exception as e:
            logger.warning("Failed to reset cache collection", error=str(e))

        logger.info("Flushed all caches")

    async def get_stats(self) -> dict:
        """Get cache statistics."""
        # Count Redis cache entries
        cursor = 0
        exact_count = 0
        while True:
            cursor, keys = await self.redis.scan(cursor, match="cache:*", count=100)
            exact_count += len(keys)
            if cursor == 0:
                break

        # Count semantic cache entries
        try:
            collection_info = await self.qdrant.get_collection(self.cache_collection)
            semantic_count = collection_info.points_count
        except Exception:
            semantic_count = 0

        return {
            "exact_cache_entries": exact_count,
            "semantic_cache_entries": semantic_count,
        }

    # --- Layer 1: Exact Match ---

    async def _get_exact(self, cache_key: str) -> SearchResponse | None:
        data = await self.redis.get(f"cache:{cache_key}")
        if data:
            return SearchResponse(**orjson.loads(data))
        return None

    async def _set_exact(self, cache_key: str, response: SearchResponse):
        serialized = orjson.dumps(response.model_dump())
        await self.redis.setex(f"cache:{cache_key}", self.exact_ttl, serialized)

        # Track which documents are referenced by this cache entry
        for source in response.sources:
            doc_key = f"doc:{source.chunk_id}:cache_keys"
            await self.redis.sadd(doc_key, cache_key)
            await self.redis.expire(doc_key, self.semantic_ttl)

    # --- Layer 2: Semantic Match ---

    async def _get_semantic(self, query: str) -> SearchResponse | None:
        try:
            query_embedding = await self.embedder.embed_query(query)

            results = await self.qdrant.query_points(
                collection_name=self.cache_collection,
                query=query_embedding,
                limit=1,
                score_threshold=self.semantic_threshold,
            )

            if not results.points:
                return None

            point = results.points[0]
            matched_cache_key = point.payload.get("cache_key", "")

            # Fetch the cached response from Redis
            return await self._get_exact(matched_cache_key)

        except Exception as e:
            logger.debug("Semantic cache lookup failed", error=str(e))
            return None

    async def _set_semantic(
        self, query: str, cache_key: str, response: SearchResponse
    ):
        try:
            query_embedding = await self.embedder.embed_query(query)

            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=query_embedding,
                payload={
                    "cache_key": cache_key,
                    "query": query,
                },
            )

            await self.qdrant.upsert(
                collection_name=self.cache_collection,
                points=[point],
            )
        except Exception as e:
            logger.debug("Semantic cache store failed", error=str(e))
