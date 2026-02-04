from __future__ import annotations

import structlog
from fastapi import APIRouter
from sqlalchemy import func, select

from src.api.schemas import HealthResponse, MetricsResponse
from src.db.models import Chunk, Document, Feedback, QueryLog
from src.db.session import async_session
from src.dependencies import get_qdrant_client, get_redis_client

logger = structlog.get_logger()
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    deps = {}

    # Check Redis
    try:
        redis = get_redis_client()
        await redis.ping()
        deps["redis"] = "healthy"
    except Exception as e:
        deps["redis"] = f"unhealthy: {e}"

    # Check Qdrant
    try:
        qdrant = get_qdrant_client()
        await qdrant.get_collections()
        deps["qdrant"] = "healthy"
    except Exception as e:
        deps["qdrant"] = f"unhealthy: {e}"

    # Check PostgreSQL
    try:
        async with async_session() as session:
            await session.execute(select(func.now()))
        deps["postgres"] = "healthy"
    except Exception as e:
        deps["postgres"] = f"unhealthy: {e}"

    all_healthy = all(v == "healthy" for v in deps.values())

    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        version="0.1.0",
        dependencies=deps,
    )


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    async with async_session() as session:
        total_docs = (await session.execute(select(func.count(Document.id)))).scalar() or 0
        total_chunks = (await session.execute(select(func.count(Chunk.id)))).scalar() or 0
        total_queries = (await session.execute(select(func.count(QueryLog.id)))).scalar() or 0
        total_feedback = (await session.execute(select(func.count(Feedback.id)))).scalar() or 0

        cache_hits = (
            await session.execute(
                select(func.count(QueryLog.id)).where(QueryLog.cache_hit == True)  # noqa: E712
            )
        ).scalar() or 0

        avg_latency = (
            await session.execute(select(func.avg(QueryLog.latency_ms)))
        ).scalar() or 0.0

    cache_hit_rate = (cache_hits / total_queries * 100) if total_queries > 0 else 0.0

    return MetricsResponse(
        total_documents=total_docs,
        total_chunks=total_chunks,
        total_queries=total_queries,
        cache_hit_rate=round(cache_hit_rate, 2),
        avg_latency_ms=round(float(avg_latency), 2),
        feedback_count=total_feedback,
    )
