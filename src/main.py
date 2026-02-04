from __future__ import annotations

import structlog
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.middleware import RequestLoggingMiddleware
from src.api.routes import search, feedback, ingest, documents, health
from src.config import settings
from src.db.session import init_db
from src.dependencies import get_qdrant_client, get_redis_client

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting support-rag", version="0.1.0")

    # Initialize database tables
    await init_db()

    # Ensure Qdrant collections exist
    await _ensure_qdrant_collections()

    # Verify Redis connection
    redis = get_redis_client()
    await redis.ping()
    logger.info("Redis connected")

    yield

    # Cleanup
    qdrant = get_qdrant_client()
    await qdrant.close()
    await redis.aclose()
    logger.info("Shutdown complete")


async def _ensure_qdrant_collections():
    from qdrant_client.models import (
        Distance,
        VectorParams,
        SparseVectorParams,
        SparseIndexParams,
        Modifier,
        PayloadSchemaType,
    )

    qdrant = get_qdrant_client()

    # Main documents collection
    collections = [c.name for c in (await qdrant.get_collections()).collections]

    if settings.qdrant_collection not in collections:
        await qdrant.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config={
                "dense": VectorParams(
                    size=settings.embedding_dimensions,
                    distance=Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(on_disk=False),
                    modifier=Modifier.IDF,
                ),
            },
        )
        # Create payload indexes for filtering
        for field, schema_type in [
            ("source_type", PayloadSchemaType.KEYWORD),
            ("document_id", PayloadSchemaType.KEYWORD),
            ("feedback_score", PayloadSchemaType.FLOAT),
        ]:
            await qdrant.create_payload_index(
                collection_name=settings.qdrant_collection,
                field_name=field,
                field_schema=schema_type,
            )
        logger.info("Created Qdrant collection", name=settings.qdrant_collection)

    # Query cache collection (for semantic cache)
    if settings.qdrant_cache_collection not in collections:
        await qdrant.create_collection(
            collection_name=settings.qdrant_cache_collection,
            vectors_config=VectorParams(
                size=settings.embedding_dimensions,
                distance=Distance.COSINE,
            ),
        )
        logger.info("Created Qdrant cache collection", name=settings.qdrant_cache_collection)


app = FastAPI(
    title="Support RAG",
    description="Production-grade RAG system for support search",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(RequestLoggingMiddleware)

app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(search.router, prefix="/api/v1", tags=["search"])
app.include_router(feedback.router, prefix="/api/v1", tags=["feedback"])
app.include_router(ingest.router, prefix="/api/v1", tags=["ingest"])
app.include_router(documents.router, prefix="/api/v1", tags=["documents"])
