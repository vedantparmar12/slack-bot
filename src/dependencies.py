from __future__ import annotations

from functools import lru_cache

import redis.asyncio as aioredis
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient

from src.config import settings


@lru_cache
def get_openai_client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=settings.openai_api_key)


@lru_cache
def get_qdrant_client() -> AsyncQdrantClient:
    return AsyncQdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
    )


@lru_cache
def get_redis_client() -> aioredis.Redis:
    return aioredis.from_url(settings.redis_url, decode_responses=False)
