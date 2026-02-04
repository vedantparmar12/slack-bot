from __future__ import annotations

import structlog

from src.cache.warming import CacheWarmer
from src.search.engine import SearchEngine

logger = structlog.get_logger()


async def warm_cache(top_n: int = 100):
    """Standalone function to warm cache with top queries."""
    engine = SearchEngine()
    warmer = CacheWarmer(engine)
    warmed = await warmer.warm(top_n=top_n)
    logger.info("Cache warming completed", queries_warmed=warmed)
    return warmed
