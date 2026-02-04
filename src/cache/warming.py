from __future__ import annotations

import structlog
from sqlalchemy import func, select

from src.db.models import QueryLog
from src.db.session import async_session

logger = structlog.get_logger()


class CacheWarmer:
    """Pre-compute and cache results for frequently asked queries."""

    def __init__(self, search_engine):
        self.search_engine = search_engine

    async def warm(self, top_n: int = 100):
        """Fetch top N most frequent queries and pre-compute their results."""
        queries = await self._get_top_queries(top_n)

        warmed = 0
        for query_text, count in queries:
            try:
                await self.search_engine.search(query=query_text, source="cache_warmer")
                warmed += 1
            except Exception as e:
                logger.warning("Cache warming failed for query", query=query_text[:50], error=str(e))

        logger.info("Cache warming completed", warmed=warmed, total=len(queries))
        return warmed

    @staticmethod
    async def _get_top_queries(top_n: int) -> list[tuple[str, int]]:
        """Get top N most frequent queries from the last 30 days."""
        async with async_session() as session:
            stmt = (
                select(QueryLog.query_text, func.count().label("cnt"))
                .where(QueryLog.cache_hit == False)  # noqa: E712
                .group_by(QueryLog.query_text)
                .order_by(func.count().desc())
                .limit(top_n)
            )
            result = await session.execute(stmt)
            return [(row.query_text, row.cnt) for row in result]
