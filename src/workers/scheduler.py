from __future__ import annotations

import structlog
from arq import create_pool
from arq.connections import RedisSettings

from src.config import settings

logger = structlog.get_logger()


def get_redis_settings() -> RedisSettings:
    """Parse Redis URL into ARQ RedisSettings."""
    # redis://localhost:6379 → host=localhost, port=6379
    url = settings.redis_url.replace("redis://", "")
    parts = url.split(":")
    host = parts[0] if parts else "localhost"
    port = int(parts[1]) if len(parts) > 1 else 6379
    return RedisSettings(host=host, port=port)


async def run_sync_slack(ctx):
    """Worker: Sync Slack channels."""
    from src.ingestion.chunker import SmartChunker
    from src.ingestion.embedder import Embedder
    from src.ingestion.slack import SlackIngester

    logger.info("Running Slack sync worker")
    chunker = SmartChunker()
    embedder = Embedder()
    ingester = SlackIngester(chunker, embedder)
    count = await ingester.ingest()
    logger.info("Slack sync completed", ingested=count)
    return count


async def run_sync_confluence(ctx):
    """Worker: Sync Confluence spaces."""
    from src.ingestion.chunker import SmartChunker
    from src.ingestion.confluence import ConfluenceIngester
    from src.ingestion.embedder import Embedder

    logger.info("Running Confluence sync worker")
    chunker = SmartChunker()
    embedder = Embedder()
    ingester = ConfluenceIngester(chunker, embedder)
    count = await ingester.ingest()
    logger.info("Confluence sync completed", ingested=count)
    return count


async def run_sync_git(ctx):
    """Worker: Sync Git doc repos."""
    from src.ingestion.chunker import SmartChunker
    from src.ingestion.embedder import Embedder
    from src.ingestion.git_docs import GitDocsIngester

    logger.info("Running Git docs sync worker")
    chunker = SmartChunker()
    embedder = Embedder()
    ingester = GitDocsIngester(chunker, embedder)
    count = await ingester.ingest()
    logger.info("Git docs sync completed", ingested=count)
    return count


async def run_feedback_aggregation(ctx):
    """Worker: Aggregate feedback and adjust scores."""
    from src.feedback.adjuster import FeedbackAdjuster

    logger.info("Running feedback aggregation")
    adjuster = FeedbackAdjuster()
    updated = await adjuster.adjust_all()
    logger.info("Feedback aggregation completed", updated=updated)
    return updated


async def run_cache_warming(ctx):
    """Worker: Warm cache with top queries."""
    from src.cache.warming import CacheWarmer
    from src.search.engine import SearchEngine

    logger.info("Running cache warming")
    engine = SearchEngine()
    warmer = CacheWarmer(engine)
    warmed = await warmer.warm(top_n=100)
    logger.info("Cache warming completed", warmed=warmed)
    return warmed


class WorkerSettings:
    """ARQ worker settings with cron jobs."""

    functions = [
        run_sync_slack,
        run_sync_confluence,
        run_sync_git,
        run_feedback_aggregation,
        run_cache_warming,
    ]

    cron_jobs = [
        # Slack: every 15 minutes
        {"coroutine": run_sync_slack, "minute": {0, 15, 30, 45}},
        # Confluence: every hour
        {"coroutine": run_sync_confluence, "minute": {5}},
        # Git: every 30 minutes
        {"coroutine": run_sync_git, "minute": {10, 40}},
        # Feedback aggregation: every 6 hours
        {"coroutine": run_feedback_aggregation, "hour": {0, 6, 12, 18}, "minute": {20}},
        # Cache warming: daily at 2 AM
        {"coroutine": run_cache_warming, "hour": {2}, "minute": {0}},
    ]

    redis_settings = get_redis_settings()
    max_jobs = 5
    job_timeout = 600  # 10 minutes
