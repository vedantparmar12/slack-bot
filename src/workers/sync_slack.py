from __future__ import annotations

import structlog

from src.ingestion.chunker import SmartChunker
from src.ingestion.embedder import Embedder
from src.ingestion.slack import SlackIngester

logger = structlog.get_logger()


async def sync_slack_channels():
    """Standalone function to sync all configured Slack channels."""
    chunker = SmartChunker()
    embedder = Embedder()
    ingester = SlackIngester(chunker, embedder)

    count = await ingester.ingest()
    logger.info("Slack sync completed", documents_ingested=count)
    return count
