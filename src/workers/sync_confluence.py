from __future__ import annotations

import structlog

from src.ingestion.chunker import SmartChunker
from src.ingestion.confluence import ConfluenceIngester
from src.ingestion.embedder import Embedder

logger = structlog.get_logger()


async def sync_confluence_spaces():
    """Standalone function to sync all configured Confluence spaces."""
    chunker = SmartChunker()
    embedder = Embedder()
    ingester = ConfluenceIngester(chunker, embedder)

    count = await ingester.ingest()
    logger.info("Confluence sync completed", documents_ingested=count)
    return count
