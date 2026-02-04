from __future__ import annotations

import structlog

from src.ingestion.chunker import SmartChunker
from src.ingestion.embedder import Embedder
from src.ingestion.git_docs import GitDocsIngester

logger = structlog.get_logger()


async def sync_git_repos():
    """Standalone function to sync all configured Git doc repos."""
    chunker = SmartChunker()
    embedder = Embedder()
    ingester = GitDocsIngester(chunker, embedder)

    count = await ingester.ingest()
    logger.info("Git docs sync completed", documents_ingested=count)
    return count
