from __future__ import annotations

import structlog

from src.feedback.adjuster import FeedbackAdjuster

logger = structlog.get_logger()


async def aggregate_feedback():
    """Standalone function to aggregate feedback and adjust Qdrant scores."""
    adjuster = FeedbackAdjuster()
    updated = await adjuster.adjust_all()
    logger.info("Feedback aggregation completed", chunks_updated=updated)
    return updated
