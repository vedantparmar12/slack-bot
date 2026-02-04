from __future__ import annotations

import uuid
from datetime import datetime

import structlog
from sqlalchemy import select

from src.db.models import Feedback
from src.db.session import async_session

logger = structlog.get_logger()


class FeedbackCollector:
    """Collect and store user feedback on search results."""

    async def record(
        self,
        query_text: str,
        chunk_id: str,
        user_id: str,
        vote: int,
    ) -> str:
        """Record a feedback vote. Returns feedback ID. Latest vote wins per user-chunk-query."""
        async with async_session() as session:
            stmt = select(Feedback).where(
                Feedback.chunk_id == uuid.UUID(chunk_id),
                Feedback.user_id == user_id,
                Feedback.query_text == query_text,
            )
            result = await session.execute(stmt)
            existing = result.scalar_one_or_none()

            if existing:
                existing.vote = vote
                existing.created_at = datetime.utcnow()
                feedback_id = str(existing.id)
            else:
                fb = Feedback(
                    query_text=query_text,
                    chunk_id=uuid.UUID(chunk_id),
                    user_id=user_id,
                    vote=vote,
                )
                session.add(fb)
                feedback_id = str(fb.id)

            await session.commit()

        logger.info(
            "Feedback recorded",
            chunk_id=chunk_id,
            user=user_id,
            vote=vote,
        )
        return feedback_id

    async def get_feedback_for_chunk(self, chunk_id: str) -> list[Feedback]:
        """Get all feedback records for a specific chunk."""
        async with async_session() as session:
            stmt = select(Feedback).where(
                Feedback.chunk_id == uuid.UUID(chunk_id)
            ).order_by(Feedback.created_at.desc())
            result = await session.execute(stmt)
            return list(result.scalars().all())
