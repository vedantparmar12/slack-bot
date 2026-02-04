from __future__ import annotations

import structlog
from sqlalchemy import select, func

from src.config import settings
from src.db.models import Chunk, Feedback
from src.db.session import async_session
from src.dependencies import get_qdrant_client
from src.feedback.scorer import FeedbackScorer

logger = structlog.get_logger()


class FeedbackAdjuster:
    """Periodically recompute feedback scores and update Qdrant payloads."""

    def __init__(self):
        self.scorer = FeedbackScorer()
        self.qdrant = get_qdrant_client()
        self.collection = settings.qdrant_collection

    async def adjust_all(self) -> int:
        """Recompute scores for all chunks that have feedback. Returns count of updated chunks."""
        updated = 0

        async with async_session() as session:
            # Find all chunks that have at least one feedback entry
            stmt = (
                select(Chunk.id, Chunk.qdrant_point_id)
                .join(Feedback, Feedback.chunk_id == Chunk.id)
                .group_by(Chunk.id, Chunk.qdrant_point_id)
                .having(func.count(Feedback.id) >= 1)
            )
            result = await session.execute(stmt)
            chunks_with_feedback = result.all()

        for chunk_id, qdrant_point_id in chunks_with_feedback:
            if not qdrant_point_id:
                continue

            async with async_session() as session:
                stmt = select(Feedback).where(Feedback.chunk_id == chunk_id)
                result = await session.execute(stmt)
                feedbacks = list(result.scalars().all())

            new_score = self.scorer.compute_score(feedbacks)

            # Update Qdrant payload
            try:
                await self.qdrant.set_payload(
                    collection_name=self.collection,
                    points=[qdrant_point_id],
                    payload={"feedback_score": new_score},
                )
                updated += 1
            except Exception as e:
                logger.warning(
                    "Failed to update Qdrant payload",
                    point_id=qdrant_point_id,
                    error=str(e),
                )

        logger.info("Feedback adjustment completed", updated=updated)

        # Flag consistently negative documents
        await self._flag_negative_documents()

        return updated

    async def _flag_negative_documents(self):
        """Log documents with consistently negative feedback for review."""
        async with async_session() as session:
            stmt = (
                select(
                    Chunk.document_id,
                    func.avg(Feedback.vote).label("avg_vote"),
                    func.count(Feedback.id).label("vote_count"),
                )
                .join(Feedback, Feedback.chunk_id == Chunk.id)
                .group_by(Chunk.document_id)
                .having(
                    func.count(Feedback.id) >= self.scorer.min_votes,
                    func.avg(Feedback.vote) < -0.5,
                )
            )
            result = await session.execute(stmt)
            negative_docs = result.all()

        for doc_id, avg_vote, vote_count in negative_docs:
            logger.warning(
                "Document has consistently negative feedback",
                document_id=str(doc_id),
                avg_vote=round(float(avg_vote), 2),
                vote_count=vote_count,
            )
