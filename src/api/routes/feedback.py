from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from src.api.schemas import FeedbackRequest, FeedbackResponse
from src.db.models import Feedback
from src.db.session import async_session

router = APIRouter()


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    if request.vote not in (1, -1):
        raise HTTPException(status_code=400, detail="Vote must be 1 or -1")

    async with async_session() as session:
        # Upsert: latest vote wins per user-chunk-query triple
        stmt = select(Feedback).where(
            Feedback.chunk_id == uuid.UUID(request.chunk_id),
            Feedback.query_text == request.query_id,
            Feedback.user_id == "api_user",  # In real usage, extract from auth
        )
        result = await session.execute(stmt)
        existing = result.scalar_one_or_none()

        if existing:
            existing.vote = request.vote
            feedback_id = str(existing.id)
        else:
            fb = Feedback(
                query_text=request.query_id,
                chunk_id=uuid.UUID(request.chunk_id),
                user_id="api_user",
                vote=request.vote,
            )
            session.add(fb)
            feedback_id = str(fb.id)

        await session.commit()

    return FeedbackResponse(id=feedback_id, status="recorded")
