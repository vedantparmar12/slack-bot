from __future__ import annotations

import math
from datetime import datetime

from src.config import settings
from src.db.models import Feedback


class FeedbackScorer:
    """Compute time-decayed feedback scores for chunks."""

    def __init__(self):
        self.min_votes = settings.feedback_min_votes
        self.decay_rate = settings.feedback_decay_rate  # ~0.05 → half-life ~14 days

    def compute_score(self, feedbacks: list[Feedback]) -> float:
        """Compute a feedback score in [-1, 1] using exponential time decay.

        Returns 0.0 if fewer than min_votes feedbacks exist.
        """
        if len(feedbacks) < self.min_votes:
            return 0.0

        now = datetime.utcnow()
        weighted_votes: list[float] = []
        total_weight = 0.0

        for fb in feedbacks:
            age_days = max(0, (now - fb.created_at).total_seconds() / 86400)
            weight = math.exp(-self.decay_rate * age_days)
            weighted_votes.append(fb.vote * weight)
            total_weight += weight

        if total_weight == 0:
            return 0.0

        raw_score = sum(weighted_votes) / total_weight
        return max(-1.0, min(1.0, raw_score))
