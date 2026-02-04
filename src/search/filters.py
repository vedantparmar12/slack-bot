from __future__ import annotations

from datetime import datetime

from qdrant_client.models import FieldCondition, Filter, MatchAny, Range

from src.api.schemas import SearchFilters


def build_qdrant_filter(filters: SearchFilters | None) -> Filter | None:
    """Convert API search filters to Qdrant filter conditions."""
    if not filters:
        return None

    conditions = []

    if filters.source_types:
        conditions.append(
            FieldCondition(
                key="source_type",
                match=MatchAny(any=filters.source_types),
            )
        )

    if filters.date_from or filters.date_to:
        range_kwargs = {}
        if filters.date_from:
            range_kwargs["gte"] = filters.date_from.timestamp()
        if filters.date_to:
            range_kwargs["lte"] = filters.date_to.timestamp()
        if range_kwargs:
            conditions.append(
                FieldCondition(key="timestamp", range=Range(**range_kwargs))
            )

    if filters.tags:
        conditions.append(
            FieldCondition(
                key="labels",
                match=MatchAny(any=filters.tags),
            )
        )

    if not conditions:
        return None

    return Filter(must=conditions)
