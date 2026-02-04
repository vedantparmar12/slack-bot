from __future__ import annotations

from fastapi import APIRouter

from src.api.schemas import SearchRequest, SearchResponse
from src.search.engine import SearchEngine

router = APIRouter()

_engine: SearchEngine | None = None


def _get_engine() -> SearchEngine:
    global _engine
    if _engine is None:
        _engine = SearchEngine()
    return _engine


@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    engine = _get_engine()
    return await engine.search(
        query=request.query,
        filters=request.filters,
        limit=request.limit,
        source="api",
    )
