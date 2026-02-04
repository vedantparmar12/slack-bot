from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


# --- Search ---


class SearchFilters(BaseModel):
    source_types: list[str] | None = None
    tags: list[str] | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    filters: SearchFilters | None = None
    limit: int = Field(5, ge=1, le=20)
    offset: int = Field(0, ge=0)


class SourceResult(BaseModel):
    chunk_id: str
    title: str
    url: str | None
    snippet: str
    score: float
    source_type: str
    feedback_score: float = 0.0


class SearchResponse(BaseModel):
    answer: str
    sources: list[SourceResult]
    query_id: str
    cached: bool = False
    latency_ms: float
    confidence: str  # high, medium, low


# --- Feedback ---


class FeedbackRequest(BaseModel):
    query_id: str
    chunk_id: str
    vote: int = Field(..., ge=-1, le=1)  # +1 or -1


class FeedbackResponse(BaseModel):
    id: str
    status: str = "recorded"


# --- Ingest ---


class IngestTriggerRequest(BaseModel):
    source_type: str = Field(..., pattern="^(slack|confluence|git)$")
    source_id: str | None = None


class IngestStatusResponse(BaseModel):
    source_type: str
    source_id: str
    last_sync_at: datetime | None
    status: str  # idle, syncing, error


# --- Documents ---


class DocumentSummary(BaseModel):
    id: UUID
    source_type: str
    title: str
    url: str | None
    chunk_count: int
    updated_at: datetime


class DocumentDetail(BaseModel):
    id: UUID
    source_type: str
    source_id: str
    title: str
    url: str | None
    chunk_count: int
    metadata: dict | None
    created_at: datetime
    updated_at: datetime


class PaginatedResponse(BaseModel):
    items: list
    total: int
    offset: int
    limit: int


# --- Health ---


class HealthResponse(BaseModel):
    status: str
    version: str
    dependencies: dict[str, str]


class MetricsResponse(BaseModel):
    total_documents: int
    total_chunks: int
    total_queries: int
    cache_hit_rate: float
    avg_latency_ms: float
    feedback_count: int
