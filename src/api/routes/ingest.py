from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException
from sqlalchemy import select

from src.api.schemas import IngestStatusResponse, IngestTriggerRequest
from src.db.models import SyncState
from src.db.session import async_session
from src.ingestion.chunker import SmartChunker
from src.ingestion.embedder import Embedder

router = APIRouter()

# Track running ingestion jobs
_running_jobs: dict[str, bool] = {}


async def _run_ingestion(source_type: str, source_id: str | None):
    """Run ingestion in background."""
    job_key = f"{source_type}:{source_id or 'all'}"
    _running_jobs[job_key] = True

    try:
        chunker = SmartChunker()
        embedder = Embedder()

        if source_type == "slack":
            from src.ingestion.slack import SlackIngester

            ingester = SlackIngester(chunker, embedder)
        elif source_type == "confluence":
            from src.ingestion.confluence import ConfluenceIngester

            ingester = ConfluenceIngester(chunker, embedder)
        elif source_type == "git":
            from src.ingestion.git_docs import GitDocsIngester

            ingester = GitDocsIngester(chunker, embedder)
        else:
            return

        await ingester.ingest()
    finally:
        _running_jobs.pop(job_key, None)


@router.post("/ingest/trigger")
async def trigger_ingest(request: IngestTriggerRequest, background_tasks: BackgroundTasks):
    job_key = f"{request.source_type}:{request.source_id or 'all'}"

    if _running_jobs.get(job_key):
        raise HTTPException(status_code=409, detail=f"Ingestion already running for {job_key}")

    background_tasks.add_task(_run_ingestion, request.source_type, request.source_id)

    return {"status": "started", "source_type": request.source_type}


@router.get("/ingest/status", response_model=list[IngestStatusResponse])
async def ingest_status():
    async with async_session() as session:
        result = await session.execute(select(SyncState))
        states = result.scalars().all()

    return [
        IngestStatusResponse(
            source_type=s.source_type,
            source_id=s.source_id,
            last_sync_at=s.last_sync_at,
            status="syncing" if f"{s.source_type}:{s.source_id}" in _running_jobs else "idle",
        )
        for s in states
    ]
