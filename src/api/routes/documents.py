from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, HTTPException
from sqlalchemy import func, select

from src.api.schemas import DocumentDetail, DocumentSummary, PaginatedResponse
from src.db.models import Document
from src.db.session import async_session
from src.ingestion.embedder import Embedder

router = APIRouter()


@router.get("/documents", response_model=PaginatedResponse)
async def list_documents(
    offset: int = 0,
    limit: int = 20,
    source_type: str | None = None,
):
    async with async_session() as session:
        stmt = select(Document).where(Document.is_deleted == False)  # noqa: E712
        count_stmt = select(func.count(Document.id)).where(Document.is_deleted == False)  # noqa: E712

        if source_type:
            stmt = stmt.where(Document.source_type == source_type)
            count_stmt = count_stmt.where(Document.source_type == source_type)

        total = (await session.execute(count_stmt)).scalar() or 0

        stmt = stmt.order_by(Document.updated_at.desc()).offset(offset).limit(limit)
        result = await session.execute(stmt)
        docs = result.scalars().all()

    items = [
        DocumentSummary(
            id=d.id,
            source_type=d.source_type,
            title=d.title,
            url=d.url,
            chunk_count=d.chunk_count,
            updated_at=d.updated_at,
        )
        for d in docs
    ]

    return PaginatedResponse(items=items, total=total, offset=offset, limit=limit)


@router.get("/documents/{doc_id}", response_model=DocumentDetail)
async def get_document(doc_id: UUID):
    async with async_session() as session:
        result = await session.execute(
            select(Document).where(Document.id == doc_id, Document.is_deleted == False)  # noqa: E712
        )
        doc = result.scalar_one_or_none()

    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    return DocumentDetail(
        id=doc.id,
        source_type=doc.source_type,
        source_id=doc.source_id,
        title=doc.title,
        url=doc.url,
        chunk_count=doc.chunk_count,
        metadata=doc.metadata_json,
        created_at=doc.created_at,
        updated_at=doc.updated_at,
    )


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: UUID):
    async with async_session() as session:
        result = await session.execute(
            select(Document).where(Document.id == doc_id)
        )
        doc = result.scalar_one_or_none()

        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        # Delete chunks from Qdrant
        point_ids = [c.qdrant_point_id for c in doc.chunks if c.qdrant_point_id]
        if point_ids:
            embedder = Embedder()
            await embedder.delete_points(point_ids)

        # Soft delete
        doc.is_deleted = True
        await session.commit()

    return {"status": "deleted", "document_id": str(doc_id)}
