from __future__ import annotations

import hashlib
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime

import structlog

from src.db.models import Document, Chunk, SyncState
from src.db.session import async_session
from src.ingestion.chunker import SmartChunker
from src.ingestion.embedder import Embedder

from sqlalchemy import select

logger = structlog.get_logger()


@dataclass
class RawDocument:
    source_type: str
    source_id: str
    title: str
    content: str
    url: str | None = None
    metadata: dict = field(default_factory=dict)
    timestamp: datetime | None = None


class BaseIngester(ABC):
    def __init__(self, chunker: SmartChunker, embedder: Embedder):
        self.chunker = chunker
        self.embedder = embedder

    @abstractmethod
    async def fetch(self, since: datetime | None = None) -> list[RawDocument]:
        """Fetch documents from the source. If `since` is provided, only fetch updated docs."""

    async def ingest(self, since: datetime | None = None) -> int:
        """Full ingestion pipeline: fetch → chunk → embed → store. Returns count of new/updated docs."""
        raw_docs = await self.fetch(since=since)
        if not raw_docs:
            logger.info("No new documents to ingest", source=self.source_type)
            return 0

        ingested = 0
        for raw_doc in raw_docs:
            content_hash = hashlib.sha256(raw_doc.content.encode()).hexdigest()

            async with async_session() as session:
                # Check for existing document
                stmt = select(Document).where(
                    Document.source_type == raw_doc.source_type,
                    Document.source_id == raw_doc.source_id,
                )
                result = await session.execute(stmt)
                existing = result.scalar_one_or_none()

                if existing and existing.content_hash == content_hash:
                    # Content unchanged, skip
                    existing.last_synced_at = datetime.utcnow()
                    await session.commit()
                    continue

                if existing:
                    # Content changed, remove old chunks from Qdrant
                    old_point_ids = [
                        c.qdrant_point_id for c in existing.chunks if c.qdrant_point_id
                    ]
                    if old_point_ids:
                        await self.embedder.delete_points(old_point_ids)
                    # Delete old chunks from DB
                    for chunk in existing.chunks:
                        await session.delete(chunk)
                    doc = existing
                    doc.title = raw_doc.title
                    doc.url = raw_doc.url
                    doc.raw_content = raw_doc.content
                    doc.content_hash = content_hash
                    doc.metadata_json = raw_doc.metadata
                    doc.updated_at = datetime.utcnow()
                    doc.last_synced_at = datetime.utcnow()
                else:
                    doc = Document(
                        id=uuid.uuid4(),
                        source_type=raw_doc.source_type,
                        source_id=raw_doc.source_id,
                        title=raw_doc.title,
                        url=raw_doc.url,
                        raw_content=raw_doc.content,
                        content_hash=content_hash,
                        metadata_json=raw_doc.metadata,
                    )
                    session.add(doc)

                # Chunk the document
                text_chunks = self.chunker.chunk(
                    content=raw_doc.content,
                    source_type=raw_doc.source_type,
                    metadata=raw_doc.metadata,
                )

                # Embed and upsert to Qdrant
                point_ids = await self.embedder.embed_and_upsert(
                    chunks=text_chunks,
                    document_id=str(doc.id),
                    source_type=raw_doc.source_type,
                    title=raw_doc.title,
                    url=raw_doc.url,
                    metadata=raw_doc.metadata,
                )

                # Store chunks in DB
                db_chunks = []
                for i, (tc, point_id) in enumerate(zip(text_chunks, point_ids)):
                    db_chunks.append(
                        Chunk(
                            id=uuid.uuid4(),
                            document_id=doc.id,
                            content=tc.text,
                            chunk_index=i,
                            token_count=tc.token_count,
                            metadata_json=tc.metadata,
                            qdrant_point_id=point_id,
                        )
                    )
                session.add_all(db_chunks)
                doc.chunk_count = len(db_chunks)

                await session.commit()
                ingested += 1
                logger.info(
                    "Ingested document",
                    source=raw_doc.source_type,
                    title=raw_doc.title,
                    chunks=len(db_chunks),
                )

        return ingested

    @property
    @abstractmethod
    def source_type(self) -> str:
        """Return the source type identifier."""

    async def get_sync_cursor(self, source_id: str) -> str | None:
        async with async_session() as session:
            stmt = select(SyncState).where(
                SyncState.source_type == self.source_type,
                SyncState.source_id == source_id,
            )
            result = await session.execute(stmt)
            state = result.scalar_one_or_none()
            return state.last_cursor if state else None

    async def update_sync_cursor(self, source_id: str, cursor: str):
        async with async_session() as session:
            stmt = select(SyncState).where(
                SyncState.source_type == self.source_type,
                SyncState.source_id == source_id,
            )
            result = await session.execute(stmt)
            state = result.scalar_one_or_none()

            if state:
                state.last_cursor = cursor
                state.last_sync_at = datetime.utcnow()
            else:
                state = SyncState(
                    source_type=self.source_type,
                    source_id=source_id,
                    last_cursor=cursor,
                    last_sync_at=datetime.utcnow(),
                )
                session.add(state)

            await session.commit()
