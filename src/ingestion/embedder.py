from __future__ import annotations

import asyncio
import uuid

import structlog
from qdrant_client.models import PointStruct, SparseVector

from src.config import settings
from src.dependencies import get_openai_client, get_qdrant_client
from src.ingestion.chunker import TextChunk

logger = structlog.get_logger()


def _tokenize_for_sparse(text: str) -> dict[int, float]:
    """Simple whitespace tokenization with term frequency for sparse vectors.

    Qdrant's IDF modifier will handle the IDF part server-side.
    """
    words = text.lower().split()
    freq: dict[str, int] = {}
    for w in words:
        # Strip punctuation
        w = w.strip(".,!?;:\"'()[]{}")
        if len(w) < 2:
            continue
        freq[w] = freq.get(w, 0) + 1

    # Map words to integer indices via hash
    indices: dict[int, float] = {}
    for word, count in freq.items():
        idx = abs(hash(word)) % (2**31)
        indices[idx] = float(count)
    return indices


class Embedder:
    def __init__(self):
        self.client = get_openai_client()
        self.qdrant = get_qdrant_client()
        self.model = settings.embedding_model
        self.dimensions = settings.embedding_dimensions
        self.batch_size = settings.embedding_batch_size
        self.collection = settings.qdrant_collection

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts using OpenAI API with batching."""
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            response = await self.client.embeddings.create(
                input=batch,
                model=self.model,
                dimensions=self.dimensions,
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

            if i + self.batch_size < len(texts):
                await asyncio.sleep(0.1)  # Rate limiting

        return all_embeddings

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        result = await self.embed_texts([query])
        return result[0]

    async def embed_and_upsert(
        self,
        chunks: list[TextChunk],
        document_id: str,
        source_type: str,
        title: str,
        url: str | None,
        metadata: dict | None = None,
    ) -> list[str]:
        """Embed chunks and upsert to Qdrant. Returns list of point IDs."""
        if not chunks:
            return []

        texts = [c.text for c in chunks]
        embeddings = await self.embed_texts(texts)

        points = []
        point_ids = []

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point_id = str(uuid.uuid4())
            point_ids.append(point_id)

            # Build sparse vector
            sparse_data = _tokenize_for_sparse(chunk.text)

            payload = {
                "document_id": document_id,
                "source_type": source_type,
                "title": title,
                "url": url or "",
                "chunk_index": i,
                "text": chunk.text,
                "token_count": chunk.token_count,
                "feedback_score": 0.0,
                **(metadata or {}),
                **(chunk.metadata or {}),
            }

            points.append(
                PointStruct(
                    id=point_id,
                    vector={
                        "dense": embedding,
                        "sparse": SparseVector(
                            indices=list(sparse_data.keys()),
                            values=list(sparse_data.values()),
                        ),
                    },
                    payload=payload,
                )
            )

        # Upsert in batches of 100
        for i in range(0, len(points), 100):
            batch = points[i : i + 100]
            await self.qdrant.upsert(
                collection_name=self.collection,
                points=batch,
            )

        logger.info(
            "Upserted to Qdrant",
            collection=self.collection,
            count=len(points),
            document_id=document_id,
        )

        return point_ids

    async def delete_points(self, point_ids: list[str]):
        """Delete points from Qdrant by their IDs."""
        if not point_ids:
            return
        await self.qdrant.delete(
            collection_name=self.collection,
            points_selector=point_ids,
        )
        logger.info("Deleted from Qdrant", count=len(point_ids))
