from __future__ import annotations

import asyncio
import re
import uuid

import structlog
from qdrant_client.models import PointStruct, SparseVector

from src.config import settings
from src.dependencies import get_openai_client, get_qdrant_client
from src.ingestion.chunker import TextChunk

logger = structlog.get_logger()

# Common English stop words to exclude from sparse vectors
_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "out", "off", "over",
    "under", "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "each", "every", "both", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "just", "because", "but", "and",
    "or", "if", "while", "about", "up", "it", "its", "this", "that",
    "these", "those", "he", "she", "they", "we", "you", "i", "me", "my",
    "his", "her", "our", "your", "what", "which", "who", "whom",
})

# Tokenization pattern: word chars, hyphens, dots (for URLs/versions)
_WORD_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]*[a-z0-9]|[a-z0-9]+", re.IGNORECASE)

MAX_RETRIES = 3
BASE_DELAY = 1.0  # seconds


def _tokenize_for_sparse(text: str) -> dict[int, float]:
    """Tokenize text for sparse vectors with bigrams and stop-word removal.

    Produces both unigrams and bigrams for better keyword matching.
    Qdrant's IDF modifier handles the IDF part server-side.
    """
    words = _WORD_PATTERN.findall(text.lower())
    # Filter stop words and very short tokens
    words = [w for w in words if w not in _STOP_WORDS and len(w) >= 2]

    freq: dict[str, int] = {}

    # Unigrams
    for w in words:
        freq[w] = freq.get(w, 0) + 1

    # Bigrams (captures phrases like "password reset", "api key")
    for i in range(len(words) - 1):
        bigram = f"{words[i]}_{words[i+1]}"
        freq[bigram] = freq.get(bigram, 0) + 1

    # Map to integer indices via hash (consistent across runs)
    indices: dict[int, float] = {}
    for term, count in freq.items():
        idx = abs(hash(term)) % (2**31)
        # If collision, add (rare but possible)
        indices[idx] = indices.get(idx, 0.0) + float(count)
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
        """Embed a list of texts using OpenAI API with batching and retry."""
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            embeddings = await self._embed_batch_with_retry(batch)
            all_embeddings.extend(embeddings)

            if i + self.batch_size < len(texts):
                await asyncio.sleep(0.1)  # Rate limiting between batches

        return all_embeddings

    async def _embed_batch_with_retry(self, texts: list[str]) -> list[list[float]]:
        """Embed a single batch with exponential backoff retry."""
        last_error = None

        for attempt in range(MAX_RETRIES):
            try:
                response = await self.client.embeddings.create(
                    input=texts,
                    model=self.model,
                    dimensions=self.dimensions,
                )
                return [item.embedding for item in response.data]

            except Exception as e:
                last_error = e
                error_str = str(e).lower()

                # Token limit exceeded: try to recover by truncating the offending text
                if "maximum context length" in error_str or "too many tokens" in error_str:
                    logger.warning(
                        "Batch exceeded token limit, truncating texts",
                        batch_size=len(texts),
                        attempt=attempt,
                    )
                    texts = [self._truncate_to_token_limit(t) for t in texts]
                    continue

                # Rate limit: back off
                if "rate" in error_str or "429" in error_str:
                    delay = BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        "Rate limited, backing off",
                        delay=delay,
                        attempt=attempt,
                    )
                    await asyncio.sleep(delay)
                    continue

                # Transient error: retry with backoff
                if attempt < MAX_RETRIES - 1:
                    delay = BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        "Embedding failed, retrying",
                        error=str(e),
                        delay=delay,
                        attempt=attempt,
                    )
                    await asyncio.sleep(delay)
                else:
                    raise

        raise last_error  # type: ignore

    def _truncate_to_token_limit(self, text: str, max_tokens: int = 8000) -> str:
        """Truncate text to fit within the embedding model's token limit."""
        import tiktoken

        encoder = tiktoken.encoding_for_model("gpt-4o")
        tokens = encoder.encode(text)
        if len(tokens) <= max_tokens:
            return text
        truncated = encoder.decode(tokens[:max_tokens])
        logger.debug("Truncated text for embedding", original_tokens=len(tokens), truncated_to=max_tokens)
        return truncated

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

            # Build sparse vector with improved tokenization
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
            await self._upsert_with_retry(batch)

        logger.info(
            "Upserted to Qdrant",
            collection=self.collection,
            count=len(points),
            document_id=document_id,
        )

        return point_ids

    async def _upsert_with_retry(self, points: list[PointStruct]):
        """Upsert to Qdrant with retry on transient failures."""
        for attempt in range(MAX_RETRIES):
            try:
                await self.qdrant.upsert(
                    collection_name=self.collection,
                    points=points,
                )
                return
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    delay = BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        "Qdrant upsert failed, retrying",
                        error=str(e),
                        attempt=attempt,
                        delay=delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    raise

    async def delete_points(self, point_ids: list[str]):
        """Delete points from Qdrant by their IDs."""
        if not point_ids:
            return
        # Batch deletes too (Qdrant can handle up to ~1000 per call)
        for i in range(0, len(point_ids), 500):
            batch = point_ids[i : i + 500]
            await self.qdrant.delete(
                collection_name=self.collection,
                points_selector=batch,
            )
        logger.info("Deleted from Qdrant", count=len(point_ids))
