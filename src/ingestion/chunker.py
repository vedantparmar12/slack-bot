from __future__ import annotations

import re
from dataclasses import dataclass, field

import tiktoken

from src.config import settings


@dataclass
class TextChunk:
    text: str
    token_count: int
    metadata: dict = field(default_factory=dict)


class SmartChunker:
    """Hierarchy-aware chunker that preserves document structure."""

    def __init__(
        self,
        target_tokens: int = 800,
        max_tokens: int = 1500,
        overlap_tokens: int = 200,
    ):
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.encoder = tiktoken.encoding_for_model("gpt-4o")

    def count_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))

    def chunk(
        self,
        content: str,
        source_type: str,
        metadata: dict | None = None,
    ) -> list[TextChunk]:
        metadata = metadata or {}

        if source_type == "slack":
            return self._chunk_slack(content, metadata)
        elif source_type in ("confluence", "git"):
            return self._chunk_markdown(content, metadata)
        else:
            return self._chunk_plain(content, metadata)

    def _chunk_slack(self, content: str, metadata: dict) -> list[TextChunk]:
        """Keep Slack threads as single chunks when possible."""
        tokens = self.count_tokens(content)

        if tokens <= self.max_tokens:
            return [
                TextChunk(
                    text=content,
                    token_count=tokens,
                    metadata={**metadata, "chunk_type": "slack_thread"},
                )
            ]

        # Thread too long, split by messages (double newline)
        return self._split_with_overlap(content, metadata, chunk_type="slack_thread_part")

    def _chunk_markdown(self, content: str, metadata: dict) -> list[TextChunk]:
        """Split on markdown headings, preserving hierarchy."""
        # Split by headings (H1, H2, H3)
        heading_pattern = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
        sections = []
        last_end = 0
        heading_chain: list[str] = []

        for match in heading_pattern.finditer(content):
            # Save content before this heading
            if last_end < match.start():
                text = content[last_end : match.start()].strip()
                if text:
                    sections.append((list(heading_chain), text))

            level = len(match.group(1))
            title = match.group(2).strip()

            # Update heading chain
            heading_chain = heading_chain[: level - 1]
            while len(heading_chain) < level:
                heading_chain.append("")
            heading_chain[level - 1] = title
            last_end = match.end()

        # Remaining content after last heading
        remaining = content[last_end:].strip()
        if remaining:
            sections.append((list(heading_chain), remaining))

        if not sections:
            return self._chunk_plain(content, metadata)

        # Merge small sections and split large ones
        chunks: list[TextChunk] = []
        buffer_text = ""
        buffer_headings: list[str] = []

        for headings, text in sections:
            combined = f"{buffer_text}\n\n{text}".strip() if buffer_text else text
            combined_tokens = self.count_tokens(combined)

            if combined_tokens <= self.target_tokens:
                buffer_text = combined
                buffer_headings = headings or buffer_headings
            else:
                # Flush buffer if it has content
                if buffer_text:
                    tokens = self.count_tokens(buffer_text)
                    if tokens <= self.max_tokens:
                        chunks.append(
                            TextChunk(
                                text=buffer_text,
                                token_count=tokens,
                                metadata={
                                    **metadata,
                                    "heading_chain": buffer_headings,
                                    "chunk_type": "markdown_section",
                                },
                            )
                        )
                    else:
                        chunks.extend(
                            self._split_with_overlap(
                                buffer_text,
                                {**metadata, "heading_chain": buffer_headings},
                                chunk_type="markdown_section_part",
                            )
                        )

                # Start new buffer
                text_tokens = self.count_tokens(text)
                if text_tokens <= self.target_tokens:
                    buffer_text = text
                    buffer_headings = headings
                else:
                    # Section itself is too large, split it
                    chunks.extend(
                        self._split_with_overlap(
                            text,
                            {**metadata, "heading_chain": headings},
                            chunk_type="markdown_section_part",
                        )
                    )
                    buffer_text = ""
                    buffer_headings = []

        # Flush remaining buffer
        if buffer_text:
            tokens = self.count_tokens(buffer_text)
            chunks.append(
                TextChunk(
                    text=buffer_text,
                    token_count=tokens,
                    metadata={
                        **metadata,
                        "heading_chain": buffer_headings,
                        "chunk_type": "markdown_section",
                    },
                )
            )

        return chunks if chunks else self._chunk_plain(content, metadata)

    def _chunk_plain(self, content: str, metadata: dict) -> list[TextChunk]:
        """Fallback: split by paragraphs with overlap."""
        return self._split_with_overlap(content, metadata, chunk_type="plain")

    def _split_with_overlap(
        self,
        text: str,
        metadata: dict,
        chunk_type: str = "plain",
    ) -> list[TextChunk]:
        """Split text into overlapping chunks by sentence/paragraph boundaries."""
        # Split by paragraphs first, then by sentences
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        if not paragraphs:
            tokens = self.count_tokens(text)
            return [
                TextChunk(
                    text=text,
                    token_count=tokens,
                    metadata={**metadata, "chunk_type": chunk_type},
                )
            ]

        chunks: list[TextChunk] = []
        current_parts: list[str] = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = self.count_tokens(para)

            if current_tokens + para_tokens > self.target_tokens and current_parts:
                # Flush current chunk
                chunk_text = "\n\n".join(current_parts)
                chunks.append(
                    TextChunk(
                        text=chunk_text,
                        token_count=self.count_tokens(chunk_text),
                        metadata={**metadata, "chunk_type": chunk_type},
                    )
                )

                # Keep overlap: find how many trailing paragraphs fit in overlap budget
                overlap_parts: list[str] = []
                overlap_tokens = 0
                for p in reversed(current_parts):
                    pt = self.count_tokens(p)
                    if overlap_tokens + pt > self.overlap_tokens:
                        break
                    overlap_parts.insert(0, p)
                    overlap_tokens += pt

                current_parts = overlap_parts
                current_tokens = overlap_tokens

            current_parts.append(para)
            current_tokens += para_tokens

        # Flush remaining
        if current_parts:
            chunk_text = "\n\n".join(current_parts)
            chunks.append(
                TextChunk(
                    text=chunk_text,
                    token_count=self.count_tokens(chunk_text),
                    metadata={**metadata, "chunk_type": chunk_type},
                )
            )

        return chunks
