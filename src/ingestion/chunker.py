from __future__ import annotations

import re
from dataclasses import dataclass, field

import tiktoken

from src.config import settings

# OpenAI embedding model max input tokens
EMBEDDING_MAX_TOKENS = 8191
# Safety margin below the hard limit
EMBEDDING_SAFE_LIMIT = 8000
# Maximum document size we'll process (in characters, ~500k tokens)
MAX_DOCUMENT_CHARS = 2_000_000

# Sentence splitting pattern (handles ., !, ?, and common abbreviations)
_SENTENCE_SPLIT = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z])"
    r"|(?<=[.!?])\s*\n"
)


@dataclass
class TextChunk:
    text: str
    token_count: int
    metadata: dict = field(default_factory=dict)


class SmartChunker:
    """Hybrid chunker: hierarchy-aware for docs, conversation-aware for Slack,
    with code block preservation and embedding-safe token limits."""

    def __init__(
        self,
        target_tokens: int = 800,
        max_tokens: int = 1500,
        overlap_tokens: int = 200,
    ):
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        # Hard ceiling: chunks must never exceed embedding model input limit
        self.absolute_max = EMBEDDING_SAFE_LIMIT
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

        # Guard: truncate absurdly large documents
        if len(content) > MAX_DOCUMENT_CHARS:
            content = content[:MAX_DOCUMENT_CHARS]
            metadata["truncated"] = True

        if source_type == "slack":
            return self._chunk_slack(content, metadata)
        elif source_type in ("confluence", "git"):
            return self._chunk_markdown(content, metadata)
        else:
            return self._chunk_plain(content, metadata)

    # ---- Slack: conversation-aware chunking ----

    def _chunk_slack(self, content: str, metadata: dict) -> list[TextChunk]:
        """Split Slack threads preserving Q&A pairs.

        Strategy:
        1. If thread fits in one chunk, keep it whole.
        2. Otherwise, split by message boundaries (double newline),
           grouping consecutive messages into chunks that keep
           question + answer together.
        """
        tokens = self.count_tokens(content)
        if tokens <= self.max_tokens:
            return [
                TextChunk(
                    text=content,
                    token_count=tokens,
                    metadata={**metadata, "chunk_type": "slack_thread"},
                )
            ]

        # Split by individual messages (each message starts with [username]:)
        messages = re.split(r"\n\n(?=\[)", content)
        if not messages or len(messages) == 1:
            # Fallback: split by double newline
            return self._split_with_overlap(content, metadata, chunk_type="slack_thread_part")

        # Group messages into chunks, keeping Q&A pairs together
        chunks: list[TextChunk] = []
        current_msgs: list[str] = []
        current_tokens = 0

        for msg in messages:
            msg = msg.strip()
            if not msg:
                continue
            msg_tokens = self.count_tokens(msg)

            # Single message exceeds limit — split it further
            if msg_tokens > self.max_tokens:
                # Flush buffer
                if current_msgs:
                    chunks.append(self._make_chunk(
                        "\n\n".join(current_msgs), metadata, "slack_thread_part"
                    ))
                    current_msgs = []
                    current_tokens = 0
                # Split the oversized message
                chunks.extend(self._split_with_overlap(
                    msg, metadata, chunk_type="slack_message_part"
                ))
                continue

            if current_tokens + msg_tokens > self.target_tokens and current_msgs:
                chunks.append(self._make_chunk(
                    "\n\n".join(current_msgs), metadata, "slack_thread_part"
                ))

                # Overlap: keep last message as context bridge
                last_msg = current_msgs[-1]
                last_tokens = self.count_tokens(last_msg)
                if last_tokens <= self.overlap_tokens:
                    current_msgs = [last_msg]
                    current_tokens = last_tokens
                else:
                    current_msgs = []
                    current_tokens = 0

            current_msgs.append(msg)
            current_tokens += msg_tokens

        if current_msgs:
            chunks.append(self._make_chunk(
                "\n\n".join(current_msgs), metadata, "slack_thread_part"
            ))

        return chunks if chunks else [self._make_chunk(content, metadata, "slack_thread")]

    # ---- Markdown: hierarchy + code-block aware chunking ----

    def _chunk_markdown(self, content: str, metadata: dict) -> list[TextChunk]:
        """Split on markdown headings, preserving code blocks intact."""
        # Step 1: Extract and protect code blocks
        content, code_blocks = self._protect_code_blocks(content)

        # Step 2: Split by headings (H1–H4)
        heading_pattern = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)
        sections: list[tuple[list[str], str]] = []
        last_end = 0
        heading_chain: list[str] = []

        for match in heading_pattern.finditer(content):
            if last_end < match.start():
                text = content[last_end : match.start()].strip()
                if text:
                    sections.append((list(heading_chain), text))

            level = len(match.group(1))
            title = match.group(2).strip()
            heading_chain = heading_chain[: level - 1]
            while len(heading_chain) < level:
                heading_chain.append("")
            heading_chain[level - 1] = title
            last_end = match.end()

        remaining = content[last_end:].strip()
        if remaining:
            sections.append((list(heading_chain), remaining))

        if not sections:
            restored = self._restore_code_blocks(content, code_blocks)
            return self._chunk_plain(restored, metadata)

        # Step 3: Merge small sections, split large ones
        chunks: list[TextChunk] = []
        buffer_text = ""
        buffer_headings: list[str] = []

        for headings, text in sections:
            # Restore code blocks in this section
            text = self._restore_code_blocks(text, code_blocks)

            combined = f"{buffer_text}\n\n{text}".strip() if buffer_text else text
            combined_tokens = self.count_tokens(combined)

            if combined_tokens <= self.target_tokens:
                buffer_text = combined
                buffer_headings = headings or buffer_headings
            else:
                if buffer_text:
                    self._flush_section(
                        buffer_text, buffer_headings, metadata, chunks
                    )

                text_tokens = self.count_tokens(text)
                if text_tokens <= self.target_tokens:
                    buffer_text = text
                    buffer_headings = headings
                else:
                    self._flush_section(text, headings, metadata, chunks)
                    buffer_text = ""
                    buffer_headings = []

        if buffer_text:
            self._flush_section(buffer_text, buffer_headings, metadata, chunks)

        return chunks if chunks else self._chunk_plain(
            self._restore_code_blocks(content, code_blocks), metadata
        )

    def _flush_section(
        self,
        text: str,
        headings: list[str],
        metadata: dict,
        chunks: list[TextChunk],
    ):
        """Add a section as one or more chunks, splitting if too large."""
        tokens = self.count_tokens(text)
        if tokens <= self.max_tokens:
            chunks.append(
                TextChunk(
                    text=text,
                    token_count=tokens,
                    metadata={
                        **metadata,
                        "heading_chain": headings,
                        "chunk_type": "markdown_section",
                    },
                )
            )
        else:
            chunks.extend(
                self._split_with_overlap(
                    text,
                    {**metadata, "heading_chain": headings},
                    chunk_type="markdown_section_part",
                )
            )

    @staticmethod
    def _protect_code_blocks(text: str) -> tuple[str, dict[str, str]]:
        """Replace fenced code blocks with placeholders so they aren't split mid-block."""
        code_blocks: dict[str, str] = {}
        counter = 0

        def replacer(match):
            nonlocal counter
            key = f"__CODE_BLOCK_{counter}__"
            code_blocks[key] = match.group(0)
            counter += 1
            return key

        protected = re.sub(r"```[\s\S]*?```", replacer, text)
        return protected, code_blocks

    @staticmethod
    def _restore_code_blocks(text: str, code_blocks: dict[str, str]) -> str:
        """Restore code block placeholders to their original content."""
        for key, code in code_blocks.items():
            text = text.replace(key, code)
        return text

    # ---- Plain text / fallback ----

    def _chunk_plain(self, content: str, metadata: dict) -> list[TextChunk]:
        return self._split_with_overlap(content, metadata, chunk_type="plain")

    def _split_with_overlap(
        self,
        text: str,
        metadata: dict,
        chunk_type: str = "plain",
    ) -> list[TextChunk]:
        """Split text into overlapping chunks by paragraph/sentence boundaries.

        Falls back to sentence splitting for oversized paragraphs,
        and to hard token slicing as a last resort.
        """
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        if not paragraphs:
            return [self._make_chunk(text, metadata, chunk_type)]

        # Expand: if any paragraph exceeds max_tokens, split it by sentences
        expanded: list[str] = []
        for para in paragraphs:
            para_tokens = self.count_tokens(para)
            if para_tokens <= self.max_tokens:
                expanded.append(para)
            else:
                # Try sentence splitting
                sentences = _SENTENCE_SPLIT.split(para)
                if len(sentences) > 1:
                    for sent in sentences:
                        sent = sent.strip()
                        if sent:
                            expanded.append(sent)
                else:
                    # Last resort: hard-split by tokens
                    expanded.extend(self._hard_split(para))

        chunks: list[TextChunk] = []
        current_parts: list[str] = []
        current_tokens = 0

        for part in expanded:
            part_tokens = self.count_tokens(part)

            # Single part still over absolute max → hard split
            if part_tokens > self.absolute_max:
                if current_parts:
                    chunks.append(self._make_chunk(
                        "\n\n".join(current_parts), metadata, chunk_type
                    ))
                    current_parts = []
                    current_tokens = 0
                for sub in self._hard_split(part):
                    chunks.append(self._make_chunk(sub, metadata, chunk_type))
                continue

            if current_tokens + part_tokens > self.target_tokens and current_parts:
                chunks.append(self._make_chunk(
                    "\n\n".join(current_parts), metadata, chunk_type
                ))

                # Keep overlap
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

            current_parts.append(part)
            current_tokens += part_tokens

        if current_parts:
            chunks.append(self._make_chunk(
                "\n\n".join(current_parts), metadata, chunk_type
            ))

        return chunks

    def _hard_split(self, text: str) -> list[str]:
        """Split text into pieces that each fit within absolute_max tokens.
        Uses token-level slicing as a last resort for content that has
        no paragraph or sentence boundaries (e.g. minified code, base64).
        """
        tokens = self.encoder.encode(text)
        pieces = []
        for i in range(0, len(tokens), self.absolute_max):
            chunk_tokens = tokens[i : i + self.absolute_max]
            pieces.append(self.encoder.decode(chunk_tokens))
        return pieces

    def _make_chunk(self, text: str, metadata: dict, chunk_type: str) -> TextChunk:
        """Create a TextChunk, enforcing the embedding token safety limit."""
        token_count = self.count_tokens(text)
        if token_count > self.absolute_max:
            # Truncate to fit (should rarely happen due to _hard_split)
            tokens = self.encoder.encode(text)[: self.absolute_max]
            text = self.encoder.decode(tokens)
            token_count = self.absolute_max
        return TextChunk(
            text=text,
            token_count=token_count,
            metadata={**metadata, "chunk_type": chunk_type},
        )
