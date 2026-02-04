from __future__ import annotations

from datetime import datetime

import structlog
from slack_sdk.web.async_client import AsyncWebClient

from src.config import settings
from src.ingestion.base import BaseIngester, RawDocument
from src.ingestion.chunker import SmartChunker
from src.ingestion.embedder import Embedder

logger = structlog.get_logger()


class SlackIngester(BaseIngester):
    def __init__(self, chunker: SmartChunker, embedder: Embedder):
        super().__init__(chunker, embedder)
        self.client = AsyncWebClient(token=settings.slack_bot_token)
        self.channels = settings.slack_channel_ids

    @property
    def source_type(self) -> str:
        return "slack"

    async def fetch(self, since: datetime | None = None) -> list[RawDocument]:
        docs: list[RawDocument] = []

        for channel_id in self.channels:
            channel_docs = await self._fetch_channel(channel_id, since)
            docs.extend(channel_docs)

        logger.info("Fetched Slack messages", channels=len(self.channels), documents=len(docs))
        return docs

    async def _fetch_channel(
        self, channel_id: str, since: datetime | None
    ) -> list[RawDocument]:
        docs: list[RawDocument] = []

        # Get sync cursor (oldest timestamp we've synced from)
        cursor_str = await self.get_sync_cursor(channel_id)
        oldest = cursor_str or ("0" if since is None else str(since.timestamp()))

        # Fetch channel info for context
        channel_info = await self.client.conversations_info(channel=channel_id)
        channel_name = channel_info["channel"].get("name", channel_id)

        # Paginate through channel history
        latest_ts = "0"
        cursor = None

        while True:
            kwargs = {
                "channel": channel_id,
                "oldest": oldest,
                "limit": 200,
                "inclusive": False,
            }
            if cursor:
                kwargs["cursor"] = cursor

            response = await self.client.conversations_history(**kwargs)
            messages = response.get("messages", [])

            for msg in messages:
                # Skip non-user messages
                if msg.get("subtype") in (
                    "channel_join",
                    "channel_leave",
                    "bot_message",
                    "channel_topic",
                    "channel_purpose",
                ):
                    continue

                # Skip messages without text
                if not msg.get("text", "").strip():
                    continue

                ts = msg["ts"]
                if float(ts) > float(latest_ts):
                    latest_ts = ts

                # Fetch thread replies if thread exists
                thread_ts = msg.get("thread_ts", ts)
                thread_text = msg["text"]
                participants = {msg.get("user", "unknown")}

                if msg.get("reply_count", 0) > 0:
                    thread_text, thread_participants = await self._fetch_thread(
                        channel_id, thread_ts
                    )
                    participants.update(thread_participants)

                # Build document
                doc = RawDocument(
                    source_type="slack",
                    source_id=f"slack:{channel_id}:{thread_ts}",
                    title=f"#{channel_name} - {self._truncate(msg['text'], 100)}",
                    content=thread_text,
                    url=f"https://slack.com/archives/{channel_id}/p{thread_ts.replace('.', '')}",
                    metadata={
                        "channel_id": channel_id,
                        "channel_name": channel_name,
                        "thread_ts": thread_ts,
                        "participants": list(participants),
                        "reply_count": msg.get("reply_count", 0),
                        "reactions": self._extract_reactions(msg),
                    },
                    timestamp=datetime.fromtimestamp(float(ts)),
                )
                docs.append(doc)

            # Pagination
            response_metadata = response.get("response_metadata", {})
            cursor = response_metadata.get("next_cursor")
            if not cursor:
                break

        # Update sync cursor
        if latest_ts != "0":
            await self.update_sync_cursor(channel_id, latest_ts)

        return docs

    async def _fetch_thread(
        self, channel_id: str, thread_ts: str
    ) -> tuple[str, set[str]]:
        """Fetch all replies in a thread and combine into a single text."""
        parts: list[str] = []
        participants: set[str] = set()
        cursor = None

        while True:
            kwargs = {"channel": channel_id, "ts": thread_ts, "limit": 200}
            if cursor:
                kwargs["cursor"] = cursor

            response = await self.client.conversations_replies(**kwargs)
            messages = response.get("messages", [])

            for msg in messages:
                if msg.get("subtype"):
                    continue
                user = msg.get("user", "unknown")
                text = msg.get("text", "").strip()
                if text:
                    parts.append(f"[{user}]: {text}")
                    participants.add(user)

            response_metadata = response.get("response_metadata", {})
            cursor = response_metadata.get("next_cursor")
            if not cursor:
                break

        return "\n\n".join(parts), participants

    @staticmethod
    def _extract_reactions(msg: dict) -> list[dict]:
        reactions = msg.get("reactions", [])
        return [{"name": r["name"], "count": r["count"]} for r in reactions]

    @staticmethod
    def _truncate(text: str, max_len: int) -> str:
        text = text.replace("\n", " ").strip()
        return text[:max_len] + "..." if len(text) > max_len else text
