from __future__ import annotations

import re
from datetime import datetime

import structlog
from slack_sdk.web.async_client import AsyncWebClient

from src.config import settings
from src.ingestion.base import BaseIngester, RawDocument
from src.ingestion.chunker import SmartChunker
from src.ingestion.embedder import Embedder

logger = structlog.get_logger()

# Slack mrkdwn patterns
_USER_MENTION = re.compile(r"<@(\w+)>")
_CHANNEL_MENTION = re.compile(r"<#(\w+)\|([^>]+)>")
_URL_PATTERN = re.compile(r"<(https?://[^|>]+)(?:\|([^>]+))?>")


class SlackIngester(BaseIngester):
    def __init__(self, chunker: SmartChunker, embedder: Embedder):
        super().__init__(chunker, embedder)
        self.client = AsyncWebClient(token=settings.slack_bot_token)
        self.channels = settings.slack_channel_ids
        self._user_cache: dict[str, str] = {}

    @property
    def source_type(self) -> str:
        return "slack"

    async def fetch(self, since: datetime | None = None) -> list[RawDocument]:
        docs: list[RawDocument] = []

        for channel_id in self.channels:
            try:
                channel_docs = await self._fetch_channel(channel_id, since)
                docs.extend(channel_docs)
            except Exception as e:
                logger.error("Failed to fetch channel", channel=channel_id, error=str(e))

        logger.info("Fetched Slack messages", channels=len(self.channels), documents=len(docs))
        return docs

    async def _fetch_channel(
        self, channel_id: str, since: datetime | None
    ) -> list[RawDocument]:
        docs: list[RawDocument] = []

        cursor_str = await self.get_sync_cursor(channel_id)
        oldest = cursor_str or ("0" if since is None else str(since.timestamp()))

        channel_info = await self.client.conversations_info(channel=channel_id)
        channel_name = channel_info["channel"].get("name", channel_id)

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
                if msg.get("subtype") in (
                    "channel_join",
                    "channel_leave",
                    "bot_message",
                    "channel_topic",
                    "channel_purpose",
                    "channel_name",
                    "pinned_item",
                ):
                    continue

                if not msg.get("text", "").strip() and not msg.get("files") and not msg.get("attachments"):
                    continue

                ts = msg["ts"]
                if float(ts) > float(latest_ts):
                    latest_ts = ts

                thread_ts = msg.get("thread_ts", ts)
                participants: set[str] = set()

                # Build the full thread content
                if msg.get("reply_count", 0) > 0:
                    thread_parts, participants = await self._fetch_thread(channel_id, thread_ts)
                else:
                    thread_parts = [await self._format_message(msg)]
                    if msg.get("user"):
                        participants.add(msg["user"])

                content = "\n\n".join(thread_parts)

                # Extract file/attachment context
                attachments_text = self._extract_attachments(msg)
                if attachments_text:
                    content += "\n\n---\nAttachments:\n" + attachments_text

                doc = RawDocument(
                    source_type="slack",
                    source_id=f"slack:{channel_id}:{thread_ts}",
                    title=f"#{channel_name} - {self._truncate(msg.get('text', 'file/attachment'), 100)}",
                    content=content,
                    url=f"https://slack.com/archives/{channel_id}/p{thread_ts.replace('.', '')}",
                    metadata={
                        "channel_id": channel_id,
                        "channel_name": channel_name,
                        "thread_ts": thread_ts,
                        "participants": list(participants),
                        "reply_count": msg.get("reply_count", 0),
                        "reactions": self._extract_reactions(msg),
                        "has_files": bool(msg.get("files")),
                        "has_attachments": bool(msg.get("attachments")),
                    },
                    timestamp=datetime.fromtimestamp(float(ts)),
                )
                docs.append(doc)

            response_metadata = response.get("response_metadata", {})
            cursor = response_metadata.get("next_cursor")
            if not cursor:
                break

        if latest_ts != "0":
            await self.update_sync_cursor(channel_id, latest_ts)

        return docs

    async def _fetch_thread(
        self, channel_id: str, thread_ts: str
    ) -> tuple[list[str], set[str]]:
        """Fetch all replies in a thread, including nested replies and attachments."""
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
                if msg.get("subtype") in ("channel_join", "channel_leave"):
                    continue

                user = msg.get("user", msg.get("bot_id", "unknown"))
                participants.add(user)

                formatted = await self._format_message(msg)
                if formatted:
                    parts.append(formatted)

            response_metadata = response.get("response_metadata", {})
            cursor = response_metadata.get("next_cursor")
            if not cursor:
                break

        return parts, participants

    async def _format_message(self, msg: dict) -> str:
        """Format a single Slack message with all its content types."""
        user = msg.get("user", msg.get("bot_id", "unknown"))
        display_name = await self._resolve_user(user)
        parts = []

        # Main text
        text = msg.get("text", "").strip()
        if text:
            text = self._clean_slack_markup(text)
            parts.append(text)

        # Rich text blocks (Slack's block kit content)
        for block in msg.get("blocks", []):
            if block.get("type") == "rich_text":
                rich = self._extract_rich_text(block)
                if rich and rich not in parts:
                    parts.append(rich)

        # File shares
        for file_info in msg.get("files", []):
            file_text = self._format_file(file_info)
            if file_text:
                parts.append(file_text)

        # Attachments (legacy and unfurled links)
        att_text = self._extract_attachments(msg)
        if att_text:
            parts.append(att_text)

        content = "\n".join(parts)
        return f"[{display_name}]: {content}" if content else ""

    def _extract_rich_text(self, block: dict) -> str:
        """Recursively extract text from Slack rich_text blocks."""
        parts = []
        for element in block.get("elements", []):
            el_type = element.get("type", "")

            if el_type == "rich_text_section":
                for item in element.get("elements", []):
                    if item.get("type") == "text":
                        parts.append(item.get("text", ""))
                    elif item.get("type") == "link":
                        url = item.get("url", "")
                        text = item.get("text", url)
                        parts.append(f"[{text}]({url})")
                    elif item.get("type") == "user":
                        parts.append(f"@{item.get('user_id', 'user')}")
                    elif item.get("type") == "emoji":
                        parts.append(f":{item.get('name', '')}:")

            elif el_type == "rich_text_preformatted":
                code_parts = [
                    item.get("text", "")
                    for item in element.get("elements", [])
                    if item.get("type") == "text"
                ]
                if code_parts:
                    parts.append(f"```\n{''.join(code_parts)}\n```")

            elif el_type == "rich_text_list":
                for item in element.get("elements", []):
                    sub = self._extract_rich_text(item)
                    if sub:
                        parts.append(f"  - {sub}")

            elif el_type == "rich_text_quote":
                for item in element.get("elements", []):
                    if item.get("type") == "text":
                        parts.append(f"> {item.get('text', '')}")

        return "\n".join(parts)

    @staticmethod
    def _format_file(file_info: dict) -> str:
        """Extract useful text from file shares."""
        name = file_info.get("name", "unnamed")
        filetype = file_info.get("filetype", "")
        title = file_info.get("title", name)
        preview = file_info.get("preview", "") or file_info.get("plain_text", "")

        parts = [f"[File: {title} ({filetype})]"]

        # Include file preview/content if available (Slack provides this for text files)
        if preview:
            # Cap preview at 2000 chars to avoid bloating
            if len(preview) > 2000:
                preview = preview[:2000] + "...(truncated)"
            parts.append(preview)

        # Include initial comment if present
        initial_comment = file_info.get("initial_comment", {})
        if initial_comment and initial_comment.get("comment"):
            parts.append(f"Comment: {initial_comment['comment']}")

        return "\n".join(parts)

    @staticmethod
    def _extract_attachments(msg: dict) -> str:
        """Extract text from message attachments (unfurled links, bot messages, etc.)."""
        parts = []
        for att in msg.get("attachments", []):
            att_parts = []
            if att.get("title"):
                att_parts.append(f"**{att['title']}**")
            if att.get("text"):
                att_parts.append(att["text"])
            elif att.get("fallback"):
                att_parts.append(att["fallback"])
            if att.get("pretext"):
                att_parts.insert(0, att["pretext"])

            # Fields (key-value pairs in attachments)
            for field in att.get("fields", []):
                title = field.get("title", "")
                value = field.get("value", "")
                if title and value:
                    att_parts.append(f"{title}: {value}")

            if att_parts:
                parts.append("\n".join(att_parts))

        return "\n---\n".join(parts)

    def _clean_slack_markup(self, text: str) -> str:
        """Convert Slack mrkdwn to readable text."""
        # Replace user mentions with display names (best effort)
        text = _USER_MENTION.sub(r"@\1", text)
        # Replace channel mentions
        text = _CHANNEL_MENTION.sub(r"#\2", text)
        # Replace URLs
        text = _URL_PATTERN.sub(lambda m: m.group(2) or m.group(1), text)
        return text

    async def _resolve_user(self, user_id: str) -> str:
        """Resolve a Slack user ID to display name with caching."""
        if user_id in self._user_cache:
            return self._user_cache[user_id]

        try:
            response = await self.client.users_info(user=user_id)
            profile = response["user"].get("profile", {})
            name = (
                profile.get("display_name")
                or profile.get("real_name")
                or response["user"].get("name", user_id)
            )
            self._user_cache[user_id] = name
            return name
        except Exception:
            self._user_cache[user_id] = user_id
            return user_id

    @staticmethod
    def _extract_reactions(msg: dict) -> list[dict]:
        reactions = msg.get("reactions", [])
        return [{"name": r["name"], "count": r["count"]} for r in reactions]

    @staticmethod
    def _truncate(text: str, max_len: int) -> str:
        text = text.replace("\n", " ").strip()
        return text[:max_len] + "..." if len(text) > max_len else text
