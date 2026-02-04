from __future__ import annotations

import uuid

import structlog
from slack_bolt.async_app import AsyncApp

from src.config import settings
from src.search.engine import SearchEngine
from src.slack_bot.formatters import format_search_response, format_error_response

logger = structlog.get_logger()

_engine: SearchEngine | None = None


def _get_engine() -> SearchEngine:
    global _engine
    if _engine is None:
        _engine = SearchEngine()
    return _engine


def register_handlers(app: AsyncApp):
    """Register all Slack event handlers."""

    @app.event("app_mention")
    async def handle_mention(event, say):
        """Handle @bot mentions in any channel."""
        text = event.get("text", "")
        # Remove the bot mention from the query
        # Format is typically "<@BOT_ID> query text"
        parts = text.split(">", 1)
        query = parts[1].strip() if len(parts) > 1 else text.strip()

        if not query:
            await say(
                text="Please ask me a question! For example: `@bot How do I reset my password?`",
                thread_ts=event.get("thread_ts", event["ts"]),
            )
            return

        await _handle_search(query, event, say)

    @app.event("message")
    async def handle_message(event, say):
        """Handle messages in support channels."""
        channel = event.get("channel", "")

        # Only respond in configured support channels
        if channel not in settings.slack_channel_ids:
            return

        # Skip bot messages, thread replies (unless mentioning bot), and edits
        if event.get("subtype") or event.get("bot_id"):
            return

        # Skip thread replies (we only respond to top-level messages)
        if event.get("thread_ts") and event["thread_ts"] != event["ts"]:
            return

        query = event.get("text", "").strip()
        if not query:
            return

        await _handle_search(query, event, say)

    @app.command("/search")
    async def handle_search_command(ack, command, say):
        """Handle /search slash command."""
        await ack()
        query = command.get("text", "").strip()

        if not query:
            await say("Please provide a search query. Usage: `/search your question here`")
            return

        await _handle_search(query, command, say, is_command=True)

    @app.action("feedback_up")
    async def handle_feedback_up(ack, body, say):
        """Handle thumbs up feedback button."""
        await ack()
        await _handle_feedback(body, vote=1)

    @app.action("feedback_down")
    async def handle_feedback_down(ack, body, say):
        """Handle thumbs down feedback button."""
        await ack()
        await _handle_feedback(body, vote=-1)


async def _handle_search(query: str, event_or_command: dict, say, is_command: bool = False):
    """Core search handler used by all entry points."""
    thread_ts = event_or_command.get("thread_ts", event_or_command.get("ts"))
    user_id = event_or_command.get("user", event_or_command.get("user_id", "unknown"))

    try:
        engine = _get_engine()
        response = await engine.search(query=query, source="slack")

        # Format and send response
        blocks = format_search_response(response, query)

        await say(
            text=response.answer,  # Fallback text
            blocks=blocks,
            thread_ts=thread_ts,
        )

        logger.info(
            "Slack search completed",
            user=user_id,
            query=query[:50],
            confidence=response.confidence,
        )

    except Exception as e:
        logger.error("Slack search failed", error=str(e), query=query[:50])
        blocks = format_error_response()
        await say(
            text="Sorry, I encountered an error while searching.",
            blocks=blocks,
            thread_ts=thread_ts,
        )


async def _handle_feedback(body: dict, vote: int):
    """Process feedback from Slack interactive buttons."""
    try:
        action = body.get("actions", [{}])[0]
        value = action.get("value", "")  # Format: "query_id:chunk_id"
        user_id = body.get("user", {}).get("id", "unknown")

        if ":" not in value:
            return

        query_id, chunk_id = value.split(":", 1)

        from src.db.models import Feedback
        from src.db.session import async_session
        from sqlalchemy import select

        async with async_session() as session:
            stmt = select(Feedback).where(
                Feedback.chunk_id == uuid.UUID(chunk_id),
                Feedback.user_id == user_id,
                Feedback.query_text == query_id,
            )
            result = await session.execute(stmt)
            existing = result.scalar_one_or_none()

            if existing:
                existing.vote = vote
            else:
                fb = Feedback(
                    query_text=query_id,
                    chunk_id=uuid.UUID(chunk_id),
                    user_id=user_id,
                    vote=vote,
                )
                session.add(fb)

            await session.commit()

        logger.info("Slack feedback recorded", user=user_id, vote=vote)

    except Exception as e:
        logger.warning("Failed to record Slack feedback", error=str(e))
