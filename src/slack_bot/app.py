from __future__ import annotations

import structlog
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler

from src.config import settings
from src.slack_bot.handlers import register_handlers

logger = structlog.get_logger()


def create_slack_app() -> AsyncApp:
    app = AsyncApp(
        token=settings.slack_bot_token,
        signing_secret=settings.slack_signing_secret,
    )
    register_handlers(app)
    return app


async def start_slack_bot():
    """Start the Slack bot in Socket Mode."""
    app = create_slack_app()
    handler = AsyncSocketModeHandler(app, settings.slack_app_token)
    logger.info("Starting Slack bot in Socket Mode")
    await handler.start_async()
