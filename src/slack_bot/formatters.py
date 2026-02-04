from __future__ import annotations

from src.api.schemas import SearchResponse


def format_search_response(response: SearchResponse, query: str) -> list[dict]:
    """Format search response as Slack Block Kit blocks."""
    blocks = []

    # Confidence indicator
    confidence_emoji = {
        "high": ":large_green_circle:",
        "medium": ":large_yellow_circle:",
        "low": ":red_circle:",
    }.get(response.confidence, ":white_circle:")

    # Answer section
    blocks.append(
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": response.answer,
            },
        }
    )

    # Confidence bar
    blocks.append(
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"{confidence_emoji} Confidence: *{response.confidence}*  |  "
                    f":zap: {response.latency_ms:.0f}ms  |  "
                    f"{'cached :floppy_disk:' if response.cached else 'live search'}",
                }
            ],
        }
    )

    blocks.append({"type": "divider"})

    # Sources
    if response.sources:
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*Sources:*",
                },
            }
        )

        for i, source in enumerate(response.sources[:5], 1):
            source_icon = {
                "slack": ":speech_balloon:",
                "confluence": ":page_facing_up:",
                "git": ":file_folder:",
            }.get(source.source_type, ":link:")

            link_text = f"<{source.url}|{source.title}>" if source.url else source.title
            snippet = source.snippet[:200] + "..." if len(source.snippet) > 200 else source.snippet

            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"{source_icon} *[{i}]* {link_text}\n>{snippet}",
                    },
                }
            )

    # Feedback buttons
    if response.sources:
        first_chunk_id = response.sources[0].chunk_id
        blocks.append(
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": ":thumbsup: Helpful"},
                        "action_id": "feedback_up",
                        "value": f"{response.query_id}:{first_chunk_id}",
                        "style": "primary",
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": ":thumbsdown: Not helpful"},
                        "action_id": "feedback_down",
                        "value": f"{response.query_id}:{first_chunk_id}",
                        "style": "danger",
                    },
                ],
            }
        )

    return blocks


def format_error_response() -> list[dict]:
    """Format an error message for Slack."""
    return [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": ":warning: Sorry, I encountered an error while searching. "
                "Please try again or contact the support team directly.",
            },
        }
    ]
