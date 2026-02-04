from __future__ import annotations

import structlog

from src.config import settings
from src.dependencies import get_openai_client

logger = structlog.get_logger()

EXPANSION_PROMPT = """You are a query expansion assistant. Given a user's support question, generate 2-3 alternative phrasings that preserve the original intent but use different words or perspectives.

Rules:
- Keep each variation concise (under 50 words)
- Preserve the core intent
- Use different terminology and phrasing
- Output only the variations, one per line, no numbering

User query: {query}"""


class QueryExpander:
    def __init__(self):
        self.client = get_openai_client()
        self.model = settings.llm_expansion_model

    async def expand(self, query: str) -> list[str]:
        """Generate query variations. Returns original + expansions."""
        # Skip expansion for very short queries
        word_count = len(query.split())
        if word_count < 3:
            return [query]

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": EXPANSION_PROMPT.format(query=query)},
                ],
                temperature=0.7,
                max_tokens=200,
            )

            variations_text = response.choices[0].message.content or ""
            variations = [
                v.strip().lstrip("- ").lstrip("0123456789.)")
                for v in variations_text.strip().split("\n")
                if v.strip()
            ]

            # Always include original query first
            result = [query] + variations[:3]
            logger.debug("Query expanded", original=query, variations=len(result) - 1)
            return result

        except Exception as e:
            logger.warning("Query expansion failed, using original", error=str(e))
            return [query]
