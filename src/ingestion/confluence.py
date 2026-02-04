from __future__ import annotations

import re
from datetime import datetime

import structlog
from atlassian import Confluence
from bs4 import BeautifulSoup

from src.config import settings
from src.ingestion.base import BaseIngester, RawDocument
from src.ingestion.chunker import SmartChunker
from src.ingestion.embedder import Embedder

logger = structlog.get_logger()


class ConfluenceIngester(BaseIngester):
    def __init__(self, chunker: SmartChunker, embedder: Embedder):
        super().__init__(chunker, embedder)
        self.client = Confluence(
            url=settings.confluence_url,
            username=settings.confluence_username,
            password=settings.confluence_api_token,
            cloud=True,
        )
        self.spaces = settings.confluence_space_keys

    @property
    def source_type(self) -> str:
        return "confluence"

    async def fetch(self, since: datetime | None = None) -> list[RawDocument]:
        docs: list[RawDocument] = []

        for space_key in self.spaces:
            space_docs = await self._fetch_space(space_key, since)
            docs.extend(space_docs)

        logger.info("Fetched Confluence pages", spaces=len(self.spaces), documents=len(docs))
        return docs

    async def _fetch_space(
        self, space_key: str, since: datetime | None
    ) -> list[RawDocument]:
        docs: list[RawDocument] = []

        # Build CQL query
        cql = f'space = "{space_key}" AND type = "page"'
        if since:
            cql += f' AND lastModified > "{since.strftime("%Y-%m-%d")}"'

        start = 0
        limit = 50

        while True:
            # Note: atlassian-python-api is sync; wrap in executor if needed
            results = self.client.cql(
                cql=cql,
                start=start,
                limit=limit,
                expand="body.storage,version,ancestors,metadata.labels",
            )

            pages = results.get("results", [])
            if not pages:
                break

            for page in pages:
                content_data = page.get("content", page)
                page_id = str(content_data.get("id", ""))
                title = content_data.get("title", "Untitled")

                # Extract body
                body_html = (
                    content_data.get("body", {}).get("storage", {}).get("value", "")
                )
                clean_text = self._html_to_text(body_html)

                if not clean_text.strip():
                    continue

                # Extract metadata
                version = content_data.get("version", {})
                ancestors = content_data.get("ancestors", [])
                labels_data = (
                    content_data.get("metadata", {}).get("labels", {}).get("results", [])
                )

                page_url = f"{settings.confluence_url}/wiki/spaces/{space_key}/pages/{page_id}"

                doc = RawDocument(
                    source_type="confluence",
                    source_id=f"confluence:{space_key}:{page_id}",
                    title=title,
                    content=clean_text,
                    url=page_url,
                    metadata={
                        "space_key": space_key,
                        "page_id": page_id,
                        "version": version.get("number", 1),
                        "last_modifier": version.get("by", {}).get("displayName", ""),
                        "labels": [l.get("name", "") for l in labels_data],
                        "parent_titles": [a.get("title", "") for a in ancestors],
                    },
                    timestamp=(
                        datetime.fromisoformat(
                            version["when"].replace("Z", "+00:00")
                        )
                        if version.get("when")
                        else None
                    ),
                )
                docs.append(doc)

            start += limit
            if start >= results.get("totalSize", 0):
                break

        return docs

    @staticmethod
    def _html_to_text(html: str) -> str:
        """Convert Confluence storage format HTML to clean markdown-like text."""
        if not html:
            return ""

        soup = BeautifulSoup(html, "html.parser")

        # Remove script and style elements
        for tag in soup(["script", "style"]):
            tag.decompose()

        # Convert headings
        for level in range(1, 7):
            for heading in soup.find_all(f"h{level}"):
                heading.replace_with(f"\n{'#' * level} {heading.get_text().strip()}\n")

        # Convert lists
        for li in soup.find_all("li"):
            li.replace_with(f"\n- {li.get_text().strip()}")

        # Convert code blocks
        for code in soup.find_all("code"):
            code.replace_with(f"`{code.get_text()}`")

        for pre in soup.find_all("pre"):
            pre.replace_with(f"\n```\n{pre.get_text()}\n```\n")

        # Convert links
        for a in soup.find_all("a"):
            href = a.get("href", "")
            text = a.get_text().strip()
            if href and text:
                a.replace_with(f"[{text}]({href})")

        # Convert tables to simple text
        for table in soup.find_all("table"):
            rows = []
            for tr in table.find_all("tr"):
                cells = [td.get_text().strip() for td in tr.find_all(["td", "th"])]
                rows.append(" | ".join(cells))
            table.replace_with("\n" + "\n".join(rows) + "\n")

        text = soup.get_text()

        # Clean up whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)

        return text.strip()
