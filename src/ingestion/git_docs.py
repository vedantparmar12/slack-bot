from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import structlog
from git import Repo

import frontmatter

from src.config import settings
from src.ingestion.base import BaseIngester, RawDocument
from src.ingestion.chunker import SmartChunker
from src.ingestion.embedder import Embedder

logger = structlog.get_logger()

DOC_EXTENSIONS = {".md", ".mdx", ".html", ".htm", ".rst", ".txt"}


class GitDocsIngester(BaseIngester):
    def __init__(self, chunker: SmartChunker, embedder: Embedder):
        super().__init__(chunker, embedder)
        self.repos = settings.git_repo_urls
        self.branch = settings.git_doc_branch
        self.clone_dir = Path(settings.git_clone_dir)

    @property
    def source_type(self) -> str:
        return "git"

    async def fetch(self, since: datetime | None = None) -> list[RawDocument]:
        docs: list[RawDocument] = []

        for repo_url in self.repos:
            repo_docs = await self._fetch_repo(repo_url, since)
            docs.extend(repo_docs)

        logger.info("Fetched git docs", repos=len(self.repos), documents=len(docs))
        return docs

    async def _fetch_repo(
        self, repo_url: str, since: datetime | None
    ) -> list[RawDocument]:
        docs: list[RawDocument] = []
        repo_name = self._repo_name(repo_url)
        repo_dir = self.clone_dir / repo_name

        # Clone or pull
        if repo_dir.exists():
            repo = Repo(str(repo_dir))
            origin = repo.remotes.origin
            origin.pull(self.branch)
            logger.info("Pulled repo", repo=repo_name)
        else:
            self.clone_dir.mkdir(parents=True, exist_ok=True)
            repo = Repo.clone_from(repo_url, str(repo_dir), branch=self.branch)
            logger.info("Cloned repo", repo=repo_name)

        # Get the last synced commit
        last_cursor = await self.get_sync_cursor(repo_url)

        # Find changed files since last sync
        if since and last_cursor:
            changed_files = self._get_changed_files(repo, last_cursor)
        else:
            changed_files = None  # Process all files

        # Walk the repo for doc files
        for root, _, files in os.walk(str(repo_dir)):
            # Skip .git directory
            if ".git" in root:
                continue

            for filename in files:
                filepath = Path(root) / filename
                if filepath.suffix.lower() not in DOC_EXTENSIONS:
                    continue

                rel_path = filepath.relative_to(repo_dir)

                # If incremental, skip unchanged files
                if changed_files is not None and str(rel_path) not in changed_files:
                    continue

                content = filepath.read_text(encoding="utf-8", errors="replace")
                if not content.strip():
                    continue

                # Parse frontmatter for .md/.mdx files
                meta = {}
                title = filename
                if filepath.suffix.lower() in (".md", ".mdx"):
                    try:
                        post = frontmatter.loads(content)
                        meta = dict(post.metadata)
                        title = meta.pop("title", filepath.stem)
                        content = post.content
                    except Exception:
                        title = filepath.stem

                # Build URL (GitHub convention)
                base_url = repo_url.rstrip(".git")
                file_url = f"{base_url}/blob/{self.branch}/{rel_path}"

                doc = RawDocument(
                    source_type="git",
                    source_id=f"git:{repo_name}:{rel_path}",
                    title=title,
                    content=content,
                    url=file_url,
                    metadata={
                        "repo": repo_name,
                        "repo_url": repo_url,
                        "file_path": str(rel_path),
                        "extension": filepath.suffix,
                        **meta,
                    },
                )
                docs.append(doc)

        # Update sync cursor to current HEAD
        current_sha = str(repo.head.commit.hexsha)
        await self.update_sync_cursor(repo_url, current_sha)

        return docs

    @staticmethod
    def _repo_name(repo_url: str) -> str:
        """Extract repository name from URL."""
        name = repo_url.rstrip("/").split("/")[-1]
        if name.endswith(".git"):
            name = name[:-4]
        return name

    @staticmethod
    def _get_changed_files(repo: Repo, since_commit: str) -> set[str]:
        """Get files changed since a specific commit."""
        try:
            diff = repo.head.commit.diff(since_commit)
            changed = set()
            for d in diff:
                if d.a_path:
                    changed.add(d.a_path)
                if d.b_path:
                    changed.add(d.b_path)
            return changed
        except Exception:
            return set()  # If commit not found, return empty (will process all)
