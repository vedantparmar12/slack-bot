from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # OpenAI
    openai_api_key: str

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "support_docs"
    qdrant_cache_collection: str = "query_cache"

    # Redis
    redis_url: str = "redis://localhost:6379"

    # PostgreSQL
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/support_rag"

    # Slack
    slack_bot_token: str = ""
    slack_app_token: str = ""
    slack_signing_secret: str = ""
    slack_support_channels: str = ""  # Comma-separated channel IDs

    # Confluence
    confluence_url: str = ""
    confluence_username: str = ""
    confluence_api_token: str = ""
    confluence_spaces: str = ""  # Comma-separated space keys

    # Git Docs
    git_doc_repos: str = ""  # Comma-separated repo URLs
    git_doc_branch: str = "main"
    git_clone_dir: str = "./data/repos"

    # Embedding
    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 3072
    embedding_batch_size: int = 100

    # LLM
    llm_model: str = "gpt-4o"
    llm_expansion_model: str = "gpt-4o-mini"

    # Reranker
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    reranker_top_k: int = 5

    # Cache
    cache_exact_ttl_seconds: int = 3600
    cache_semantic_ttl_seconds: int = 14400
    cache_semantic_threshold: float = 0.95

    # Search
    search_top_k: int = 50
    search_rerank_top_k: int = 20
    search_final_top_k: int = 5

    # Feedback
    feedback_min_votes: int = 3
    feedback_decay_rate: float = 0.05

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

    @property
    def slack_channel_ids(self) -> list[str]:
        return [c.strip() for c in self.slack_support_channels.split(",") if c.strip()]

    @property
    def confluence_space_keys(self) -> list[str]:
        return [s.strip() for s in self.confluence_spaces.split(",") if s.strip()]

    @property
    def git_repo_urls(self) -> list[str]:
        return [r.strip() for r in self.git_doc_repos.split(",") if r.strip()]


settings = Settings()
