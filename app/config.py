"""
app/config.py — Application settings loaded from .env via pydantic-settings.
All configurable values live here. Nothing is hardcoded anywhere else.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central config — reads from .env file automatically."""

    # Google / LLM
    google_api_key: str
    model_name: str = "gemini-2.0-flash-lite"
    embedding_model: str = "models/text-embedding-004"

    # FastAPI
    fastapi_port: int = 8088
    log_level: str = "INFO"

    # ChromaDB
    chroma_persist_dir: str = "./data/chroma"
    max_retrieved_examples: int = 3

    # PostgreSQL
    postgres_host: str = "db"
    postgres_port: int = 5432
    postgres_db: str = "support_agent"
    postgres_user: str = "agent"
    postgres_password: str = "agentpass"

    # Evaluation
    golden_set_path: str = "./data/golden_set/golden_set.jsonl"
    golden_set_size: int = 200

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @property
    def postgres_url(self) -> str:
        """Async-compatible PostgreSQL DSN."""
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def postgres_url_sync(self) -> str:
        """Sync PostgreSQL DSN (used for Alembic migrations)."""
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


# Singleton — import this everywhere
settings = Settings()
