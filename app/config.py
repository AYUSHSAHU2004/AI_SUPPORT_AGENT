"""
app/config.py — Application settings loaded from .env via pydantic-settings.
All configurable values live here. Nothing is hardcoded anywhere else.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central config — reads from .env file automatically."""

    # Google / LLM
    google_api_key: str
    model_name: str = "gemini-3.1-flash-lite"
    embedding_model: str = "models/text-embedding-004"

    # FastAPI
    fastapi_port: int = 8088
    log_level: str = "INFO"

    # ChromaDB
    chroma_persist_dir: str = "./data/chroma"
    max_retrieved_examples: int = 3

    # SQLite (audit logs — no server required)
    sqlite_db_path: str = "./data/support_agent.db"

    # Evaluation
    golden_set_path: str = "./data/golden_set/golden_set.jsonl"
    golden_set_size: int = 200

    # Optional tools
    kaggle_token: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


# Singleton — import this everywhere
settings = Settings()
