"""Application configuration using pydantic-settings."""

from functools import lru_cache

try:
    # preferred: pydantic v2 separate package
    from pydantic_settings import BaseSettings, SettingsConfigDict
except Exception:
    try:
        # fallback: some environments expose BaseSettings from pydantic package
        from pydantic import BaseSettings  # type: ignore

        # minimal fallback so `model_config = SettingsConfigDict(...)` works
        SettingsConfigDict = dict
    except Exception as e:
        raise ImportError(
            "pydantic-settings is not installed. Install with 'uv add pydantic-settings' or 'pip install pydantic-settings' and ensure VS Code uses the project's Python interpreter."
        ) from e


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Google Configuration
    google_api_key: str

    # Qdrant Cloud Configuration
    qdrant_url: str
    qdrant_api_key: str

    # Collection Settings
    collection_name: str = "rag_documents"

    # Document Processing Settings
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Model Configuration
    embedding_model: str = "gemini-embedding-001"
    llm_model: str = "gemini-2.5-flash"
    llm_temperature: float = 0.0

    # Retrieval Settings
    retrieval_k: int = 4

    # Logging
    log_level: str = "INFO"

    # RAGAS Evaluation Settings
    enable_ragas_evaluation: bool = True
    ragas_timeout_seconds: float = 30.0
    ragas_log_results: bool = True
    ragas_llm_model: str | None = None  # Defaults to llm_model if not set
    ragas_llm_temperature: float | None = None  # Defaults to llm_temperature if not set
    ragas_embedding_model: str | None = None  # Defaults to embedding_model if not set

    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Application Info
    app_name: str = "RAG Q&A System"
    app_version: str = "0.1.0"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
