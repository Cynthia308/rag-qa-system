"""Embedding generation module using OpenAI embeddings."""

from functools import lru_cache

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


@lru_cache
def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Get cached Google embeddings instance.

    Returns:
        Configured GoogleEmbeddings instance
    """
    settings = get_settings()
    logger.info(f"Initializing embeddings model: {settings.embedding_model}")

    embeddings = GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.google_api_key,
    )

    logger.info("Embeddings model initialized successfully")
    return embeddings


class EmbeddingService:
    """Service for generating embeddings."""

    def __init__(self):
        """Initialize embedding service."""
        settings = get_settings()
        self.embeddings = get_embeddings()
        self.model_name = settings.embedding_model

    def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a single query.

        Args:
            text: Query text

        Returns:
            Embedding vector as list of floats
        """
        logger.debug(f"Generating embedding for query: {text[:50]}...")
        return self.embeddings.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple documents.

        Args:
            texts: List of document texts

        Returns:
            List of embedding vectors
        """
        logger.debug(f"Generating embeddings for {len(texts)} documents")
        return self.embeddings.embed_documents(texts)