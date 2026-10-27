"""Text embedding abstractions and implementations."""

from rag_agent.embeddings.embedder import (
    DEFAULT_MODEL_NAME,
    Embedder,
    EmbeddingDependencyError,
    EmbeddingError,
    EmbeddingInputError,
    EmbeddingModelLoadError,
    EmbeddingVector,
    SentenceTransformerConfig,
    SentenceTransformerEmbedder,
)

__all__ = [
    "DEFAULT_MODEL_NAME",
    "Embedder",
    "EmbeddingDependencyError",
    "EmbeddingError",
    "EmbeddingInputError",
    "EmbeddingModelLoadError",
    "EmbeddingVector",
    "SentenceTransformerConfig",
    "SentenceTransformerEmbedder",
]