"""Vector-store abstractions and implementations."""

from rag_agent.retrieval.retriever import (
    EmbeddingVectorStoreRetriever,
    RetrievalResult,
    Retriever,
    RetrieverConfig,
)
from rag_agent.retrieval.vector_store import (
    DuplicateChunkIDError,
    FaissVectorStore,
    VectorSearchResult,
    VectorStore,
    VectorStoreDependencyError,
    VectorStoreError,
    VectorStoreInputError,
)

__all__ = [
    "DuplicateChunkIDError",
    "EmbeddingVectorStoreRetriever",
    "FaissVectorStore",
    "RetrievalResult",
    "Retriever",
    "RetrieverConfig",
    "VectorSearchResult",
    "VectorStore",
    "VectorStoreDependencyError",
    "VectorStoreError",
    "VectorStoreInputError",
]