"""Embedding-backed retrieval over the vector-store abstraction."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from numbers import Real

from rag_agent.corpus.models import CorpusChunk
from rag_agent.embeddings import Embedder
from rag_agent.retrieval.vector_store import VectorStore


@dataclass(frozen=True)
class RetrievalResult:
    """A retrieved chunk with its similarity score and one-based rank."""

    chunk: CorpusChunk
    score: float
    rank: int


@dataclass(frozen=True)
class RetrieverConfig:
    """Default result count and optional minimum cosine similarity."""

    default_top_k: int = 5
    score_threshold: float | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.default_top_k, bool)
            or not isinstance(self.default_top_k, int)
            or self.default_top_k <= 0
        ):
            raise ValueError("default_top_k must be a positive integer")
        if self.score_threshold is not None:
            _validate_score_threshold(self.score_threshold)


class Retriever(ABC):
    """Abstraction for retrieving ranked corpus chunks from a text query."""

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Return ranked results for a non-empty query."""


class EmbeddingVectorStoreRetriever(Retriever):
    """Connect a text embedder to a generic vector store.

    The vector store performs similarity search; this class handles query
    validation, defaults, threshold filtering, and rank assignment only.
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        config: RetrieverConfig | None = None,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.config = config if config is not None else RetrieverConfig()

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Embed ``query``, search the vector store, and rank qualifying hits."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        result_limit = self.config.default_top_k if top_k is None else top_k
        if (
            isinstance(result_limit, bool)
            or not isinstance(result_limit, int)
            or result_limit <= 0
        ):
            raise ValueError("top_k must be a positive integer")

        if self.vector_store.size == 0:
            return []

        query_embedding = self.embedder.embed_text(query)
        matches = self.vector_store.search(query_embedding, top_k=result_limit)
        ordered_matches = sorted(matches, key=lambda match: -match.score)
        threshold = self.config.score_threshold
        results: list[RetrievalResult] = []
        for match in ordered_matches:
            if threshold is not None and match.score < threshold:
                continue
            results.append(
                RetrievalResult(
                    chunk=match.chunk,
                    score=match.score,
                    rank=len(results) + 1,
                )
            )
        return results


def _validate_score_threshold(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("score_threshold must be a finite cosine score in [-1, 1]")
    try:
        numeric_value = float(value)
    except OverflowError as error:
        raise ValueError(
            "score_threshold must be a finite cosine score in [-1, 1]"
        ) from error
    if not math.isfinite(numeric_value) or not -1.0 <= numeric_value <= 1.0:
        raise ValueError("score_threshold must be a finite cosine score in [-1, 1]")