"""In-memory vector-store abstraction and exact FAISS implementation."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
import math
from numbers import Real

from rag_agent.corpus.models import CorpusChunk


class VectorStoreError(RuntimeError):
    """Raised when a vector-store operation fails."""


class VectorStoreDependencyError(ImportError):
    """Raised when the optional FAISS CPU dependency is missing."""


class VectorStoreInputError(ValueError):
    """Raised when chunks or vectors do not meet the vector-store contract."""


class DuplicateChunkIDError(VectorStoreInputError):
    """Raised when a chunk ID is already indexed or duplicated in a batch."""


@dataclass(frozen=True)
class VectorSearchResult:
    """A matched corpus chunk and its cosine-similarity score."""

    chunk: CorpusChunk
    score: float


class VectorStore(ABC):
    """Interface for storing corpus chunks and searching their embeddings."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the required embedding dimension."""

    @property
    @abstractmethod
    def size(self) -> int:
        """Return the number of indexed chunks."""

    @abstractmethod
    def add(
        self,
        chunks: Sequence[CorpusChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        """Add chunks and their corresponding embeddings."""

    @abstractmethod
    def search(
        self,
        query_embedding: Sequence[float],
        top_k: int = 5,
    ) -> list[VectorSearchResult]:
        """Return the nearest chunks ordered by descending similarity."""


class FaissVectorStore(VectorStore):
    """Exact in-memory FAISS index over L2-normalized embeddings.

    ``IndexFlatIP`` uses inner product. Since both stored and query vectors are
    normalized on input, inner product equals cosine similarity. Exact flat
    search is deterministic for the same ordered inputs and avoids training or
    approximate-index randomness. Chunk objects are retained alongside FAISS
    row positions, preserving their metadata unchanged.
    """

    def __init__(self, dimension: int) -> None:
        if isinstance(dimension, bool) or not isinstance(dimension, int):
            raise TypeError("dimension must be an integer")
        if dimension <= 0:
            raise ValueError("dimension must be greater than zero")
        try:
            import faiss
            import numpy as np
        except ImportError as error:
            raise VectorStoreDependencyError(
                "FaissVectorStore requires the 'faiss-cpu' dependency. "
                "Install it with: pip install -r requirements-rag-agent.txt"
            ) from error

        self._np = np
        try:
            self._index = faiss.IndexFlatIP(dimension)
        except Exception as error:
            raise VectorStoreError(
                f"Could not create FAISS inner-product index for dimension {dimension}"
            ) from error
        self._dimension = dimension
        self._chunks: list[CorpusChunk] = []
        self._chunk_ids: set[str] = set()

    @property
    def dimension(self) -> int:
        """Return the embedding dimension configured for this store."""
        return self._dimension

    @property
    def size(self) -> int:
        """Return the number of indexed chunks."""
        return len(self._chunks)

    def add(
        self,
        chunks: Sequence[CorpusChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        """Add a batch after validating all records and vectors."""
        if not _is_sequence(chunks):
            raise VectorStoreInputError("chunks must be a sequence")
        if not _is_sequence(embeddings):
            raise VectorStoreInputError("embeddings must be a sequence")
        if len(chunks) != len(embeddings):
            raise VectorStoreInputError(
                "chunks and embeddings must contain the same number of items"
            )
        if not chunks:
            return

        batch_ids: set[str] = set()
        for index, chunk in enumerate(chunks):
            if not isinstance(chunk, CorpusChunk):
                raise VectorStoreInputError(
                    f"chunk at index {index} must be a CorpusChunk"
                )
            if not isinstance(chunk.chunk_id, str) or not chunk.chunk_id:
                raise VectorStoreInputError(
                    f"chunk at index {index} must have a non-empty chunk_id"
                )
            if chunk.chunk_id in self._chunk_ids or chunk.chunk_id in batch_ids:
                raise DuplicateChunkIDError(
                    f"Duplicate chunk ID: {chunk.chunk_id!r}"
                )
            batch_ids.add(chunk.chunk_id)

        normalized_vectors = [
            self._validated_vector(embedding, f"embedding at index {index}")
            for index, embedding in enumerate(embeddings)
        ]
        matrix = self._np.ascontiguousarray(
            normalized_vectors,
            dtype=self._np.float32,
        )
        try:
            self._index.add(matrix)
        except Exception as error:
            raise VectorStoreError("FAISS failed to add embeddings") from error

        self._chunks.extend(chunks)
        self._chunk_ids.update(batch_ids)

    def search(
        self,
        query_embedding: Sequence[float],
        top_k: int = 5,
    ) -> list[VectorSearchResult]:
        """Search by cosine similarity, returning no more than ``top_k`` items."""
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be an integer")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        query = self._validated_vector(query_embedding, "query_embedding")
        if not self._chunks:
            return []

        query_matrix = self._np.ascontiguousarray(
            [query],
            dtype=self._np.float32,
        )
        result_count = min(top_k, self.size)
        try:
            scores, indices = self._index.search(query_matrix, result_count)
        except Exception as error:
            raise VectorStoreError("FAISS failed to search embeddings") from error

        hits = [
            (int(index), float(score))
            for index, score in zip(indices[0], scores[0])
            if int(index) >= 0
        ]
        hits.sort(key=lambda hit: (-hit[1], hit[0]))
        return [
            VectorSearchResult(chunk=self._chunks[index], score=score)
            for index, score in hits
        ]

    def _validated_vector(
        self,
        vector: Sequence[float],
        field_name: str,
    ) -> tuple[float, ...]:
        if not _is_sequence(vector):
            raise VectorStoreInputError(f"{field_name} must be a numeric sequence")
        if len(vector) != self.dimension:
            raise VectorStoreInputError(
                f"{field_name} has dimension {len(vector)}; "
                f"expected {self.dimension}"
            )
        if any(
            isinstance(value, bool) or not isinstance(value, Real)
            for value in vector
        ):
            raise VectorStoreInputError(f"{field_name} contains a non-numeric value")
        values = tuple(float(value) for value in vector)
        if any(not math.isfinite(value) for value in values):
            raise VectorStoreInputError(f"{field_name} contains a non-finite value")
        norm = math.sqrt(sum(value * value for value in values))
        if norm == 0:
            raise VectorStoreInputError(f"{field_name} has zero magnitude")
        return tuple(value / norm for value in values)


def _is_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))