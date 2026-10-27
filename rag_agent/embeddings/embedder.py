"""Embedding interfaces and a local Sentence Transformers implementation."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
import math
from numbers import Real
from typing import TypeAlias


EmbeddingVector: TypeAlias = tuple[float, ...]

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingError(RuntimeError):
    """Raised when an embedding model cannot produce valid vectors."""


class EmbeddingDependencyError(ImportError):
    """Raised when the optional Sentence Transformers dependency is missing."""


class EmbeddingModelLoadError(EmbeddingError):
    """Raised when the configured sentence embedding model cannot be loaded."""


class EmbeddingInputError(ValueError):
    """Raised when embedding input is not a string or sequence of strings."""


class Embedder(ABC):
    """Abstraction for converting text into fixed-dimension vectors."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the number of values in each embedding vector."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier or implementation name."""

    @abstractmethod
    def embed_texts(self, texts: Sequence[str]) -> list[EmbeddingVector]:
        """Embed multiple texts, returning one vector per input."""

    def embed_text(self, text: str) -> EmbeddingVector:
        """Embed one text as a fixed-dimension vector."""
        _validate_text(text, 0)
        return self.embed_texts([text])[0]


@dataclass(frozen=True)
class SentenceTransformerConfig:
    """Configuration for the local Sentence Transformers model."""

    model_name: str = DEFAULT_MODEL_NAME
    device: str | None = None
    batch_size: int = 32

    def __post_init__(self) -> None:
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        if self.device is not None and (
            not isinstance(self.device, str) or not self.device.strip()
        ):
            raise ValueError("device must be None or a non-empty string")
        if (
            isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, int)
            or self.batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer")


class SentenceTransformerEmbedder(Embedder):
    """Generate L2-normalized embeddings with a local Sentence Transformer.

    Vectors are unit length so their dot product is equivalent to cosine
    similarity, which is a suitable metric for the later vector-store layer.
    No API service is used; the selected model is loaded locally by the
    ``sentence-transformers`` package.
    """

    def __init__(self, config: SentenceTransformerConfig | None = None) -> None:
        self.config = config if config is not None else SentenceTransformerConfig()
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise EmbeddingDependencyError(
                "SentenceTransformerEmbedder requires the optional "
                "'sentence-transformers' package. Install it with: "
                "pip install sentence-transformers"
            ) from error

        try:
            model_arguments: dict[str, str] = {}
            if self.config.device is not None:
                model_arguments["device"] = self.config.device
            self._model = SentenceTransformer(
                self.config.model_name,
                **model_arguments,
            )
            dimension = self._model.get_sentence_embedding_dimension()
        except Exception as error:
            raise EmbeddingModelLoadError(
                f"Could not load embedding model {self.config.model_name!r}"
            ) from error

        if (
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension <= 0
        ):
            raise EmbeddingModelLoadError(
                f"Model {self.config.model_name!r} reported invalid "
                f"embedding dimension: {dimension!r}"
            )
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        """Return the loaded model's embedding dimension."""
        return self._dimension

    @property
    def model_name(self) -> str:
        """Return the configured model identifier."""
        return self.config.model_name

    def embed_texts(self, texts: Sequence[str]) -> list[EmbeddingVector]:
        """Embed text inputs, returning an empty list for an empty batch."""
        if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
            raise EmbeddingInputError("texts must be a sequence of strings")
        for index, text in enumerate(texts):
            _validate_text(text, index)
        if not texts:
            return []

        try:
            encoded = self._model.encode(
                list(texts),
                batch_size=self.config.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            rows = encoded if isinstance(encoded, list) else encoded.tolist()
        except Exception as error:
            raise EmbeddingError(
                f"Embedding failed for {len(texts)} text input(s) "
                f"using {self.model_name!r}"
            ) from error

        if not isinstance(rows, list) or len(rows) != len(texts):
            raise EmbeddingError(
                "Embedding model returned a different number of vectors "
                "than the number of input texts"
            )
        return [self._normalize_vector(row, index) for index, row in enumerate(rows)]

    def _normalize_vector(self, values: object, index: int) -> EmbeddingVector:
        if not isinstance(values, list) or len(values) != self.dimension:
            raise EmbeddingError(
                f"Embedding {index} must contain exactly {self.dimension} values"
            )
        if any(
            isinstance(value, bool) or not isinstance(value, Real)
            for value in values
        ):
            raise EmbeddingError(f"Embedding {index} contains a non-numeric value")
        vector = tuple(float(value) for value in values)
        if any(not math.isfinite(value) for value in vector):
            raise EmbeddingError(f"Embedding {index} contains a non-finite value")
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            raise EmbeddingError(f"Embedding {index} has zero magnitude")
        return tuple(value / norm for value in vector)


def _validate_text(text: object, index: int) -> None:
    if not isinstance(text, str):
        raise EmbeddingInputError(f"text at index {index} must be a string")