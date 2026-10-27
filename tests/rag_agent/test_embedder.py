"""Tests for the embedding abstraction and local embedder wrapper."""

import math
import sys
import types
from collections.abc import Sequence
from unittest.mock import Mock, patch

import pytest

from rag_agent.embeddings import (
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


class FakeEmbedder(Embedder):
    """Small deterministic implementation used to exercise the abstraction."""

    dimension = 2
    model_name = "fake"

    def embed_texts(self, texts: Sequence[str]) -> list[EmbeddingVector]:
        return [(float(len(text)), 1.0) for text in texts]


def _fake_sentence_transformers(
    model: Mock | None = None,
) -> tuple[types.ModuleType, Mock]:
    module = types.ModuleType("sentence_transformers")
    instance = model if model is not None else Mock()
    instance.get_sentence_embedding_dimension.return_value = 2
    instance.encode.side_effect = lambda texts, **kwargs: [
        [3.0, 4.0] for _ in texts
    ]
    module.SentenceTransformer = Mock(return_value=instance)
    return module, instance


def test_abstraction_supports_single_and_batch_text() -> None:
    embedder = FakeEmbedder()

    assert embedder.embed_text("abcd") == (4.0, 1.0)
    assert embedder.embed_texts(["a", "bc"]) == [(1.0, 1.0), (2.0, 1.0)]
    assert embedder.dimension == 2
    assert embedder.model_name == "fake"


def test_default_config_selects_lightweight_local_model() -> None:
    assert SentenceTransformerConfig().model_name == DEFAULT_MODEL_NAME
    assert DEFAULT_MODEL_NAME == "sentence-transformers/all-MiniLM-L6-v2"


def test_sentence_transformer_configures_model_and_normalizes_vectors() -> None:
    module, model = _fake_sentence_transformers()
    with patch.dict(sys.modules, {"sentence_transformers": module}):
        embedder = SentenceTransformerEmbedder(
            SentenceTransformerConfig(model_name="test/model", device="cpu")
        )

        vectors = embedder.embed_texts(["first", "second"])

    assert module.SentenceTransformer.call_args.args == ("test/model",)
    assert module.SentenceTransformer.call_args.kwargs == {"device": "cpu"}
    assert vectors == [(0.6, 0.8), (0.6, 0.8)]
    assert embedder.dimension == 2
    assert embedder.model_name == "test/model"
    model.encode.assert_called_once_with(
        ["first", "second"],
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    assert math.sqrt(sum(value * value for value in vectors[0])) == pytest.approx(1.0)


def test_single_text_uses_same_normalized_batch_representation() -> None:
    module, _ = _fake_sentence_transformers()
    with patch.dict(sys.modules, {"sentence_transformers": module}):
        embedder = SentenceTransformerEmbedder()

        vector = embedder.embed_text("text")

    assert vector == (0.6, 0.8)


def test_empty_batch_returns_no_vectors_without_encoding() -> None:
    module, model = _fake_sentence_transformers()
    with patch.dict(sys.modules, {"sentence_transformers": module}):
        embedder = SentenceTransformerEmbedder()

        assert embedder.embed_texts([]) == []

    model.encode.assert_not_called()


@pytest.mark.parametrize("value", [None, 42, ["valid", 3]])
def test_invalid_text_inputs_raise_clear_error(value: object) -> None:
    module, _ = _fake_sentence_transformers()
    with patch.dict(sys.modules, {"sentence_transformers": module}):
        embedder = SentenceTransformerEmbedder()

        with pytest.raises(EmbeddingInputError):
            if isinstance(value, list):
                embedder.embed_texts(value)
            else:
                embedder.embed_text(value)


def test_missing_dependency_raises_actionable_error() -> None:
    with patch.dict(sys.modules, {"sentence_transformers": None}):
        with pytest.raises(EmbeddingDependencyError, match="pip install sentence-transformers"):
            SentenceTransformerEmbedder()


def test_model_loading_failure_is_reported() -> None:
    module, _ = _fake_sentence_transformers()
    module.SentenceTransformer.side_effect = OSError("model files unavailable")

    with patch.dict(sys.modules, {"sentence_transformers": module}):
        with pytest.raises(EmbeddingModelLoadError, match="test/model"):
            SentenceTransformerEmbedder(
                SentenceTransformerConfig(model_name="test/model")
            )


def test_model_encoding_failure_is_not_replaced_with_fake_vectors() -> None:
    module, model = _fake_sentence_transformers()
    model.encode.side_effect = OSError("encoding failed")
    with patch.dict(sys.modules, {"sentence_transformers": module}):
        embedder = SentenceTransformerEmbedder()

        with pytest.raises(EmbeddingError, match="Embedding failed"):
            embedder.embed_texts(["text"])


def test_invalid_config_is_rejected() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        SentenceTransformerConfig(batch_size=0)
