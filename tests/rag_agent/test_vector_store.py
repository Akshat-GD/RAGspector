"""Tests for the FAISS-backed in-memory vector store."""

from collections.abc import Sequence

import pytest

from rag_agent.corpus.models import CorpusChunk
from rag_agent.retrieval import (
    DuplicateChunkIDError,
    FaissVectorStore,
    VectorStoreInputError,
)


def _chunk(chunk_id: str, title: str) -> CorpusChunk:
    return CorpusChunk(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        example_id=f"example-{chunk_id}",
        source_title=title,
        text=f"Text for {title}",
        source_sentence_indices=(1, 2),
        contains_supporting_fact=True,
        metadata={"category": "fixture", "title": title},
    )


@pytest.fixture
def store() -> FaissVectorStore:
    return FaissVectorStore(dimension=3)


def test_creates_index_and_exposes_dimension(store: FaissVectorStore) -> None:
    assert store.dimension == 3
    assert store.size == 0


def test_adds_embeddings_and_preserves_chunk_metadata(store: FaissVectorStore) -> None:
    chunk = _chunk("chunk-1", "First")

    store.add([chunk], [(1.0, 0.0, 0.0)])

    assert store.size == 1
    result = store.search((1.0, 0.0, 0.0))[0]
    assert result.chunk is chunk
    assert result.chunk.chunk_id == "chunk-1"
    assert result.chunk.document_id == "doc-chunk-1"
    assert result.chunk.example_id == "example-chunk-1"
    assert result.chunk.source_title == "First"
    assert result.chunk.text == "Text for First"
    assert result.chunk.source_sentence_indices == (1, 2)
    assert result.chunk.contains_supporting_fact is True
    assert dict(result.chunk.metadata) == {"category": "fixture", "title": "First"}


def test_similarity_search_orders_results_and_returns_scores(
    store: FaissVectorStore,
) -> None:
    chunks = [_chunk("aligned", "Aligned"), _chunk("diagonal", "Diagonal"), _chunk("opposite", "Opposite")]
    store.add(
        chunks,
        [
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (-1.0, 0.0, 0.0),
        ],
    )

    results = store.search((5.0, 0.0, 0.0), top_k=3)

    assert [result.chunk.chunk_id for result in results] == [
        "aligned",
        "diagonal",
        "opposite",
    ]
    assert [result.score for result in results] == pytest.approx(
        [1.0, 1.0 / 2**0.5, -1.0],
        abs=1e-6,
    )


def test_top_k_limits_results_and_caps_to_available_chunks(
    store: FaissVectorStore,
) -> None:
    store.add(
        [_chunk("a", "A"), _chunk("b", "B")],
        [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
    )

    assert len(store.search((1.0, 0.0, 0.0), top_k=1)) == 1
    assert len(store.search((1.0, 0.0, 0.0), top_k=10)) == 2


def test_empty_index_and_empty_add_batch_are_supported(
    store: FaissVectorStore,
) -> None:
    store.add([], [])

    assert store.size == 0
    assert store.search((1.0, 0.0, 0.0)) == []


@pytest.mark.parametrize(
    ("chunks", "embeddings"),
    [
        ([_chunk("short", "Short")], [(1.0, 0.0)]),
        ([_chunk("long", "Long")], [(1.0, 0.0, 0.0, 0.0)]),
    ],
)
def test_rejects_invalid_embedding_dimensions(
    store: FaissVectorStore,
    chunks: list[CorpusChunk],
    embeddings: list[Sequence[float]],
) -> None:
    with pytest.raises(VectorStoreInputError, match="expected 3"):
        store.add(chunks, embeddings)
    assert store.size == 0


def test_rejects_invalid_query_vector(store: FaissVectorStore) -> None:
    with pytest.raises(VectorStoreInputError, match="expected 3"):
        store.search((1.0, 0.0))
    with pytest.raises(VectorStoreInputError, match="zero magnitude"):
        store.search((0.0, 0.0, 0.0))


@pytest.mark.parametrize("top_k", [0, -1])
def test_rejects_non_positive_top_k(store: FaissVectorStore, top_k: int) -> None:
    with pytest.raises(ValueError, match="top_k"):
        store.search((1.0, 0.0, 0.0), top_k=top_k)


def test_duplicate_ids_are_rejected_without_partial_add(
    store: FaissVectorStore,
) -> None:
    original = _chunk("duplicate", "Original")
    store.add([original], [(1.0, 0.0, 0.0)])

    with pytest.raises(DuplicateChunkIDError, match="duplicate"):
        store.add(
            [_chunk("new", "New"), _chunk("duplicate", "Duplicate")],
            [(0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
        )

    assert store.size == 1
    assert [result.chunk.chunk_id for result in store.search((1.0, 0.0, 0.0))] == [
        "duplicate"
    ]


def test_duplicate_ids_within_batch_are_rejected(store: FaissVectorStore) -> None:
    with pytest.raises(DuplicateChunkIDError):
        store.add(
            [_chunk("same", "First"), _chunk("same", "Second")],
            [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        )


def test_mismatched_chunk_and_embedding_counts_are_rejected(
    store: FaissVectorStore,
) -> None:
    with pytest.raises(VectorStoreInputError, match="same number"):
        store.add([_chunk("one", "One")], [])


def test_zero_dimension_is_rejected() -> None:
    with pytest.raises(ValueError, match="dimension"):
        FaissVectorStore(dimension=0)

