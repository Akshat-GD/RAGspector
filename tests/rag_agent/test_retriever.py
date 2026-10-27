"""Tests for the embedding-backed corpus retriever."""

from collections.abc import Sequence

import pytest

from rag_agent.corpus.models import CorpusChunk
from rag_agent.embeddings import Embedder, EmbeddingVector
from rag_agent.retrieval import (
    EmbeddingVectorStoreRetriever,
    RetrievalResult,
    RetrieverConfig,
    VectorSearchResult,
    VectorStore,
)


class FakeEmbedder(Embedder):
    dimension = 2
    model_name = "fake"

    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed_texts(self, texts: Sequence[str]) -> list[EmbeddingVector]:
        self.queries.extend(texts)
        return [(1.0, 0.0) for _ in texts]


class FakeVectorStore(VectorStore):
    def __init__(self, matches: list[VectorSearchResult]) -> None:
        self.matches = matches
        self.search_calls: list[tuple[tuple[float, ...], int]] = []

    @property
    def dimension(self) -> int:
        return 2

    @property
    def size(self) -> int:
        return len(self.matches)

    def add(
        self,
        chunks: Sequence[CorpusChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        raise NotImplementedError("Tests provide precomputed search matches")

    def search(
        self,
        query_embedding: Sequence[float],
        top_k: int = 5,
    ) -> list[VectorSearchResult]:
        self.search_calls.append((tuple(query_embedding), top_k))
        return sorted(self.matches, key=lambda match: -match.score)[:top_k]


def _chunk(chunk_id: str, *, category: str) -> CorpusChunk:
    return CorpusChunk(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        example_id=f"example-{chunk_id}",
        source_title=f"Source {chunk_id}",
        text=f"Text for {chunk_id}",
        source_sentence_indices=(0, 2),
        contains_supporting_fact=True,
        metadata={"category": category},
    )


def _matches() -> list[VectorSearchResult]:
    return [
        VectorSearchResult(_chunk("low", category="low"), 0.25),
        VectorSearchResult(_chunk("high", category="high"), 0.95),
        VectorSearchResult(_chunk("middle", category="middle"), 0.65),
    ]


def _retriever(
    *,
    config: RetrieverConfig | None = None,
    matches: list[VectorSearchResult] | None = None,
) -> tuple[EmbeddingVectorStoreRetriever, FakeEmbedder, FakeVectorStore]:
    embedder = FakeEmbedder()
    vector_store = FakeVectorStore(_matches() if matches is None else matches)
    return (
        EmbeddingVectorStoreRetriever(
            embedder,
            vector_store,
            config=config,
        ),
        embedder,
        vector_store,
    )


def test_retriever_embeds_query_and_passes_vector_to_store() -> None:
    retriever, embedder, vector_store = _retriever()

    results = retriever.retrieve(" find evidence ")

    assert embedder.queries == [" find evidence "]
    assert vector_store.search_calls == [((1.0, 0.0), 5)]
    assert [result.chunk.chunk_id for result in results] == [
        "high",
        "middle",
        "low",
    ]


def test_default_and_per_call_top_k_are_respected() -> None:
    retriever, _, vector_store = _retriever(
        config=RetrieverConfig(default_top_k=2)
    )

    default_results = retriever.retrieve("query")
    limited_results = retriever.retrieve("query", top_k=1)

    assert [result.chunk.chunk_id for result in default_results] == ["high", "middle"]
    assert [result.chunk.chunk_id for result in limited_results] == ["high"]
    assert [call[1] for call in vector_store.search_calls] == [2, 1]


def test_score_and_one_based_rank_are_propagated() -> None:
    retriever, _, _ = _retriever()

    results = retriever.retrieve("query")

    assert [(result.rank, result.score) for result in results] == [
        (1, 0.95),
        (2, 0.65),
        (3, 0.25),
    ]


def test_similarity_threshold_filters_results_and_reranks() -> None:
    retriever, _, _ = _retriever(
        config=RetrieverConfig(default_top_k=3, score_threshold=0.6)
    )

    results = retriever.retrieve("query")

    assert [(result.chunk.chunk_id, result.rank) for result in results] == [
        ("high", 1),
        ("middle", 2),
    ]


def test_results_preserve_original_chunk_and_metadata() -> None:
    matches = _matches()
    retriever, _, _ = _retriever(matches=matches)

    results = retriever.retrieve("query")
    high_result = results[0]
    original = next(match.chunk for match in matches if match.chunk.chunk_id == "high")

    assert high_result.chunk is original
    assert high_result.chunk.source_title == "Source high"
    assert high_result.chunk.source_sentence_indices == (0, 2)
    assert high_result.chunk.contains_supporting_fact is True
    assert dict(high_result.chunk.metadata) == {"category": "high"}


@pytest.mark.parametrize("query", ["", "  ", "\t\n"])
def test_empty_or_whitespace_query_is_rejected(query: str) -> None:
    retriever, _, _ = _retriever()

    with pytest.raises(ValueError, match="query"):
        retriever.retrieve(query)


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_invalid_top_k_is_rejected(top_k: int | float) -> None:
    retriever, _, _ = _retriever()

    with pytest.raises(ValueError, match="top_k"):
        retriever.retrieve("query", top_k=top_k)


def test_empty_vector_store_returns_no_results_without_embedding() -> None:
    retriever, embedder, vector_store = _retriever(matches=[])

    assert retriever.retrieve("query") == []
    assert embedder.queries == []
    assert vector_store.search_calls == []


@pytest.mark.parametrize("threshold", [-1.01, 1.01, float("nan"), float("inf"), True])
def test_invalid_similarity_threshold_is_rejected(threshold: object) -> None:
    with pytest.raises(ValueError, match="score_threshold"):
        RetrieverConfig(score_threshold=threshold)  # type: ignore[arg-type]


def test_retrieve_result_type_is_public() -> None:
    retriever, _, _ = _retriever()

    assert isinstance(retriever.retrieve("query")[0], RetrievalResult)
