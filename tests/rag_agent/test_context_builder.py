"""Tests for context construction and controlled evidence injection."""

import pytest

from rag_agent.context import (
    ContextBuilderConfig,
    ContextLimitError,
    InvalidContextOrderError,
    RetrievalContextBuilder,
)
from rag_agent.corpus.models import CorpusChunk
from rag_agent.retrieval import RetrievalResult


def _chunk(chunk_id: str, *, supporting: bool = False) -> CorpusChunk:
    return CorpusChunk(
        chunk_id=chunk_id,
        document_id=f"document-{chunk_id}",
        example_id=f"example-{chunk_id}",
        source_title=f"Title {chunk_id}",
        text=f"Evidence text {chunk_id}",
        source_sentence_indices=(1, 3),
        contains_supporting_fact=supporting,
        metadata={"origin": "fixture", "chunk": chunk_id},
    )


def _results() -> list[RetrievalResult]:
    return [
        RetrievalResult(_chunk("A", supporting=True), score=0.9, rank=1),
        RetrievalResult(_chunk("B"), score=0.8, rank=2),
        RetrievalResult(_chunk("C"), score=0.7, rank=3),
    ]


def test_builds_context_from_retrieval_results() -> None:
    context = RetrievalContextBuilder().build(_results())

    assert [item.chunk_id for item in context.items] == ["A", "B", "C"]
    assert context.text == (
        "Evidence text A\n\nEvidence text B\n\nEvidence text C"
    )
    assert [item.evidence_type for item in context.items] == [
        "retrieved",
        "retrieved",
        "retrieved",
    ]


def test_maximum_chunk_limit_is_applied_after_ordering() -> None:
    context = RetrievalContextBuilder(
        ContextBuilderConfig(max_chunks=2)
    ).build(_results())

    assert [item.chunk_id for item in context.items] == ["A", "B"]


def test_explicit_order_reorders_without_mutating_retrieval_metadata() -> None:
    context = RetrievalContextBuilder().build(
        _results(),
        order=["C", "A", "B"],
    )

    assert [item.chunk_id for item in context.items] == ["C", "A", "B"]
    assert [(item.retrieval_score, item.retrieval_rank) for item in context.items] == [
        (0.7, 3),
        (0.9, 1),
        (0.8, 2),
    ]


@pytest.mark.parametrize(
    "order",
    [
        ["A", "B", "unknown"],
        ["A", "B"],
        ["A", "A", "C"],
    ],
)
def test_invalid_order_ids_are_rejected(order: list[str]) -> None:
    with pytest.raises(InvalidContextOrderError):
        RetrievalContextBuilder().build(_results(), order=order)


def test_context_preserves_all_chunk_provenance_and_metadata() -> None:
    result = _results()[0]

    item = RetrievalContextBuilder().build([result]).items[0]

    assert item.chunk_id == result.chunk.chunk_id
    assert item.text == result.chunk.text
    assert item.document_id == result.chunk.document_id
    assert item.source_title == result.chunk.source_title
    assert item.source_example_id == result.chunk.example_id
    assert item.source_sentence_indices == (1, 3)
    assert item.contains_supporting_fact is True
    assert item.retrieval_score == result.score
    assert item.retrieval_rank == result.rank
    assert dict(item.metadata) == {"origin": "fixture", "chunk": "A"}


def test_controlled_evidence_is_explicit_and_distinguishable() -> None:
    controlled = _chunk("CONTROLLED", supporting=True)

    context = RetrievalContextBuilder().build(
        _results(),
        controlled_evidence=[controlled],
    )

    injected = context.items[-1]
    assert injected.chunk_id == "CONTROLLED"
    assert injected.text == "Evidence text CONTROLLED"
    assert injected.evidence_type == "controlled"
    assert injected.retrieval_score is None
    assert injected.retrieval_rank is None
    assert injected.contains_supporting_fact is True
    assert "answer" not in injected.text.lower()


def test_supporting_fact_flag_does_not_inject_unretrieved_evidence() -> None:
    results = [RetrievalResult(_chunk("A", supporting=True), score=0.9, rank=1)]

    context = RetrievalContextBuilder().build(results)

    assert [item.chunk_id for item in context.items] == ["A"]


def test_retrieved_and_controlled_duplicate_is_deduplicated_by_default() -> None:
    retrieved = _results()[0]
    duplicate = _chunk("A", supporting=True)

    context = RetrievalContextBuilder().build(
        [retrieved],
        controlled_evidence=[duplicate],
    )

    assert len(context.items) == 1
    assert context.items[0].evidence_type == "retrieved"
    assert context.items[0].retrieval_score == retrieved.score


def test_duplicate_can_be_explicitly_retained_by_configuration() -> None:
    result = _results()[0]
    context = RetrievalContextBuilder(
        ContextBuilderConfig(allow_duplicate_chunks=True)
    ).build([result], controlled_evidence=[_chunk("A")])

    assert [item.evidence_type for item in context.items] == [
        "retrieved",
        "controlled",
    ]
    assert [item.chunk_id for item in context.items] == ["A", "A"]


def test_duplicate_controlled_evidence_is_deduplicated_by_default() -> None:
    controlled = _chunk("CONTROLLED")
    context = RetrievalContextBuilder().build(
        [],
        controlled_evidence=[controlled, controlled],
    )

    assert [item.chunk_id for item in context.items] == ["CONTROLLED"]


def test_controlled_evidence_is_kept_within_chunk_limit() -> None:
    context = RetrievalContextBuilder(
        ContextBuilderConfig(max_chunks=2)
    ).build(
        _results(),
        controlled_evidence=[_chunk("CONTROLLED")],
    )

    assert [item.chunk_id for item in context.items] == ["A", "CONTROLLED"]
    assert context.items[-1].evidence_type == "controlled"


def test_controlled_evidence_over_limit_raises_explicit_error() -> None:
    with pytest.raises(ContextLimitError, match="controlled evidence"):
        RetrievalContextBuilder(
            ContextBuilderConfig(max_chunks=1)
        ).build(
            [],
            controlled_evidence=[_chunk("X"), _chunk("Y")],
        )


def test_output_order_is_deterministic() -> None:
    builder = RetrievalContextBuilder()

    first = builder.build(
        _results(),
        order=["C", "A", "B"],
        controlled_evidence=[_chunk("D")],
    )
    second = builder.build(
        _results(),
        order=["C", "A", "B"],
        controlled_evidence=[_chunk("D")],
    )

    assert first == second


def test_empty_retrieval_results_produce_empty_context() -> None:
    context = RetrievalContextBuilder().build([])

    assert context.items == ()
    assert context.text == ""


def test_context_builder_rejects_non_positive_max_chunks() -> None:
    with pytest.raises(ValueError, match="max_chunks"):
        ContextBuilderConfig(max_chunks=0)

