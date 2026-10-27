"""Tests for end-to-end standalone RAG agent orchestration."""

from collections.abc import Sequence

import pytest

from rag_agent import (
    RAGAgent,
    RAGAgentExecutionMetadata,
    RAGAgentInputError,
    RAGAgentRequest,
    RAGAgentResult,
)
from rag_agent.context import (
    BuiltContext,
    ContextBuilder,
    ContextItem,
    RetrievalContextBuilder,
)
from rag_agent.corpus.models import CorpusChunk
from rag_agent.generation import GenerationResult, Generator
from rag_agent.retrieval import RetrievalResult, Retriever


def _chunk(chunk_id: str, *, origin: str = "retrieved") -> CorpusChunk:
    return CorpusChunk(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        example_id=f"example-{chunk_id}",
        source_title=f"Source {chunk_id}",
        text=f"Evidence {chunk_id}",
        source_sentence_indices=(1, 2),
        contains_supporting_fact=True,
        metadata={"origin": origin, "id": chunk_id},
    )


def _retrieval_results() -> list[RetrievalResult]:
    return [
        RetrievalResult(_chunk("A"), score=0.93, rank=1),
        RetrievalResult(_chunk("B"), score=0.82, rank=2),
    ]


class FakeRetriever(Retriever):
    def __init__(
        self,
        results: list[RetrievalResult] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.results = _retrieval_results() if results is None else results
        self.error = error
        self.calls: list[tuple[str, int | None]] = []

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        self.calls.append((query, top_k))
        if self.error is not None:
            raise self.error
        return self.results


class RecordingContextBuilder(ContextBuilder):
    def __init__(self) -> None:
        self.received_results: tuple[RetrievalResult, ...] | None = None
        self.received_order: Sequence[str] | None = None
        self.received_controlled: Sequence[CorpusChunk] | None = None
        self.delegate = RetrievalContextBuilder()

    def build(
        self,
        retrieval_results: Sequence[RetrievalResult],
        *,
        order: Sequence[str] | None = None,
        controlled_evidence: Sequence[CorpusChunk] = (),
    ) -> BuiltContext:
        self.received_results = tuple(retrieval_results)
        self.received_order = order
        self.received_controlled = controlled_evidence
        return self.delegate.build(
            retrieval_results,
            order=order,
            controlled_evidence=controlled_evidence,
        )


class FakeGenerator(Generator):
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.received_query: str | None = None
        self.received_context: BuiltContext | None = None

    def generate(self, query: str, context: BuiltContext) -> GenerationResult:
        self.received_query = query
        self.received_context = context
        if self.error is not None:
            raise self.error
        return GenerationResult(
            answer="Answer from supplied evidence",
            model_name="fake-generator",
            latency_seconds=0.01,
            metadata={"backend": "test"},
        )


def _agent(
    *,
    results: list[RetrievalResult] | None = None,
    retriever_error: Exception | None = None,
    generator_error: Exception | None = None,
) -> tuple[RAGAgent, FakeRetriever, RecordingContextBuilder, FakeGenerator]:
    retriever = FakeRetriever(results, retriever_error)
    context_builder = RecordingContextBuilder()
    generator = FakeGenerator(generator_error)
    return (
        RAGAgent(retriever, context_builder, generator),
        retriever,
        context_builder,
        generator,
    )


def test_complete_pipeline_returns_answer_and_execution_metadata() -> None:
    agent, retriever, builder, generator = _agent()
    request = RAGAgentRequest(query="What happened?", top_k=4)

    result = agent.run(request)

    assert isinstance(result, RAGAgentResult)
    assert result.query == "What happened?"
    assert result.answer == "Answer from supplied evidence"
    assert result.generation.model_name == "fake-generator"
    assert retriever.calls == [("What happened?", 4)]
    assert builder.received_results == tuple(_retrieval_results())
    assert generator.received_query == "What happened?"
    assert generator.received_context is result.context
    assert [item.chunk_id for item in result.context.items] == ["A", "B"]
    assert isinstance(result.execution, RAGAgentExecutionMetadata)
    assert result.execution.retrieved_count == 2
    assert result.execution.context_item_count == 2
    assert result.execution.total_seconds >= 0
    assert dict(result.metadata) == {
        "retriever": "FakeRetriever",
        "context_builder": "RecordingContextBuilder",
        "generator": "fake-generator",
    }


def test_query_top_k_order_and_controlled_evidence_are_propagated() -> None:
    agent, retriever, builder, generator = _agent()
    controlled = _chunk("C", origin="controlled")
    request = RAGAgentRequest(
        query="Where?",
        top_k=7,
        context_order=("B", "A"),
        controlled_evidence=(controlled,),
    )

    result = agent.run(request)

    assert retriever.calls == [("Where?", 7)]
    assert builder.received_order == ("B", "A")
    assert builder.received_controlled == (controlled,)
    assert generator.received_context is result.context
    assert [item.chunk_id for item in result.context.items] == ["B", "A", "C"]
    assert [item.evidence_type for item in result.context.items] == [
        "retrieved",
        "retrieved",
        "controlled",
    ]


def test_result_preserves_retrieval_and_context_provenance() -> None:
    agent, _, _, _ = _agent()

    result = agent.run(RAGAgentRequest(query="Question?"))

    assert [(item.chunk.chunk_id, item.score, item.rank) for item in result.retrieved_results] == [
        ("A", 0.93, 1),
        ("B", 0.82, 2),
    ]
    context_item = result.context.items[0]
    assert context_item.chunk_id == "A"
    assert context_item.document_id == "doc-A"
    assert context_item.source_title == "Source A"
    assert context_item.source_example_id == "example-A"
    assert context_item.source_sentence_indices == (1, 2)
    assert context_item.contains_supporting_fact is True
    assert context_item.retrieval_score == 0.93
    assert context_item.retrieval_rank == 1
    assert context_item.evidence_type == "retrieved"
    assert dict(context_item.metadata) == {"origin": "retrieved", "id": "A"}


def test_empty_retrieval_still_builds_context_and_generates() -> None:
    agent, _, builder, generator = _agent(results=[])

    result = agent.run(RAGAgentRequest(query="No evidence?"))

    assert result.retrieved_results == ()
    assert result.context.items == ()
    assert builder.received_results == ()
    assert generator.received_context == result.context
    assert result.answer == "Answer from supplied evidence"
    assert result.execution.retrieved_count == 0


def test_retrieval_exception_propagates_unchanged() -> None:
    original_error = RuntimeError("retrieval backend failed")
    agent, _, builder, generator = _agent(retriever_error=original_error)

    with pytest.raises(RuntimeError) as raised:
        agent.run(RAGAgentRequest(query="Question?"))

    assert raised.value is original_error
    assert builder.received_results is None
    assert generator.received_context is None


def test_generation_exception_propagates_unchanged() -> None:
    original_error = RuntimeError("generation backend failed")
    agent, _, builder, generator = _agent(generator_error=original_error)

    with pytest.raises(RuntimeError) as raised:
        agent.run(RAGAgentRequest(query="Question?"))

    assert raised.value is original_error
    assert builder.received_results == tuple(_retrieval_results())
    assert generator.received_context is not None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"query": ""},
        {"query": " \t"},
        {"query": "valid", "top_k": 0},
        {"query": "valid", "top_k": True},
        {"query": "valid", "context_order": ("",)},
        {"query": "valid", "controlled_evidence": ("not a chunk",)},
    ],
)
def test_request_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(RAGAgentInputError):
        RAGAgentRequest(**kwargs)  # type: ignore[arg-type]


def test_agent_rejects_non_request_input() -> None:
    agent, _, _, _ = _agent()

    with pytest.raises(RAGAgentInputError, match="RAGAgentRequest"):
        agent.run(None)  # type: ignore[arg-type]


def test_default_top_k_is_delegated_to_retriever_configuration() -> None:
    agent, retriever, _, _ = _agent()

    agent.run(RAGAgentRequest(query="Question?"))

    assert retriever.calls == [("Question?", None)]


def test_context_builder_order_validation_error_propagates() -> None:
    agent, _, _, _ = _agent()

    with pytest.raises(ValueError, match="order"):
        agent.run(
            RAGAgentRequest(
                query="Question?",
                context_order=("unknown",),
            )
        )


def test_context_provenance_is_not_replaced_by_generation_result() -> None:
    agent, _, _, _ = _agent()

    result = agent.run(RAGAgentRequest(query="Question?"))

    assert result.generation.answer == result.answer
    assert isinstance(result.context.items[0], ContextItem)
