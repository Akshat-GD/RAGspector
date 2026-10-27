"""Standalone orchestration for the modular retrieval-augmented agent."""

from collections.abc import Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping
from time import perf_counter

from rag_agent.context.builder import (
    BuiltContext,
    ContextBuilder,
)
from rag_agent.corpus.models import CorpusChunk
from rag_agent.generation.base import GenerationResult, Generator
from rag_agent.retrieval.retriever import RetrievalResult, Retriever


class RAGAgentInputError(ValueError):
    """Raised when an agent request contains invalid values."""


@dataclass(frozen=True)
class RAGAgentRequest:
    """Input to one orchestration run.

    The retriever's configured score threshold and the injected context
    builder's configuration remain authoritative; this request only exposes
    overrides supported directly by their interfaces.
    """

    query: str
    top_k: int | None = None
    context_order: tuple[str, ...] | None = None
    controlled_evidence: tuple[CorpusChunk, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.query, str) or not self.query.strip():
            raise RAGAgentInputError("query must be a non-empty string")
        if self.top_k is not None and (
            isinstance(self.top_k, bool)
            or not isinstance(self.top_k, int)
            or self.top_k <= 0
        ):
            raise RAGAgentInputError("top_k must be a positive integer or None")
        if self.context_order is not None:
            if isinstance(self.context_order, (str, bytes)) or not isinstance(
                self.context_order, Sequence
            ):
                raise RAGAgentInputError(
                    "context_order must be a sequence of chunk IDs or None"
                )
            if any(
                not isinstance(chunk_id, str) or not chunk_id
                for chunk_id in self.context_order
            ):
                raise RAGAgentInputError(
                    "context_order must contain non-empty chunk IDs"
                )
            object.__setattr__(self, "context_order", tuple(self.context_order))
        if isinstance(self.controlled_evidence, (str, bytes)) or not isinstance(
            self.controlled_evidence, Sequence
        ):
            raise RAGAgentInputError("controlled_evidence must be a sequence")
        if any(not isinstance(chunk, CorpusChunk) for chunk in self.controlled_evidence):
            raise RAGAgentInputError(
                "controlled_evidence must contain only CorpusChunk objects"
            )
        object.__setattr__(
            self,
            "controlled_evidence",
            tuple(self.controlled_evidence),
        )


@dataclass(frozen=True)
class RAGAgentExecutionMetadata:
    """Counts and per-stage durations for one agent execution."""

    retrieved_count: int
    context_item_count: int
    retrieval_seconds: float
    context_build_seconds: float
    generation_seconds: float
    total_seconds: float


@dataclass(frozen=True)
class RAGAgentResult:
    """Complete output and provenance from one standalone RAG run."""

    query: str
    answer: str
    retrieved_results: tuple[RetrievalResult, ...]
    context: BuiltContext
    generation: GenerationResult
    execution: RAGAgentExecutionMetadata
    metadata: Mapping[str, str | int | float | bool]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


class RAGAgent:
    """Orchestrate retrieval, context construction, and answer generation.

    Components are injected and can be independently replaced with test fakes.
    Exceptions from any component intentionally propagate unchanged.
    """

    def __init__(
        self,
        retriever: Retriever,
        context_builder: ContextBuilder,
        generator: Generator,
    ) -> None:
        self.retriever = retriever
        self.context_builder = context_builder
        self.generator = generator

    def run(self, request: RAGAgentRequest) -> RAGAgentResult:
        """Execute a request through retrieval, context building, and generation."""
        if not isinstance(request, RAGAgentRequest):
            raise RAGAgentInputError("request must be an RAGAgentRequest")

        total_started = perf_counter()
        stage_started = perf_counter()
        retrieval_results = tuple(
            self.retriever.retrieve(request.query, top_k=request.top_k)
        )
        retrieval_seconds = perf_counter() - stage_started

        stage_started = perf_counter()
        context = self.context_builder.build(
            retrieval_results,
            order=request.context_order,
            controlled_evidence=request.controlled_evidence,
        )
        context_build_seconds = perf_counter() - stage_started
        if not isinstance(context, BuiltContext):
            raise TypeError("context builder must return a BuiltContext")

        stage_started = perf_counter()
        generation = self.generator.generate(request.query, context)
        generation_seconds = perf_counter() - stage_started
        if not isinstance(generation, GenerationResult):
            raise TypeError("generator must return a GenerationResult")

        total_seconds = perf_counter() - total_started
        execution = RAGAgentExecutionMetadata(
            retrieved_count=len(retrieval_results),
            context_item_count=len(context.items),
            retrieval_seconds=retrieval_seconds,
            context_build_seconds=context_build_seconds,
            generation_seconds=generation_seconds,
            total_seconds=total_seconds,
        )
        return RAGAgentResult(
            query=request.query,
            answer=generation.answer,
            retrieved_results=retrieval_results,
            context=context,
            generation=generation,
            execution=execution,
            metadata={
                "retriever": type(self.retriever).__name__,
                "context_builder": type(self.context_builder).__name__,
                "generator": generation.model_name,
            },
        )