"""Build ordered generator context from retrieval and explicit evidence."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping

from rag_agent.corpus.models import CorpusChunk
from rag_agent.retrieval.retriever import RetrievalResult


ContextEvidenceType = Literal["retrieved", "controlled"]


class ContextBuildError(ValueError):
    """Raised when context input or configuration is invalid."""


class InvalidContextOrderError(ContextBuildError):
    """Raised when an explicit order does not cover retrieved chunks exactly."""


class ContextLimitError(ContextBuildError):
    """Raised when controlled evidence cannot fit the configured context limit."""


@dataclass(frozen=True)
class ContextItem:
    """A context entry with chunk provenance and evidence origin."""

    chunk_id: str
    text: str
    document_id: str
    source_title: str
    source_example_id: str
    source_sentence_indices: tuple[int, ...]
    contains_supporting_fact: bool
    retrieval_score: float | None
    retrieval_rank: int | None
    evidence_type: ContextEvidenceType
    metadata: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class BuiltContext:
    """Ordered context items ready for a later generator."""

    items: tuple[ContextItem, ...]

    @property
    def text(self) -> str:
        """Return context item text in order, separated by blank lines."""
        return "\n\n".join(item.text for item in self.items)


@dataclass(frozen=True)
class ContextBuilderConfig:
    """Context size and explicit duplicate policy.

    ``max_chunks`` limits the final number of context items. Explicit
    controlled evidence is always retained; retrieved chunks are trimmed to
    make room. If the controlled evidence itself exceeds the limit, building
    fails explicitly.
    """

    max_chunks: int | None = None
    allow_duplicate_chunks: bool = False

    def __post_init__(self) -> None:
        if self.max_chunks is not None and (
            isinstance(self.max_chunks, bool)
            or not isinstance(self.max_chunks, int)
            or self.max_chunks <= 0
        ):
            raise ValueError("max_chunks must be a positive integer or None")
        if not isinstance(self.allow_duplicate_chunks, bool):
            raise ValueError("allow_duplicate_chunks must be a boolean")


class ContextBuilder(ABC):
    """Abstraction for converting retrieval results into ordered context."""

    @abstractmethod
    def build(
        self,
        retrieval_results: Sequence[RetrievalResult],
        *,
        order: Sequence[str] | None = None,
        controlled_evidence: Sequence[CorpusChunk] = (),
    ) -> BuiltContext:
        """Build context without generating or modifying evidence text."""


class RetrievalContextBuilder(ContextBuilder):
    """Construct context from retrieved chunks and optionally injected evidence.

    Retrieved results are ordered according to their retrieval order unless
    ``order`` is supplied. Controlled evidence is explicitly supplied by the
    caller and appended after retrieved evidence. It is never inferred from
    the supporting-fact flag.
    """

    def __init__(self, config: ContextBuilderConfig | None = None) -> None:
        self.config = config if config is not None else ContextBuilderConfig()

    def build(
        self,
        retrieval_results: Sequence[RetrievalResult],
        *,
        order: Sequence[str] | None = None,
        controlled_evidence: Sequence[CorpusChunk] = (),
    ) -> BuiltContext:
        """Create a context preserving ranks, scores, provenance, and origin."""
        if not _is_sequence(retrieval_results):
            raise ContextBuildError("retrieval_results must be a sequence")
        if not _is_sequence(controlled_evidence):
            raise ContextBuildError("controlled_evidence must be a sequence")

        retrieval_by_id: dict[str, RetrievalResult] = {}
        for result_index, result in enumerate(retrieval_results):
            if not isinstance(result, RetrievalResult):
                raise ContextBuildError(
                    f"retrieval result at index {result_index} "
                    "must be a RetrievalResult"
                )
            chunk_id = result.chunk.chunk_id
            if not isinstance(chunk_id, str) or not chunk_id:
                raise ContextBuildError(
                    f"retrieval result at index {result_index} has an invalid chunk ID"
                )
            if chunk_id in retrieval_by_id:
                raise ContextBuildError(f"Duplicate retrieved chunk ID: {chunk_id!r}")
            retrieval_by_id[chunk_id] = result

        ordered_ids = self._ordered_ids(retrieval_by_id, order)
        retrieved_chunks = [retrieval_by_id[chunk_id] for chunk_id in ordered_ids]

        controlled_chunks: list[CorpusChunk] = []
        seen_controlled_ids: set[str] = set()
        retrieved_ids = set(retrieval_by_id)
        for evidence_index, chunk in enumerate(controlled_evidence):
            if not isinstance(chunk, CorpusChunk):
                raise ContextBuildError(
                    f"controlled evidence at index {evidence_index} "
                    "must be a CorpusChunk"
                )
            if not isinstance(chunk.chunk_id, str) or not chunk.chunk_id:
                raise ContextBuildError(
                    f"controlled evidence at index {evidence_index} "
                    "has an invalid chunk ID"
                )
            if not self.config.allow_duplicate_chunks:
                if (
                    chunk.chunk_id in retrieved_ids
                    or chunk.chunk_id in seen_controlled_ids
                ):
                    continue
            seen_controlled_ids.add(chunk.chunk_id)
            controlled_chunks.append(chunk)

        max_chunks = self.config.max_chunks
        if max_chunks is not None and len(controlled_chunks) > max_chunks:
            raise ContextLimitError(
                f"{len(controlled_chunks)} controlled evidence chunks exceed "
                f"max_chunks={max_chunks}"
            )
        if max_chunks is not None:
            retrieval_capacity = max_chunks - len(controlled_chunks)
            retrieved_chunks = retrieved_chunks[:retrieval_capacity]

        items = [
            _retrieved_context_item(result)
            for result in retrieved_chunks
        ]
        items.extend(_controlled_context_item(chunk) for chunk in controlled_chunks)
        return BuiltContext(items=tuple(items))

    @staticmethod
    def _ordered_ids(
        retrieval_by_id: Mapping[str, RetrievalResult],
        order: Sequence[str] | None,
    ) -> list[str]:
        available_ids = list(retrieval_by_id)
        if order is None:
            return available_ids
        if not _is_sequence(order) or any(
            not isinstance(chunk_id, str) or not chunk_id for chunk_id in order
        ):
            raise InvalidContextOrderError(
                "order must be a sequence of non-empty chunk IDs"
            )
        requested_ids = list(order)
        if len(requested_ids) != len(set(requested_ids)):
            raise InvalidContextOrderError("order must not contain duplicate chunk IDs")
        missing_ids = set(available_ids).difference(requested_ids)
        unknown_ids = set(requested_ids).difference(available_ids)
        if missing_ids or unknown_ids or len(requested_ids) != len(available_ids):
            details: list[str] = []
            if missing_ids:
                details.append(f"missing IDs: {', '.join(sorted(missing_ids))}")
            if unknown_ids:
                details.append(f"unknown IDs: {', '.join(sorted(unknown_ids))}")
            raise InvalidContextOrderError(
                "order must contain every retrieved chunk ID exactly once"
                + (f" ({'; '.join(details)})" if details else "")
            )
        return requested_ids


def _retrieved_context_item(result: RetrievalResult) -> ContextItem:
    return _context_item(
        result.chunk,
        evidence_type="retrieved",
        retrieval_score=result.score,
        retrieval_rank=result.rank,
    )


def _controlled_context_item(chunk: CorpusChunk) -> ContextItem:
    return _context_item(
        chunk,
        evidence_type="controlled",
        retrieval_score=None,
        retrieval_rank=None,
    )


def _context_item(
    chunk: CorpusChunk,
    *,
    evidence_type: ContextEvidenceType,
    retrieval_score: float | None,
    retrieval_rank: int | None,
) -> ContextItem:
    return ContextItem(
        chunk_id=chunk.chunk_id,
        text=chunk.text,
        document_id=chunk.document_id,
        source_title=chunk.source_title,
        source_example_id=chunk.example_id,
        source_sentence_indices=chunk.source_sentence_indices,
        contains_supporting_fact=chunk.contains_supporting_fact,
        retrieval_score=retrieval_score,
        retrieval_rank=retrieval_rank,
        evidence_type=evidence_type,
        metadata=chunk.metadata,
    )


def _is_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))