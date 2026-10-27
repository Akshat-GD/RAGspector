"""Data models for normalized corpus documents and chunks."""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class CorpusDocument:
    """A source document that retains its original sentence boundaries."""

    document_id: str
    example_id: str
    title: str
    sentences: tuple[str, ...]
    supporting_sentence_indices: tuple[int, ...] = ()
    metadata: Mapping[str, str] = field(default_factory=dict)
    supporting_facts: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class CorpusChunk:
    """A chunk of a corpus document with provenance and evidence metadata."""

    chunk_id: str
    document_id: str
    example_id: str
    source_title: str
    text: str
    source_sentence_indices: tuple[int, ...]
    contains_supporting_fact: bool
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class Corpus:
    """The normalized source documents and their retrieval-ready chunks."""

    documents: tuple[CorpusDocument, ...]
    chunks: tuple[CorpusChunk, ...]
