from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class HotpotQAExample:
    example_id: str
    question: str
    answer: str
    context: tuple[tuple[str, tuple[str, ...]], ...]
    supporting_facts: tuple[tuple[str, int], ...]
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Document:
    document_id: str
    text: str
    title: str | None = None
    source_id: str | None = None


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    document_id: str
    text: str
    chunk_index: int
