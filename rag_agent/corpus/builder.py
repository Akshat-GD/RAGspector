"""Convert normalized HotpotQA examples into corpus documents and chunks."""

from collections.abc import Iterable, Mapping, Sequence
from urllib.parse import quote
from typing import Any

from rag_agent.corpus.chunker import CorpusChunker
from rag_agent.corpus.models import Corpus, CorpusDocument
from rag_agent.models import HotpotQAExample


class CorpusDataError(ValueError):
    """Raised when normalized examples contain malformed corpus data."""


def build_documents(
    examples: Iterable[HotpotQAExample],
) -> tuple[CorpusDocument, ...]:
    """Convert HotpotQA context documents without flattening sentence boundaries.

    A document ID is based on the source example ID and its context position,
    so repeated titles in different examples remain distinct.
    """
    documents: list[CorpusDocument] = []
    seen_example_ids: set[str] = set()
    for example_index, example in enumerate(examples):
        if not isinstance(example, HotpotQAExample):
            raise CorpusDataError(
                f"Example {example_index} must be a HotpotQAExample"
            )
        if not isinstance(example.example_id, str) or not example.example_id.strip():
            raise CorpusDataError(
                f"Example {example_index} must have a non-empty example_id"
            )
        if example.example_id in seen_example_ids:
            raise CorpusDataError(
                f"Example {example_index} duplicates example_id "
                f"{example.example_id!r}"
            )
        seen_example_ids.add(example.example_id)
        if not _is_sequence(example.context):
            raise CorpusDataError(
                f"Example {example_index} context must be a sequence"
            )
        if not isinstance(example.metadata, Mapping) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in example.metadata.items()
        ):
            raise CorpusDataError(
                f"Example {example_index} metadata must map strings to strings"
            )

        normalized_context: list[tuple[str, tuple[str, ...]]] = []
        for document_index, record in enumerate(example.context):
            if not _is_sequence(record) or len(record) != 2:
                raise CorpusDataError(
                    f"Example {example_index} context item {document_index} "
                    "must be a (title, sentences) pair"
                )
            title, raw_sentences = record
            if not isinstance(title, str) or not title.strip():
                raise CorpusDataError(
                    f"Example {example_index} context item {document_index} "
                    "must have a non-empty title"
                )
            if not _is_sequence(raw_sentences):
                raise CorpusDataError(
                    f"Example {example_index} context item {document_index} "
                    "sentences must be a sequence"
                )
            sentences: list[str] = []
            for sentence_index, sentence in enumerate(raw_sentences):
                if not isinstance(sentence, str):
                    raise CorpusDataError(
                        f"Example {example_index} context item {document_index} "
                        f"sentence {sentence_index} must be a string"
                    )
                sentences.append(sentence)
            normalized_context.append((title, tuple(sentences)))

        supporting_by_title = _supporting_sentences(example, example_index)
        document_indices_by_title: dict[str, list[int]] = {}
        for document_index, (title, _) in enumerate(normalized_context):
            document_indices_by_title.setdefault(title, []).append(document_index)

        matched_supporting_by_title: dict[str, tuple[int, ...]] = {}
        for title, support_indices in supporting_by_title.items():
            matching_document_indices = document_indices_by_title.get(title, [])
            for document_index in matching_document_indices:
                sentences = normalized_context[document_index][1]
                invalid_indices = [
                    sentence_index
                    for sentence_index in support_indices
                    if sentence_index >= len(sentences)
                ]
                if invalid_indices:
                    raise CorpusDataError(
                        f"Example {example_index} supporting fact index "
                        f"{invalid_indices[0]} is outside document {title!r}"
                    )
            if len(matching_document_indices) == 1:
                matched_supporting_by_title[title] = tuple(support_indices)

        for document_index, (title, sentences) in enumerate(normalized_context):
            support_indices = matched_supporting_by_title.get(title, ())
            documents.append(
                CorpusDocument(
                    document_id=(
                        f"{quote(example.example_id, safe='')}::context-{document_index}"
                    ),
                    example_id=example.example_id,
                    title=title,
                    sentences=sentences,
                    supporting_sentence_indices=tuple(support_indices),
                    supporting_facts=tuple(example.supporting_facts),
                    metadata=example.metadata,
                )
            )
    return tuple(documents)


def build_corpus(
    examples: Iterable[HotpotQAExample],
    chunker: CorpusChunker | None = None,
) -> Corpus:
    """Build corpus documents and deterministic chunks from examples."""
    documents = build_documents(examples)
    active_chunker = chunker if chunker is not None else CorpusChunker()
    return Corpus(
        documents=documents,
        chunks=active_chunker.chunk_documents(documents),
    )


def _supporting_sentences(
    example: HotpotQAExample,
    example_index: int,
) -> dict[str, list[int]]:
    if not _is_sequence(example.supporting_facts):
        raise CorpusDataError(
            f"Example {example_index} supporting_facts must be a sequence"
        )
    supporting: dict[str, list[int]] = {}
    for fact_index, fact in enumerate(example.supporting_facts):
        if not _is_sequence(fact) or len(fact) != 2:
            raise CorpusDataError(
                f"Example {example_index} supporting fact {fact_index} "
                "must be a (title, sentence_index) pair"
            )
        title, sentence_index = fact
        if not isinstance(title, str) or not title.strip():
            raise CorpusDataError(
                f"Example {example_index} supporting fact {fact_index} "
                "must have a non-empty title"
            )
        if (
            isinstance(sentence_index, bool)
            or not isinstance(sentence_index, int)
            or sentence_index < 0
        ):
            raise CorpusDataError(
                f"Example {example_index} supporting fact {fact_index} "
                "must have a non-negative sentence index"
            )
        supporting.setdefault(title, []).append(sentence_index)
    return supporting


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))