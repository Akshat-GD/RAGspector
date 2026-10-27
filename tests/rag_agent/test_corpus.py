"""Tests for HotpotQA corpus building and sentence-based chunking."""

from dataclasses import replace

import pytest

from rag_agent.corpus.builder import CorpusDataError, build_corpus, build_documents
from rag_agent.corpus.chunker import CorpusChunker
from rag_agent.models import HotpotQAExample


def _example(
    example_id: str = "case-1",
    *,
    sentences: tuple[str, ...] = (
        "Sentence zero.",
        "Sentence one.",
        "Sentence two.",
        "Sentence three.",
        "Sentence four.",
    ),
) -> HotpotQAExample:
    return HotpotQAExample(
        example_id=example_id,
        question="What happened?",
        answer="An example.",
        context=(("Example Article", sentences),),
        supporting_facts=tuple(
            ("Example Article", index)
            for index in (1, 4)
            if index < len(sentences)
        ),
        metadata={"type": "bridge", "level": "easy"},
    )


def test_builds_corpus_with_document_identity_and_metadata() -> None:
    corpus = build_corpus([_example()], CorpusChunker(chunk_size=3, overlap=1))

    assert len(corpus.documents) == 1
    document = corpus.documents[0]
    assert document.example_id == "case-1"
    assert document.title == "Example Article"
    assert document.sentences == _example().context[0][1]
    assert document.document_id.startswith("case-1::context-")
    assert dict(document.metadata) == {"type": "bridge", "level": "easy"}
    assert corpus.chunks[0].document_id == document.document_id
    assert corpus.chunks[0].source_title == "Example Article"
    assert corpus.chunks[0].example_id == "case-1"


def test_chunks_preserve_supporting_fact_and_source_indices() -> None:
    corpus = build_corpus([_example()], CorpusChunker(chunk_size=3, overlap=1))

    assert [chunk.source_sentence_indices for chunk in corpus.chunks] == [
        (0, 1, 2),
        (2, 3, 4),
    ]
    assert [chunk.contains_supporting_fact for chunk in corpus.chunks] == [True, True]
    assert dict(corpus.chunks[0].metadata) == {"type": "bridge", "level": "easy"}


def test_exact_supporting_title_match_marks_only_matching_document() -> None:
    example = replace(
        _example(),
        context=(
            ("Example Article", ("Sentence zero.", "Sentence one.")),
            ("Other Article", ("Other sentence.",)),
        ),
        supporting_facts=(("Example Article", 1),),
    )

    documents = build_documents([example])

    assert documents[0].supporting_sentence_indices == (1,)
    assert documents[1].supporting_sentence_indices == ()
    assert documents[0].supporting_facts == example.supporting_facts
    assert documents[1].supporting_facts == example.supporting_facts


def test_unmatched_supporting_titles_are_preserved_without_markers() -> None:
    original_supporting_facts = (
        ("Scott Derrickson", 0),
        ("Ed Wood", 0),
    )
    example = replace(
        _example(),
        context=(("Ed Wood (film)", ("Film sentence.",)),),
        supporting_facts=original_supporting_facts,
    )

    documents = build_documents([example])

    assert len(documents) == 1
    assert documents[0].title == "Ed Wood (film)"
    assert documents[0].supporting_sentence_indices == ()
    assert documents[0].supporting_facts == original_supporting_facts


def test_chunk_ids_are_deterministic_for_same_input_and_configuration() -> None:
    chunker = CorpusChunker(chunk_size=2, overlap=1)

    first = build_corpus([_example()], chunker)
    second = build_corpus([_example()], chunker)

    assert [chunk.chunk_id for chunk in first.chunks] == [
        chunk.chunk_id for chunk in second.chunks
    ]
    assert len({chunk.chunk_id for chunk in first.chunks}) == len(first.chunks)


def test_chunk_size_and_overlap_define_sentence_windows() -> None:
    chunks = CorpusChunker(chunk_size=2, overlap=1).chunk_document(
        build_documents([_example()])[0]
    )

    assert [chunk.text.splitlines() for chunk in chunks] == [
        ["Sentence zero.", "Sentence one."],
        ["Sentence one.", "Sentence two."],
        ["Sentence two.", "Sentence three."],
        ["Sentence three.", "Sentence four."],
    ]
    assert [chunk.source_sentence_indices for chunk in chunks] == [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
    ]


def test_small_document_is_not_split_unnecessarily() -> None:
    document = build_documents(
        [_example(sentences=("Short sentence.", "Another short sentence."))]
    )[0]

    chunks = CorpusChunker(chunk_size=5, overlap=1).chunk_document(document)

    assert len(chunks) == 1
    assert chunks[0].text == "Short sentence.\nAnother short sentence."
    assert chunks[0].source_sentence_indices == (0, 1)


def test_empty_document_is_preserved_without_empty_chunks() -> None:
    documents = build_documents([_example(sentences=())])
    corpus = build_corpus([_example(sentences=())])

    assert len(documents) == 1
    assert documents[0].sentences == ()
    assert corpus.chunks == ()


def test_blank_sentences_are_skipped_without_shifting_source_indices() -> None:
    document = build_documents(
        [_example(sentences=("First.", "  ", "Third."))]
    )[0]

    chunks = CorpusChunker(chunk_size=2, overlap=0).chunk_document(document)

    assert chunks[0].text == "First.\nThird."
    assert chunks[0].source_sentence_indices == (0, 2)


def test_malformed_sentence_content_fails_explicitly() -> None:
    malformed = replace(_example(), context=(("Example Article", ("Valid.", None)),))

    with pytest.raises(CorpusDataError, match="sentence 1 must be a string"):
        build_documents([malformed])


def test_supporting_fact_outside_document_fails_explicitly() -> None:
    malformed = replace(
        _example(sentences=("Only sentence.",)),
        supporting_facts=(("Example Article", 2),),
    )

    with pytest.raises(CorpusDataError, match="outside document"):
        build_documents([malformed])


def test_duplicate_example_ids_are_rejected_to_keep_document_ids_unique() -> None:
    with pytest.raises(CorpusDataError, match="duplicates example_id"):
        build_corpus([_example(), _example()])


def test_chunker_rejects_invalid_size_and_overlap() -> None:
    with pytest.raises(ValueError, match="chunk_size"):
        CorpusChunker(chunk_size=0)
    with pytest.raises(ValueError, match="overlap"):
        CorpusChunker(chunk_size=2, overlap=2)
