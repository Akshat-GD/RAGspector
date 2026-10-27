"""Sentence-boundary-aware chunking for HotpotQA corpus documents."""

from collections.abc import Iterable
from urllib.parse import quote

from rag_agent.corpus.models import CorpusChunk, CorpusDocument


class CorpusChunker:
    """Split documents into deterministic windows of source sentences.

    ``chunk_size`` and ``overlap`` are measured in sentences, not characters
    or tokens. This uses HotpotQA's native sentence-level context and avoids
    breaking sentence boundaries or losing their source indices.
    """

    def __init__(self, chunk_size: int = 5, overlap: int = 1) -> None:
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
            raise TypeError("chunk_size must be an integer")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if isinstance(overlap, bool) or not isinstance(overlap, int):
            raise TypeError("overlap must be an integer")
        if overlap < 0:
            raise ValueError("overlap cannot be negative")
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_document(self, document: CorpusDocument) -> tuple[CorpusChunk, ...]:
        """Create sentence windows from one document.

        Blank sentences are omitted from chunk text but retain their original
        positions for the indices of all included sentences. Documents with no
        non-blank sentences produce no chunks.
        """
        source_sentences = [
            (index, sentence)
            for index, sentence in enumerate(document.sentences)
            if sentence.strip()
        ]
        chunks: list[CorpusChunk] = []
        start = 0
        while start < len(source_sentences):
            end = min(start + self.chunk_size, len(source_sentences))
            window = source_sentences[start:end]
            source_indices = tuple(index for index, _ in window)
            chunks.append(
                CorpusChunk(
                    chunk_id=(
                        f"{quote(document.document_id, safe='')}::chunk-{len(chunks)}"
                    ),
                    document_id=document.document_id,
                    example_id=document.example_id,
                    source_title=document.title,
                    text="\n".join(sentence for _, sentence in window),
                    source_sentence_indices=source_indices,
                    contains_supporting_fact=any(
                        index in document.supporting_sentence_indices
                        for index in source_indices
                    ),
                    metadata=document.metadata,
                )
            )
            if end == len(source_sentences):
                break
            start = end - self.overlap
        return tuple(chunks)

    def chunk_documents(
        self,
        documents: Iterable[CorpusDocument],
    ) -> tuple[CorpusChunk, ...]:
        """Chunk documents in input order."""
        return tuple(
            chunk
            for document in documents
            for chunk in self.chunk_document(document)
        )