"""Optional real-component end-to-end smoke demo.

Run ``python -m rag_agent.demo`` after installing the dataset, embedding,
FAISS, and local-generation dependencies. The HotpotQA sample is limited to
three examples by default. Local generation is cache-only by default; use
``--allow-model-download`` only if downloading the configured generator model
is acceptable.
"""

import argparse
from dataclasses import dataclass

from rag_agent.agent import RAGAgent, RAGAgentRequest, RAGAgentResult
from rag_agent.context import ContextBuilderConfig, RetrievalContextBuilder
from rag_agent.corpus.builder import build_corpus
from rag_agent.dataset.hotpotqa_loader import load_hotpotqa
from rag_agent.embeddings import (
    DEFAULT_MODEL_NAME,
    SentenceTransformerConfig,
    SentenceTransformerEmbedder,
)
from rag_agent.generation import LocalGeneratorConfig, LocalTransformersGenerator
from rag_agent.retrieval import (
    EmbeddingVectorStoreRetriever,
    FaissVectorStore,
    RetrieverConfig,
)


@dataclass(frozen=True)
class DemoRun:
    """Observable output and index counts from the real component pipeline."""

    sample_size: int
    indexed_chunk_count: int
    results: tuple[RAGAgentResult, ...]


def run_demo(
    *,
    sample_size: int = 3,
    question_count: int = 2,
    top_k: int = 8,
    generator_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    allow_model_download: bool = False,
) -> DemoRun:
    """Run a small real HotpotQA-to-local-answer pipeline.

    Optional dependency, dataset, embedding-model, and generation-model errors
    propagate from their components without mock fallbacks.
    """
    for name, value in (
        ("sample_size", sample_size),
        ("question_count", question_count),
        ("top_k", top_k),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if question_count > sample_size:
        raise ValueError("question_count cannot exceed sample_size")
    if not isinstance(generator_model, str) or not generator_model.strip():
        raise ValueError("generator_model must be a non-empty model name or path")
    if not isinstance(allow_model_download, bool):
        raise ValueError("allow_model_download must be a boolean")

    embedder = SentenceTransformerEmbedder(
        SentenceTransformerConfig(model_name=DEFAULT_MODEL_NAME)
    )

    examples = load_hotpotqa(split="validation", limit=sample_size)
    if not examples:
        raise RuntimeError("HotpotQA returned no examples for the requested sample")

    corpus = build_corpus(examples)
    if not corpus.chunks:
        raise RuntimeError("The HotpotQA sample produced no corpus chunks")

    vectors = embedder.embed_texts([chunk.text for chunk in corpus.chunks])
    if len(vectors) != len(corpus.chunks):
        raise RuntimeError("Embedder returned a vector count that does not match chunks")

    vector_store = FaissVectorStore(dimension=embedder.dimension)
    vector_store.add(corpus.chunks, vectors)
    if vector_store.size != len(corpus.chunks):
        raise RuntimeError("FAISS index size does not match corpus chunk count")

    retriever = EmbeddingVectorStoreRetriever(
        embedder,
        vector_store,
        RetrieverConfig(default_top_k=top_k),
    )
    context_builder = RetrievalContextBuilder(
        ContextBuilderConfig(max_chunks=top_k)
    )
    generator = LocalTransformersGenerator(
        LocalGeneratorConfig(
            model_name_or_path=generator_model,
            local_files_only=not allow_model_download,
        )
    )
    agent = RAGAgent(retriever, context_builder, generator)
    results = tuple(
        agent.run(RAGAgentRequest(query=example.question, top_k=top_k))
        for example in examples[:question_count]
    )
    return DemoRun(
        sample_size=len(examples),
        indexed_chunk_count=vector_store.size,
        results=results,
    )


def main() -> None:
    """Run the demo and print answers with retrieval/context provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=3)
    parser.add_argument("--question-count", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--generator-model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Local model ID or path; model loading is cache-only by default.",
    )
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Allow Transformers to download the local generator model.",
    )
    args = parser.parse_args()

    run = run_demo(
        sample_size=args.sample_size,
        question_count=args.question_count,
        top_k=args.top_k,
        generator_model=args.generator_model,
        allow_model_download=args.allow_model_download,
    )
    print(f"HotpotQA sample size: {run.sample_size}")
    print(f"FAISS indexed chunks: {run.indexed_chunk_count}")
    for result in run.results:
        print(f"\nQuestion: {result.query}")
        print(f"Answer: {result.answer}")
        print("Retrieved chunks:")
        for retrieved in result.retrieved_results:
            print(
                f"  rank={retrieved.rank} score={retrieved.score:.4f} "
                f"id={retrieved.chunk.chunk_id!r} "
                f"title={retrieved.chunk.source_title!r}"
            )
        print("Generation context:")
        for item in result.context.items:
            print(
                f"  evidence={item.evidence_type} id={item.chunk_id!r} "
                f"title={item.source_title!r} "
                f"sentences={item.source_sentence_indices}"
            )
        print(f"Execution timings: {result.execution}")


if __name__ == "__main__":
    main()
