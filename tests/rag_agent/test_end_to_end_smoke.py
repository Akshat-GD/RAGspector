"""Opt-in real HotpotQA/model end-to-end smoke test.

Prerequisites: install ``datasets``, ``sentence-transformers``, and
``faiss-cpu``; install ``torch`` and ``transformers``; allow access to the
HotpotQA dataset and the default embedding model. The default local generator
must already be cached, or set up an explicitly download-enabled invocation.
Run with ``$env:RAG_AGENT_RUN_REAL_SMOKE='1'; python -m pytest
tests\\rag_agent\\test_end_to_end_smoke.py -q`` on PowerShell.
"""

import os

import pytest

from rag_agent.demo import run_demo
from rag_agent.embeddings import DEFAULT_MODEL_NAME
from rag_agent.corpus.models import CorpusChunk


pytestmark = pytest.mark.skipif(
    os.environ.get("RAG_AGENT_RUN_REAL_SMOKE") != "1",
    reason=(
        "opt-in real smoke test; set RAG_AGENT_RUN_REAL_SMOKE=1 after "
        "installing prerequisites and caching required models"
    ),
)


def test_real_hotpotqa_to_local_generation_pipeline() -> None:
    run = run_demo(sample_size=2, question_count=2, top_k=2)

    assert run.sample_size == 2
    assert run.indexed_chunk_count > 0
    assert len(run.results) == 2
    for result in run.results:
        assert result.query
        assert result.answer
        assert result.generation.model_name == "Qwen/Qwen2.5-0.5B-Instruct"
        assert result.execution.retrieved_count > 0
        assert result.execution.context_item_count > 0
        assert result.execution.total_seconds >= 0
        assert len(result.retrieved_results) <= 2
        for rank, retrieved in enumerate(result.retrieved_results, start=1):
            assert retrieved.rank == rank
            assert retrieved.score is not None
            assert isinstance(retrieved.chunk, CorpusChunk)
            assert retrieved.chunk.chunk_id
        assert all(
            item.evidence_type == "retrieved" for item in result.context.items
        )
        assert all(item.retrieval_score is not None for item in result.context.items)
        assert DEFAULT_MODEL_NAME == "sentence-transformers/all-MiniLM-L6-v2"
