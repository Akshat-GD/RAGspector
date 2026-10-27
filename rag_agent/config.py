from dataclasses import dataclass


@dataclass(frozen=True)
class RagAgentConfig:
    chunk_size: int = 500
    chunk_overlap: int = 50
    retrieval_top_k: int = 5
