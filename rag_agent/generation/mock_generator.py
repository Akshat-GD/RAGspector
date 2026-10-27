"""Deterministic, explicitly non-LLM generator for tests and development."""

from time import perf_counter

from rag_agent.context.builder import BuiltContext
from rag_agent.generation.base import (
    GenerationResult,
    Generator,
    validate_generation_input,
)


class MockGenerator(Generator):
    """Return a deterministic diagnostic string without claiming LLM output.

    The result includes the question and first context item's text verbatim.
    It is intentionally not an answer-generation simulation and reports no
    token usage.
    """

    @property
    def model_name(self) -> str:
        """Identify this implementation as a mock, not a language model."""
        return "deterministic-mock"

    def generate(self, query: str, context: BuiltContext) -> GenerationResult:
        """Build a reproducible mock output from the supplied inputs."""
        validate_generation_input(query, context)
        started = perf_counter()
        if context.items:
            evidence = context.items[0].text
            answer = (
                f"[MOCK; not an LLM] query={query!r}; "
                f"first_evidence={evidence!r}"
            )
        else:
            answer = (
                f"[MOCK; not an LLM] query={query!r}; "
                "no evidence supplied"
            )
        return GenerationResult(
            answer=answer,
            model_name=self.model_name,
            latency_seconds=perf_counter() - started,
            token_usage=None,
            metadata={
                "backend": "mock",
                "context_item_count": len(context.items),
            },
        )