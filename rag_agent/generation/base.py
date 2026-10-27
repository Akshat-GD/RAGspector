"""Generator abstractions, result models, and shared input validation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Mapping

from rag_agent.context.builder import BuiltContext


class GenerationError(RuntimeError):
    """Raised when a configured generator cannot produce a valid answer."""


class GenerationDependencyError(ImportError):
    """Raised when optional local generation dependencies are missing."""


class GenerationModelLoadError(GenerationError):
    """Raised when a local model or tokenizer cannot be loaded."""


class GenerationInputError(ValueError):
    """Raised when a generation query or context is invalid."""


@dataclass(frozen=True)
class TokenUsage:
    """Token counts returned by a generation backend when available."""

    prompt_tokens: int
    completion_tokens: int

    def __post_init__(self) -> None:
        for field_name, value in (
            ("prompt_tokens", self.prompt_tokens),
            ("completion_tokens", self.completion_tokens),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")

    @property
    def total_tokens(self) -> int:
        """Return the sum of prompt and completion tokens."""
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class GenerationResult:
    """Structured output from a generator.

    ``token_usage`` is ``None`` when the backend cannot provide token counts.
    """

    answer: str
    model_name: str
    latency_seconds: float
    token_usage: TokenUsage | None = None
    metadata: Mapping[str, str | int | float | bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        if not math.isfinite(self.latency_seconds) or self.latency_seconds < 0:
            raise ValueError("latency_seconds must be finite and non-negative")


class Generator(ABC):
    """Interface for generating a structured answer from a query and context."""

    @abstractmethod
    def generate(self, query: str, context: BuiltContext) -> GenerationResult:
        """Generate an answer using only the supplied query and built context."""


def validate_generation_input(query: str, context: BuiltContext) -> None:
    """Validate shared query and built-context inputs."""
    if not isinstance(query, str) or not query.strip():
        raise GenerationInputError("query must be a non-empty string")
    if not isinstance(context, BuiltContext):
        raise GenerationInputError("context must be a BuiltContext")