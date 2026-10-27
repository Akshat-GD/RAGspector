"""Pluggable text-generation abstractions and implementations."""

from rag_agent.generation.base import (
    GenerationDependencyError,
    GenerationError,
    GenerationInputError,
    GenerationModelLoadError,
    GenerationResult,
    Generator,
    TokenUsage,
)
from rag_agent.generation.local_model import (
    LocalGeneratorConfig,
    LocalTransformersGenerator,
    build_prompt,
)
from rag_agent.generation.mock_generator import MockGenerator

__all__ = [
    "GenerationDependencyError",
    "GenerationError",
    "GenerationInputError",
    "GenerationModelLoadError",
    "GenerationResult",
    "Generator",
    "LocalGeneratorConfig",
    "LocalTransformersGenerator",
    "MockGenerator",
    "TokenUsage",
    "build_prompt",
]