"""Load and normalize examples from the Hugging Face HotpotQA dataset."""

from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any

from rag_agent.models import HotpotQAExample


DATASET_PATH = "hotpotqa/hotpot_qa"
DEFAULT_CONFIG = "fullwiki"


class HotpotQADataError(ValueError):
    """Raised when a HotpotQA example cannot be normalized without data loss."""


def load_hotpotqa(
    split: str = "train",
    limit: int | None = None,
    config_name: str = DEFAULT_CONFIG,
) -> list[HotpotQAExample]:
    """Load and normalize a HotpotQA split from Hugging Face.

    Args:
        split: Dataset split to load, such as ``train``, ``validation``, or
            ``test``. The ``test`` split is available with the ``fullwiki``
            config.
        limit: Optional maximum number of examples. Hugging Face split slicing
            is used to avoid requesting the remainder of the split.
        config_name: HotpotQA dataset config, either ``fullwiki`` or
            ``distractor``.

    Returns:
        Normalized examples in their original dataset order.

    Raises:
        ImportError: If the optional ``datasets`` package is not installed.
        ValueError: If the split, limit, or config name is invalid.
        HotpotQADataError: If a returned example is missing or has malformed
            required data.
    """
    if not isinstance(split, str) or not split.strip():
        raise ValueError("split must be a non-empty string")
    if not isinstance(config_name, str) or config_name not in {
        "fullwiki",
        "distractor",
    }:
        raise ValueError("config_name must be 'fullwiki' or 'distractor'")
    if limit is not None and (
        isinstance(limit, bool) or not isinstance(limit, int) or limit < 0
    ):
        raise ValueError("limit must be a non-negative integer or None")
    if limit == 0:
        return []

    try:
        from datasets import load_dataset
    except ImportError as error:
        raise ImportError(
            "Loading HotpotQA requires the optional 'datasets' package. "
            "Install it with: pip install datasets"
        ) from error

    split = split.strip()
    selected_split = f"{split}[:{limit}]" if limit is not None else split
    dataset = load_dataset(
        DATASET_PATH,
        name=config_name,
        split=selected_split,
    )
    return [
        _normalize_example(raw_example, index)
        for index, raw_example in _limited_rows(dataset, limit)
    ]


def _limited_rows(
    dataset: Iterable[Any],
    limit: int | None,
) -> Iterator[tuple[int, Any]]:
    """Yield at most ``limit`` rows, preserving their source indices."""
    for index, example in enumerate(dataset):
        if limit is not None and index >= limit:
            break
        yield index, example


def _normalize_example(raw_example: Any, index: int) -> HotpotQAExample:
    if not isinstance(raw_example, Mapping):
        raise HotpotQADataError(f"Example {index} must be a mapping")

    example_id = _required_text(raw_example, "id", index, non_empty=True)
    question = _required_text(raw_example, "question", index, non_empty=True)
    answer = _required_text(raw_example, "answer", index)

    context_rows = _paired_records(
        raw_example.get("context"),
        "context",
        ("title", "sentences"),
        index,
    )
    context: list[tuple[str, tuple[str, ...]]] = []
    for document_index, (title, sentences) in enumerate(context_rows):
        if not isinstance(title, str) or not title.strip():
            raise HotpotQADataError(
                f"Example {index} context item {document_index} has an invalid title"
            )
        if not _is_sequence(sentences) or any(
            not isinstance(sentence, str) for sentence in sentences
        ):
            raise HotpotQADataError(
                f"Example {index} context item {document_index} "
                "sentences must be a sequence of strings"
            )
        context.append((title, tuple(sentences)))

    supporting_rows = _paired_records(
        raw_example.get("supporting_facts"),
        "supporting_facts",
        ("title", "sent_id"),
        index,
    )
    supporting_facts: list[tuple[str, int]] = []
    for fact_index, (title, sentence_index) in enumerate(supporting_rows):
        if not isinstance(title, str) or not title.strip():
            raise HotpotQADataError(
                f"Example {index} supporting fact {fact_index} has an invalid title"
            )
        if (
            isinstance(sentence_index, bool)
            or not isinstance(sentence_index, int)
            or sentence_index < 0
        ):
            raise HotpotQADataError(
                f"Example {index} supporting fact {fact_index} "
                "sent_id must be a non-negative integer"
            )
        supporting_facts.append((title, sentence_index))

    metadata: dict[str, str] = {}
    for key in ("type", "level"):
        value = raw_example.get(key)
        if value is not None:
            if not isinstance(value, str):
                raise HotpotQADataError(
                    f"Example {index} metadata field {key!r} must be a string"
                )
            metadata[key] = value

    return HotpotQAExample(
        example_id=example_id,
        question=question,
        answer=answer,
        context=tuple(context),
        supporting_facts=tuple(supporting_facts),
        metadata=metadata,
    )


def _required_text(
    example: Mapping[str, Any],
    field_name: str,
    index: int,
    *,
    non_empty: bool = False,
) -> str:
    value = example.get(field_name)
    if not isinstance(value, str) or (non_empty and not value.strip()):
        raise HotpotQADataError(
            f"Example {index} field {field_name!r} must be "
            f"{'a non-empty ' if non_empty else 'a '}string"
        )
    return value


def _paired_records(
    value: Any,
    field_name: str,
    pair_fields: tuple[str, str],
    example_index: int,
) -> list[tuple[Any, Any]]:
    if isinstance(value, Mapping):
        first_values = value.get(pair_fields[0])
        second_values = value.get(pair_fields[1])
        if not _is_sequence(first_values) or not _is_sequence(second_values):
            raise HotpotQADataError(
                f"Example {example_index} field {field_name!r} must contain "
                f"sequences for {pair_fields[0]!r} and {pair_fields[1]!r}"
            )
        if len(first_values) != len(second_values):
            raise HotpotQADataError(
                f"Example {example_index} field {field_name!r} has mismatched "
                "column lengths"
            )
        return list(zip(first_values, second_values))

    if not _is_sequence(value):
        raise HotpotQADataError(
            f"Example {example_index} field {field_name!r} must be a sequence"
        )
    records: list[tuple[Any, Any]] = []
    for record_index, record in enumerate(value):
        if not isinstance(record, Mapping) or any(
            name not in record for name in pair_fields
        ):
            raise HotpotQADataError(
                f"Example {example_index} field {field_name!r} item "
                f"{record_index} must contain {pair_fields[0]!r} and "
                f"{pair_fields[1]!r}"
            )
        records.append((record[pair_fields[0]], record[pair_fields[1]]))
    return records


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))