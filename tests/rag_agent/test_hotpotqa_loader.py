"""Tests for the standalone HotpotQA dataset loader."""

import sys
import types
import unittest
from unittest.mock import Mock, patch

from rag_agent.dataset.hotpotqa_loader import (
    DATASET_PATH,
    HotpotQADataError,
    load_hotpotqa,
)


def _example(example_id: str) -> dict[str, object]:
    return {
        "id": example_id,
        "question": "Which team did the player join?",
        "answer": "Example FC",
        "type": "bridge",
        "level": "easy",
        "supporting_facts": {
            "title": ["Player", "Example FC"],
            "sent_id": [0, 1],
        },
        "context": {
            "title": ["Player", "Example FC"],
            "sentences": [
                ["The player joined Example FC."],
                ["Example FC is based in Example City."],
            ],
        },
    }


class HotpotQALoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.load_dataset = Mock(return_value=[_example("case-1")])
        datasets_module = types.ModuleType("datasets")
        datasets_module.load_dataset = self.load_dataset
        self.datasets_patch = patch.dict(sys.modules, {"datasets": datasets_module})
        self.datasets_patch.start()
        self.addCleanup(self.datasets_patch.stop)

    def test_loads_and_normalizes_dataset_example(self) -> None:
        examples = load_hotpotqa()

        self.load_dataset.assert_called_once_with(
            DATASET_PATH,
            name="fullwiki",
            split="train",
        )
        self.assertEqual(len(examples), 1)
        example = examples[0]
        self.assertEqual(example.example_id, "case-1")
        self.assertEqual(example.question, "Which team did the player join?")
        self.assertEqual(example.answer, "Example FC")
        self.assertEqual(
            example.context,
            (
                ("Player", ("The player joined Example FC.",)),
                ("Example FC", ("Example FC is based in Example City.",)),
            ),
        )
        self.assertEqual(
            example.supporting_facts,
            (("Player", 0), ("Example FC", 1)),
        )
        self.assertEqual(example.metadata, {"type": "bridge", "level": "easy"})

    def test_selects_requested_split(self) -> None:
        examples = load_hotpotqa(split="test")

        self.load_dataset.assert_called_once_with(
            DATASET_PATH,
            name="fullwiki",
            split="test",
        )
        self.assertEqual(examples[0].example_id, "case-1")

    def test_selects_config_and_limits_examples(self) -> None:
        self.load_dataset.return_value = [_example("case-1"), _example("case-2")]

        examples = load_hotpotqa(split="validation", limit=1, config_name="distractor")

        self.load_dataset.assert_called_once_with(
            DATASET_PATH,
            name="distractor",
            split="validation[:1]",
        )
        self.assertEqual([example.example_id for example in examples], ["case-1"])

    def test_accepts_record_oriented_nested_features(self) -> None:
        row = _example("case-1")
        row["context"] = [
            {"title": "Player", "sentences": ["The player joined Example FC."]}
        ]
        row["supporting_facts"] = [{"title": "Player", "sent_id": 0}]
        self.load_dataset.return_value = [row]

        example = load_hotpotqa()[0]

        self.assertEqual(
            example.context,
            (("Player", ("The player joined Example FC.",)),),
        )
        self.assertEqual(example.supporting_facts, (("Player", 0),))

    def test_rejects_malformed_required_fields_with_example_index(self) -> None:
        row = _example("case-1")
        row["supporting_facts"] = {
            "title": ["Player"],
            "sent_id": [],
        }
        self.load_dataset.return_value = [row]

        with self.assertRaisesRegex(HotpotQADataError, "Example 0"):
            load_hotpotqa()

    def test_zero_limit_does_not_load_dataset(self) -> None:
        self.assertEqual(load_hotpotqa(limit=0), [])
        self.load_dataset.assert_not_called()


if __name__ == "__main__":
    unittest.main()
