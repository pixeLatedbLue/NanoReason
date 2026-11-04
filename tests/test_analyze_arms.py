import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from analyze_arms import (  # noqa: E402
    describe,
    failure_breakdown,
    load_items,
)


def item(index, correct, tokens=100, completion="2 + 2 = 4\n#### 4",
         marker=True, cap=False, prediction="4"):
    return {
        "index": index,
        "question": "q",
        "answer": "a",
        "completion": completion,
        "prediction": prediction,
        "gold": "4",
        "correct": correct,
        "generated_tokens": tokens,
        "token_cap_hit": cap,
        "has_answer_marker": marker,
        "latency_seconds": 1.0,
        "peak_allocated_bytes": 1,
    }


def write_progress(directory: Path, arm: str, items: list[dict]) -> None:
    path = directory / "progress"
    path.mkdir(parents=True, exist_ok=True)
    (path / f"{arm}.json").write_text(json.dumps({"items": items}), encoding="utf-8")


class LoadItemsTests(unittest.TestCase):
    def test_missing_arm_names_the_arm(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError) as raised:
                load_items(Path(tmp), "grpo")
            self.assertIn("grpo", str(raised.exception))

    def test_items_are_keyed_by_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            write_progress(directory, "base", [item(7, True), item(9, False)])
            loaded = load_items(directory, "base")
            self.assertEqual(sorted(loaded), [7, 9])
            self.assertTrue(loaded[7]["correct"])


class DescribeTests(unittest.TestCase):
    def test_markup_rate_counts_reference_annotation(self):
        items = {
            1: item(1, True, completion="<<2+2=4>> so 4\n#### 4"),
            2: item(2, True, completion="plain prose\n#### 4"),
        }
        summary = describe(items, [1, 2])
        self.assertAlmostEqual(summary["markup_rate"], 0.5)
        self.assertAlmostEqual(summary["accuracy"], 1.0)

    def test_median_tokens_uses_only_the_requested_keys(self):
        items = {1: item(1, True, tokens=10), 2: item(2, True, tokens=1000)}
        self.assertEqual(describe(items, [1])["median_tokens"], 10)

    def test_empty_selection_does_not_divide_by_zero(self):
        self.assertEqual(describe({}, [])["accuracy"], 0.0)


class FailureBreakdownTests(unittest.TestCase):
    def test_separates_malformed_from_confidently_wrong(self):
        items = {
            1: item(1, False),                                   # parsed, wrong
            2: item(2, False, marker=False, prediction=""),      # no marker
            3: item(3, False, cap=True),                         # truncated
            4: item(4, True),                                    # correct
        }
        counts = failure_breakdown(items, [1, 2, 3, 4])
        self.assertEqual(counts["wrong"], 3)
        self.assertEqual(counts["no_answer_marker"], 1)
        self.assertEqual(counts["token_cap_hit"], 1)
        self.assertEqual(counts["parsed_but_wrong"], 2)

if __name__ == "__main__":
    unittest.main()
