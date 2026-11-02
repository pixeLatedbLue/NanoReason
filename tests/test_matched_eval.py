import json
import tempfile
import unittest
from pathlib import Path

from nanoreason.matched_eval import (
    adapter_label,
    atomic_json,
    compare_complete,
    prepare_sample,
    resume_items,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class AdapterLabelTests(unittest.TestCase):
    """The recorded adapter label must not carry the operator's home directory.

    Provenance integrity rests on adapter_files_sha256, not on this string, so
    the label can be repo-relative. An absolute path leaks the local folder
    layout into evidence files that are committed and published.
    """

    def test_adapter_inside_the_repository_is_recorded_relative(self):
        label = adapter_label(REPO_ROOT / "artifacts" / "slm" / "original-grpo")
        self.assertEqual(label, "artifacts/slm/original-grpo")
        self.assertNotIn("Users", label)

    def test_adapter_outside_the_repository_keeps_its_absolute_path(self):
        with tempfile.TemporaryDirectory() as directory:
            outside = Path(directory).resolve() / "elsewhere"
            self.assertEqual(adapter_label(outside), str(outside))

    def test_no_adapter_is_recorded_as_none(self):
        self.assertIsNone(adapter_label(None))


class MatchedEvaluationTests(unittest.TestCase):
    def test_sample_is_fixed_and_source_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            source.write_text("\n".join(json.dumps({"question": str(i), "answer": f"#### {i}"})
                                       for i in range(10)), encoding="utf-8")
            first = prepare_sample(source, root / "sample.json", 4, 42)
            self.assertEqual(first, prepare_sample(source, root / "sample.json", 4, 42))
            self.assertEqual(len({row["index"] for row in first["rows"]}), 4)
            with self.assertRaisesRegex(ValueError, "Existing sample differs"):
                prepare_sample(source, root / "sample.json", 4, 43)
            source.write_text(source.read_text() + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Existing sample differs"):
                prepare_sample(source, root / "sample.json", 4, 42)

    def test_resume_rejects_changed_checkpoint_and_wrong_sample_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.json"
            sample = [{"index": 3}, {"index": 7}]
            identity = {"adapter_sha256": "first"}
            atomic_json(path, {"identity": identity, "items": sample[:1]})
            self.assertEqual(resume_items(path, identity, sample), sample[:1])
            with self.assertRaisesRegex(ValueError, "checkpoint changed"):
                resume_items(path, {"adapter_sha256": "second"}, sample)
            with self.assertRaisesRegex(ValueError, "ordered prefix"):
                resume_items(path, identity, sample[::-1])

    def test_paired_comparison_requires_matching_protocol_and_prompts(self):
        provenance = dict.fromkeys(("sample_sha256", "protocol", "prompt_ids_sha256", "environment"), "same")
        base = {"provenance": provenance, "results": [{"per_item": [{"index": 3, "correct": False}]}]}
        candidate = {"provenance": dict(provenance),
                     "results": [{"per_item": [{"index": 3, "correct": True}]}]}
        self.assertEqual(compare_complete(base, candidate)["only_candidate"], 1)
        candidate["provenance"]["prompt_ids_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "prompt_ids_sha256"):
            compare_complete(base, candidate)


if __name__ == "__main__":
    unittest.main()
