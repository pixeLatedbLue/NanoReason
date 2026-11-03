import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from run_matched_eval_all import build_commands, selected_arms  # noqa: E402


class BuildCommandsTests(unittest.TestCase):
    """A wrong flag here costs a multi-hour GPU session, so the command list is
    asserted rather than eyeballed."""

    KW = dict(
        config="configs/local_inference.toml",
        source="artifacts/slm/data/gsm8k-test.jsonl",
        output="artifacts/slm/evaluation-full",
        samples=1319,
        seed=42,
        max_new_tokens=512,
        sft_adapter="/kaggle/input/x/artifacts/sft-final",
        grpo_adapter="/kaggle/input/x/artifacts/grpo-final",
    )

    def test_base_runs_first_and_carries_no_adapter(self):
        commands = build_commands(**self.KW)
        self.assertEqual([c[c.index("--arm") + 1] for c in commands], ["base", "sft", "grpo"])
        self.assertNotIn("--adapter", commands[0])

    def test_trained_arms_each_get_their_own_adapter(self):
        sft, grpo = build_commands(**self.KW)[1:]
        self.assertEqual(sft[sft.index("--adapter") + 1], self.KW["sft_adapter"])
        self.assertEqual(grpo[grpo.index("--adapter") + 1], self.KW["grpo_adapter"])

    def test_every_arm_shares_sample_protocol_so_pairing_is_possible(self):
        for command in build_commands(**self.KW):
            self.assertEqual(command[command.index("--samples") + 1], "1319")
            self.assertEqual(command[command.index("--seed") + 1], "42")
            self.assertEqual(command[command.index("--output") + 1], self.KW["output"])
            self.assertEqual(command[command.index("--source") + 1], self.KW["source"])
            self.assertEqual(command[command.index("--config") + 1], self.KW["config"])
            self.assertEqual(command[command.index("--max-new-tokens") + 1], "512")

    def test_each_arm_is_a_separate_process_so_vram_is_released(self):
        for command in build_commands(**self.KW):
            self.assertEqual(command[:3], [sys.executable, "-m", "nanoreason.matched_eval"])

    def test_one_arm_can_be_run_alone(self):
        """A T4 cannot finish all three arms inside one Kaggle session.

        Measured on an RTX 4060 the three arms cost 19.8 seconds per item
        together; 1319 items is about 7 hours there and a T4 is roughly half
        that speed. Arms therefore run one per session, which pairs correctly
        because environment_info records only library versions and the OS,
        and the notebook pins those.
        """
        commands = build_commands(**self.KW, arms=("grpo",))
        self.assertEqual([c[c.index("--arm") + 1] for c in commands], ["grpo"])

    def test_selected_arms_keeps_base_first_and_rejects_unknown_names(self):
        self.assertEqual(selected_arms(None), ("base", "sft", "grpo"))
        self.assertEqual(selected_arms("grpo,base"), ("base", "grpo"))
        with self.assertRaisesRegex(ValueError, "nonsense"):
            selected_arms("nonsense")

    def test_missing_adapter_for_a_trained_arm_is_rejected(self):
        bad = dict(self.KW, grpo_adapter=None)
        with self.assertRaisesRegex(ValueError, "grpo"):
            build_commands(**bad)

    def test_an_adapter_is_only_required_for_an_arm_being_run(self):
        """Session 1 runs base alone and has no reason to hold either adapter."""
        bad = dict(self.KW, sft_adapter=None, grpo_adapter=None)
        self.assertEqual(len(build_commands(**bad, arms=("base",))), 1)


if __name__ == "__main__":
    unittest.main()
