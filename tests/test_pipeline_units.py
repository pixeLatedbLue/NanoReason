"""Unit tests for the pipeline modules that previously had none.

prompts, callbacks, modeling and export were exercised only indirectly, or only
on a GPU. Everything here runs on CPU with stubs standing in for the model, and
asserts on values the real code produces rather than on the stubs themselves.
"""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from nanoreason import callbacks as callbacks_mod  # noqa: E402
from nanoreason import export as export_mod  # noqa: E402
from nanoreason import modeling  # noqa: E402
from nanoreason.callbacks import ForgettingProbe  # noqa: E402
from nanoreason.config import ModelConfig  # noqa: E402
from nanoreason.prompts import (  # noqa: E402
    AQUA_SYSTEM,
    GSM8K_FEW_SHOT,
    GSM8K_SYSTEM,
    MMLU_SYSTEM,
    STRATEGYQA_FEW_SHOT,
    STRATEGYQA_SYSTEM,
    aqua_prompt,
    chat_prompt,
    gsm8k_prompt,
    mmlu_prompt,
    strategyqa_prompt,
)


class RecordingTokenizer:
    """Captures the message list handed to the chat template."""

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.calls.append((messages, tokenize, add_generation_prompt))
        return "<rendered>"


class PromptTests(unittest.TestCase):
    def test_gsm8k_few_shot_prompt_is_system_then_pairs_then_question(self):
        tok = RecordingTokenizer()
        gsm8k_prompt(tok, "How many?", few_shot=True)
        messages, tokenize, gen = tok.calls[0]
        self.assertFalse(tokenize)
        self.assertTrue(gen)
        self.assertEqual(len(messages), 1 + 2 * len(GSM8K_FEW_SHOT) + 1)
        self.assertEqual(messages[0], {"role": "system", "content": GSM8K_SYSTEM})
        roles = [m["role"] for m in messages[1:-1]]
        self.assertEqual(roles, ["user", "assistant"] * len(GSM8K_FEW_SHOT))
        self.assertEqual(messages[-1], {"role": "user", "content": "How many?"})

    def test_gsm8k_zero_shot_prompt_has_only_system_and_question(self):
        tok = RecordingTokenizer()
        gsm8k_prompt(tok, "How many?", few_shot=False)
        messages = tok.calls[0][0]
        self.assertEqual([m["role"] for m in messages], ["system", "user"])

    def test_gsm8k_few_shot_answers_end_with_the_marker_the_reward_requires(self):
        for example in GSM8K_FEW_SHOT:
            self.assertRegex(example["a"].strip().splitlines()[-1], r"^#### -?\d")

    def test_strategyqa_few_shot_answers_are_bare_yes_or_no(self):
        for example in STRATEGYQA_FEW_SHOT:
            self.assertIn(example["a"], {"yes", "no"})
        tok = RecordingTokenizer()
        strategyqa_prompt(tok, "Is water wet?")
        self.assertEqual(tok.calls[0][0][0]["content"], STRATEGYQA_SYSTEM)

    def test_aqua_few_shot_answer_ends_with_a_letter_marker(self):
        tok = RecordingTokenizer()
        aqua_prompt(tok, "Q\nOptions:\nA)1\nB)2\nC)3\nD)4\nE)5")
        messages = tok.calls[0][0]
        self.assertEqual(messages[0]["content"], AQUA_SYSTEM)
        assistant_turns = [m["content"] for m in messages if m["role"] == "assistant"]
        self.assertTrue(assistant_turns)
        for turn in assistant_turns:
            self.assertRegex(turn.strip().splitlines()[-1], r"^#### [A-E]$")

    def test_mmlu_prompt_lays_out_four_lettered_choices_in_order(self):
        tok = RecordingTokenizer()
        sample = {"question": "Capital of France?", "choices": ["Rome", "Paris", "Oslo", "Bern"]}
        mmlu_prompt(tok, sample)
        messages = tok.calls[0][0]
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertEqual(messages[0]["content"], MMLU_SYSTEM)
        self.assertEqual(
            messages[1]["content"].splitlines(),
            ["Capital of France?", "A. Rome", "B. Paris", "C. Oslo", "D. Bern"],
        )

    def test_chat_prompt_forwards_the_generation_flag(self):
        tok = RecordingTokenizer()
        chat_prompt(tok, [{"role": "user", "content": "x"}], add_generation_prompt=False)
        self.assertFalse(tok.calls[0][2])


class _FakeModel:
    def __init__(self):
        self.training = True
        self.mode_calls = []

    def eval(self):
        self.training = False
        self.mode_calls.append("eval")

    def train(self):
        self.training = True
        self.mode_calls.append("train")


def _state(step):
    return SimpleNamespace(global_step=step)


class ForgettingProbeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.accuracies = iter([0.61, 0.58, 0.55, 0.50])

        def fake_evaluator(tokenizer, model, cfg, few_shot):
            return SimpleNamespace(accuracy=next(self.accuracies))

        self.patch = mock.patch.dict(callbacks_mod.EVALUATORS, {"mmlu": fake_evaluator})
        self.patch.start()
        self.probe = ForgettingProbe(
            tokenizer=object(), task="mmlu", every_steps=100, output_dir=self.tmp.name
        )
        self.model = _FakeModel()

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def _step(self, n):
        with contextlib.redirect_stdout(io.StringIO()):
            self.probe.on_step_end(None, _state(n), None, model=self.model)

    def _end(self, n, model="default"):
        with contextlib.redirect_stdout(io.StringIO()):
            self.probe.on_train_end(
                None, _state(n), None, model=self.model if model == "default" else model
            )

    def test_fires_only_on_multiples_of_every_steps_and_never_at_step_zero(self):
        for step in (0, 50, 100, 150, 200):
            self._step(step)
        self.assertEqual([h["step"] for h in self.probe.history], [100, 200])

    def test_restores_training_mode_after_evaluating(self):
        self._step(100)
        self.assertTrue(self.model.training)
        self.assertEqual(self.model.mode_calls, ["eval", "train"])

    def test_does_not_re_enable_training_on_a_model_that_was_already_in_eval(self):
        self.model.training = False
        self._step(100)
        self.assertFalse(self.model.training)
        self.assertNotIn("train", self.model.mode_calls)

    def test_writes_the_full_history_as_json_each_time(self):
        self._step(100)
        self._step(200)
        path = Path(self.tmp.name) / "forgetting_probe.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["task"], "mmlu")
        self.assertEqual(
            payload["history"],
            [
                {"step": 100, "task": "mmlu", "accuracy": 0.61},
                {"step": 200, "task": "mmlu", "accuracy": 0.58},
            ],
        )

    def test_train_end_does_not_repeat_a_probe_already_taken_at_that_step(self):
        self._step(200)
        self._end(200)
        self.assertEqual(len(self.probe.history), 1)

    def test_train_end_probes_a_final_step_that_the_cadence_missed(self):
        self._step(200)
        self._end(237)
        self.assertEqual([h["step"] for h in self.probe.history], [200, 237])

    def test_zero_cadence_disables_the_periodic_probe(self):
        probe = ForgettingProbe(tokenizer=object(), every_steps=0, output_dir=self.tmp.name)
        with contextlib.redirect_stdout(io.StringIO()):
            probe.on_step_end(None, _state(100), None, model=self.model)
        self.assertEqual(probe.history, [])

    def test_missing_model_is_a_no_op(self):
        self._step(100)
        before = len(self.probe.history)
        with contextlib.redirect_stdout(io.StringIO()):
            self.probe.on_step_end(None, _state(200), None, model=None)
        self._end(300, model=None)
        self.assertEqual(len(self.probe.history), before)

    def test_unknown_task_is_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            ForgettingProbe(tokenizer=object(), task="not-a-task")


class ModelingTests(unittest.TestCase):
    def test_torch_dtype_accepts_aliases_case_insensitively(self):
        self.assertIs(modeling.torch_dtype("FP16"), torch.float16)
        self.assertIs(modeling.torch_dtype("float16"), torch.float16)
        self.assertIs(modeling.torch_dtype("bf16"), torch.bfloat16)
        self.assertIs(modeling.torch_dtype("BFloat16"), torch.bfloat16)
        self.assertIs(modeling.torch_dtype("fp32"), torch.float32)

    def test_torch_dtype_names_the_offender_when_rejecting(self):
        with self.assertRaises(ValueError) as ctx:
            modeling.torch_dtype("int4")
        self.assertIn("int4", str(ctx.exception))

    def test_quantization_disabled_returns_none_without_touching_bitsandbytes(self):
        with mock.patch("importlib.util.find_spec") as find_spec:
            result = modeling.quantization_config(ModelConfig(load_in_4bit=False))
        self.assertIsNone(result)
        find_spec.assert_not_called()

    def test_quantization_requires_bitsandbytes_when_enabled(self):
        with mock.patch("importlib.util.find_spec", return_value=None):
            with self.assertRaises(RuntimeError) as ctx:
                modeling.quantization_config(ModelConfig(load_in_4bit=True))
        self.assertIn("bitsandbytes", str(ctx.exception))

    @staticmethod
    def _pretend_bitsandbytes_installed(system="Linux"):
        """bitsandbytes is absent on this CPU box; the config's own constructor
        checks its version, so both our probe and that lookup are stubbed."""
        return (
            mock.patch("importlib.util.find_spec", return_value=object()),
            mock.patch("importlib.metadata.version", return_value="0.43.1"),
            mock.patch("platform.system", return_value=system),
        )

    def test_quantization_config_is_nf4_double_quant_in_the_configured_dtype(self):
        cfg = ModelConfig(load_in_4bit=True, torch_dtype="bfloat16")
        spec, version, system = self._pretend_bitsandbytes_installed()
        with spec, version, system:
            quant = modeling.quantization_config(cfg)
        self.assertTrue(quant.load_in_4bit)
        self.assertEqual(quant.bnb_4bit_quant_type, "nf4")
        self.assertTrue(quant.bnb_4bit_use_double_quant)
        self.assertIs(quant.bnb_4bit_compute_dtype, torch.bfloat16)

    def test_windows_warns_rather_than_refusing(self):
        cfg = ModelConfig(load_in_4bit=True)
        spec, version, system = self._pretend_bitsandbytes_installed(system="Windows")
        with spec, version, system:
            with self.assertWarns(UserWarning):
                quant = modeling.quantization_config(cfg)
        self.assertIsNotNone(quant)


class ExportTests(unittest.TestCase):
    def test_merge_rejects_a_missing_adapter_before_downloading_the_base_model(self):
        with mock.patch.object(export_mod.AutoModelForCausalLM, "from_pretrained") as load:
            with self.assertRaises(FileNotFoundError):
                export_mod.merge_adapter("some/base", "adapters/does-not-exist", "out")
        load.assert_not_called()

    def test_generate_decodes_only_the_newly_generated_tokens(self):
        prompt_ids = torch.tensor([[11, 12, 13]])
        new_ids = torch.tensor([21, 22])

        class FakeInputs(dict):
            def to(self, device):
                return self

        tokenizer = mock.MagicMock()
        tokenizer.eos_token_id = 0
        tokenizer.apply_chat_template.return_value = "rendered"
        tokenizer.return_value = FakeInputs(input_ids=prompt_ids)
        tokenizer.decode.return_value = "answer"

        model = mock.MagicMock()
        model.device = "cpu"
        model.generate.return_value = torch.cat([prompt_ids, new_ids.unsqueeze(0)], dim=1)

        out = export_mod.generate(tokenizer, model, "2 + 2?", max_new_tokens=8)

        self.assertEqual(out, "answer")
        decoded_ids = tokenizer.decode.call_args.args[0]
        self.assertEqual(decoded_ids.tolist(), [21, 22])
        self.assertTrue(tokenizer.decode.call_args.kwargs["skip_special_tokens"])
        self.assertFalse(model.generate.call_args.kwargs["do_sample"])
        self.assertEqual(model.generate.call_args.kwargs["max_new_tokens"], 8)


if __name__ == "__main__":
    unittest.main()
