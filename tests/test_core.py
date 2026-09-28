import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nanoreason.config import load_config
from nanoreason.data import aqua_to_qa, load_training_dataset, reasoning_difficulty
from nanoreason.metrics import (
    extract_final_letter,
    extract_final_number,
    gold_number,
    numeric_equal,
    token_diversity,
    verify_arithmetic_steps,
)
from nanoreason.rewards import (
    ModelProcessReward,
    build_reward_funcs,
    diversity_reward,
    format_reward,
    outcome_reward,
    process_reward,
)

HAS_TRL = importlib.util.find_spec("trl") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_DATASETS = importlib.util.find_spec("datasets") is not None


class ConfigTests(unittest.TestCase):
    def test_all_shipped_configs_load(self):
        for toml in sorted((ROOT / "configs").glob("*.toml")):
            with self.subTest(config=toml.name):
                load_config(toml)

    def test_default_and_smoke_configs_load(self):
        default = load_config(ROOT / "configs" / "default.toml")
        smoke = load_config(ROOT / "configs" / "smoke.toml")

        self.assertTrue(default.model.load_in_4bit)
        self.assertFalse(smoke.model.load_in_4bit)
        self.assertEqual(sorted(default.evaluation.tasks), ["aqua", "gsm8k", "mmlu", "strategyqa"])
        self.assertEqual(default.sft.lora_r, 64)
        self.assertTrue(default.sft.completion_only)
        self.assertTrue(default.grpo.curriculum)
        self.assertTrue(default.grpo.use_process_reward)

    def test_kaggle_and_8gb_configs(self):
        kaggle = load_config(ROOT / "configs" / "kaggle_t4.toml")
        self.assertEqual(sorted(kaggle.evaluation.tasks), ["aqua", "gsm8k", "mmlu", "strategyqa"])
        self.assertEqual(kaggle.sft.optim, "paged_adamw_8bit")
        self.assertEqual(kaggle.grpo.num_generations, 2)
        self.assertEqual(kaggle.grpo.train_split, "train[:1500]")
        self.assertTrue(kaggle.grpo.gradient_checkpointing)

        rtx = load_config(ROOT / "configs" / "rtx4060_8gb.toml")
        self.assertEqual(sorted(rtx.evaluation.tasks), ["aqua", "gsm8k", "mmlu", "strategyqa"])
        self.assertEqual(rtx.grpo.num_generations, 2)

    def test_unknown_key_names_file_and_section(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.toml"
            bad.write_text("[sft]\nlora_rr = 64\n", encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                load_config(bad)
            self.assertIn("lora_rr", str(ctx.exception))
            self.assertIn("[sft]", str(ctx.exception))

    def test_validation_rejects_degenerate_hyperparams(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.toml"
            bad.write_text("[grpo]\nnum_generations = 1\n", encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                load_config(bad)
            self.assertIn("num_generations", str(ctx.exception))

            bad.write_text("[sft]\nvalidation_fraction = 0.0\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(bad)


class MetricTests(unittest.TestCase):
    def test_number_extraction_handles_markers_commas_and_fallback(self):
        self.assertEqual(extract_final_number("answer #### 1,234.50"), "1234.50")
        self.assertEqual(extract_final_number("No marker, last number is 17"), "17")
        self.assertIsNone(extract_final_number("No numbers here"))

    def test_number_extraction_uses_last_marker(self):
        self.assertEqual(extract_final_number("#### 10 ... wait, no. #### 72"), "72")

    def test_numeric_comparison_and_gold_extraction(self):
        self.assertTrue(numeric_equal("3.0", "3"))
        self.assertFalse(numeric_equal("3.1", "3"))
        self.assertEqual(gold_number("reasoning\n#### 42"), "42")

    def test_numeric_equal_edges(self):
        self.assertTrue(numeric_equal("1e3", "1000"))
        self.assertFalse(numeric_equal("nan", "nan"))
        self.assertFalse(numeric_equal(None, "3"))
        self.assertFalse(numeric_equal("abc", "3"))
        self.assertTrue(numeric_equal("1000000.000001", "1000000"))
        self.assertFalse(numeric_equal("0.000001", "0"))

    def test_letter_extraction(self):
        self.assertEqual(extract_final_letter("steps\n#### C"), "C")
        self.assertEqual(extract_final_letter("the answer is b"), "B")
        self.assertIsNone(extract_final_letter("12345"))

    def test_letter_extraction_ignores_prose_articles(self):
        self.assertIsNone(extract_final_letter("It is a big number"))
        self.assertIsNone(extract_final_letter("A cat sat on the mat"))
        self.assertIsNone(extract_final_letter("The total is 240 km, quite a distance."))
        self.assertIsNone(extract_final_letter("We need to answer a question about trains"))

    def test_letter_extraction_confident_forms(self):
        self.assertEqual(extract_final_letter("#### 10 no wait #### D"), "D")
        self.assertEqual(extract_final_letter("So the option is (c)"), "C")
        self.assertEqual(extract_final_letter("steps...\nB)"), "B")
        self.assertEqual(extract_final_letter("steps...\nB."), "B")

    def test_arithmetic_verification_counts_only_correct_steps(self):
        good = "5 + 7 = 12\n12 * 2 = 24"
        self.assertEqual(verify_arithmetic_steps(good), (2, 2))
        mixed = "5 + 7 = 99\n2 * 3 = 6"
        self.assertEqual(verify_arithmetic_steps(mixed), (1, 2))
        self.assertEqual(verify_arithmetic_steps("no math here"), (0, 0))

    def test_arithmetic_verification_handles_commas(self):
        self.assertEqual(verify_arithmetic_steps("1,000 + 500 = 1,500"), (1, 1))
        self.assertEqual(verify_arithmetic_steps("2,500 - 1,200 = 1,300"), (1, 1))

    def test_arithmetic_verification_boundaries(self):
        self.assertEqual(verify_arithmetic_steps("5 - -3 = 8"), (1, 1))
        self.assertEqual(verify_arithmetic_steps("1.5e3 * 2 = 3000"), (0, 0))
        self.assertEqual(verify_arithmetic_steps("5 / 0 = 1"), (0, 1))
        self.assertEqual(verify_arithmetic_steps("5 + 7 = 12 * 2 = 24"), (1, 1))
        self.assertEqual(verify_arithmetic_steps("1 + 1 = 2e9"), (0, 0))
        self.assertEqual(verify_arithmetic_steps("1 + 1 = 2foo"), (0, 0))
        self.assertEqual(verify_arithmetic_steps("x1 + 1 = 2"), (0, 0))
        self.assertEqual(verify_arithmetic_steps("1 + 1 = 2.5"), (0, 1))

    def test_arithmetic_verification_accepts_honest_rounding(self):
        self.assertEqual(verify_arithmetic_steps("10 / 3 = 3.33"), (1, 1))
        self.assertEqual(verify_arithmetic_steps("10 / 3 = 3.4"), (0, 1))

    def test_arithmetic_verification_dedupe(self):
        spam = "1 + 1 = 2\n" * 10
        self.assertEqual(verify_arithmetic_steps(spam, dedupe=True), (1, 1))
        self.assertEqual(verify_arithmetic_steps(spam, dedupe=False), (10, 10))

    def test_token_diversity(self):
        self.assertEqual(token_diversity("a a a a"), 0.25)
        self.assertEqual(token_diversity(""), 0.0)


class RewardTests(unittest.TestCase):
    def test_outcome_and_format_rewards(self):
        completion = "Step 1\n1 + 1 = 2\n#### 2"
        self.assertEqual(outcome_reward([], [completion], ["#### 2"]), [2.0])
        self.assertEqual(format_reward([], [completion]), [0.5])

    def test_format_reward_edges(self):
        self.assertEqual(format_reward([], ["#### 42  \n"]), [0.5])
        self.assertEqual(format_reward([], ["#### 42."]), [0.5])
        self.assertEqual(format_reward([], ["#### $42"]), [0.5])
        self.assertEqual(format_reward([], ["#### 42\nHope this helps"]), [0.0])
        self.assertEqual(format_reward([], ["Step 1\n1 + 1 = 2"]), [0.0])

    def test_process_reward_is_hack_proof(self):
        honest = "1 + 1 = 2\n2 + 2 = 4\n3 + 3 = 6\n#### 6"
        faked = "1 + 1 = 7\n2 + 2 = 9\n#### 4"
        self.assertEqual(process_reward([], [honest]), [1.0])
        self.assertLess(process_reward([], [faked])[0], 0.0)
        self.assertEqual(process_reward([], ["no equations"]), [0.0])

    def test_process_reward_resists_trivial_and_spam_hacks(self):
        trivial = process_reward([], ["1 + 1 = 2\n#### 0"])[0]
        self.assertLessEqual(trivial, 0.34)
        spam = process_reward([], ["1 + 1 = 2\n" * 10 + "#### 0"])[0]
        self.assertEqual(spam, trivial)
        honest = process_reward([], ["2 + 3 = 5\n5 * 4 = 20\n20 - 1 = 19\n#### 19"])[0]
        self.assertGreater(honest, trivial)

    def test_process_reward_penalties_not_diluted(self):
        self.assertEqual(process_reward([], ["1 + 1 = 7"]), [-0.5])

    def test_substance_floor_follows_the_reference_solution(self):
        gold = ["5 + 7 = 12\n12 * 2 = 24\n#### 24"]
        honest = "5 + 7 = 12\n12 * 2 = 24\n#### 24"
        self.assertEqual(process_reward([], [honest], answer=gold), [1.0])

        padded = "1 + 1 = 2\n2 + 2 = 4\n3 + 3 = 6\n#### 24"
        self.assertEqual(
            process_reward([], [padded], answer=gold),
            process_reward([], [honest], answer=gold),
        )

        short = "5 + 7 = 12\n#### 24"
        self.assertLess(
            process_reward([], [short], answer=gold)[0],
            process_reward([], [honest], answer=gold)[0],
        )

    def test_substance_floor_falls_back_without_a_reference(self):
        trivial = "1 + 1 = 2\n#### 2"
        self.assertEqual(process_reward([], [trivial]), process_reward([], [trivial], answer=None))
        self.assertEqual(
            process_reward([], [trivial], answer=["#### 2"]),
            process_reward([], [trivial]),
        )
        self.assertLessEqual(process_reward([], [trivial])[0], 0.34)

    def test_diversity_reward_penalises_collapse(self):
        self.assertEqual(diversity_reward([], ["the cat sat on a warm mat quietly"]), [0.0])
        self.assertEqual(diversity_reward([], ["spam spam spam spam spam spam"]), [-0.5])

    def test_build_reward_funcs_toggles(self):
        full = build_reward_funcs(use_process=True, use_diversity=True)
        self.assertEqual(
            [f.__name__ for f in full],
            ["outcome_reward", "format_reward", "process_reward", "diversity_reward"],
        )
        orm_only = build_reward_funcs(use_process=False, use_diversity=False)
        self.assertEqual([f.__name__ for f in orm_only], ["outcome_reward", "format_reward"])

    def test_model_prm_is_lazy(self):
        funcs = build_reward_funcs(
            use_process=True, use_diversity=True, model_prm="dummy/never-loaded", model_prm_scale=2.0
        )
        self.assertEqual(funcs[-1].__name__, "model_process_reward")
        self.assertIsInstance(funcs[-1], ModelProcessReward)
        self.assertEqual(funcs[-1].scale, 2.0)
        self.assertIsNone(funcs[-1]._model)


class DataTests(unittest.TestCase):
    @unittest.skipUnless(HAS_DATASETS, "datasets not installed")
    def test_training_loader_passes_slice_to_huggingface(self):
        from datasets import Dataset

        sample = Dataset.from_dict({"question": ["q"], "answer": ["#### 1"]})
        with mock.patch("datasets.load_dataset", return_value=sample) as load:
            result = load_training_dataset("gsm8k", "main", "train[:1]")
        load.assert_called_once_with("gsm8k", "main", split="train[:1]")
        self.assertEqual(len(result), 1)

    def test_reasoning_difficulty_orders_by_steps(self):
        easy = "2 + 2 = 4\n#### 4"
        hard = "1 + 1 = 2\n2 + 3 = 5\n5 * 2 = 10\n#### 10"
        self.assertGreater(reasoning_difficulty(hard), reasoning_difficulty(easy))
        self.assertEqual(reasoning_difficulty(""), 0)

    def test_reasoning_difficulty_ignores_prose_hyphens(self):
        prose = "Twenty-three part-time workers - so twenty-three."
        real = "2 + 2 = 4\n4 * 3 = 12"
        self.assertLess(reasoning_difficulty(prose), reasoning_difficulty(real))

    def test_aqua_conversion_to_qa(self):
        row = {
            "question": "What is 2+2?",
            "options": ["A)3", "B)4", "C)5", "D)6", "E)7"],
            "rationale": "2 + 2 = 4",
            "correct": "B",
        }
        out = aqua_to_qa(row)
        self.assertIn("Options:", out["question"])
        self.assertTrue(out["answer"].strip().endswith("#### B"))

    def test_aqua_conversion_rejects_malformed_rows(self):
        with self.assertRaises(ValueError) as ctx:
            aqua_to_qa({"question": "q", "options": []})
        self.assertIn("correct", str(ctx.exception))
        with self.assertRaises(ValueError):
            aqua_to_qa({"options": [], "correct": "A"})

    @unittest.skipUnless(HAS_DATASETS, "datasets not installed")
    def test_order_by_curriculum(self):
        from datasets import Dataset

        from nanoreason.data import order_by_curriculum

        ds = Dataset.from_dict(
            {
                "question": ["q1", "q2", "q3", "q4"],
                "answer": [
                    "1+1=2\n2+3=5\n5*2=10\n#### 10",
                    "2+2=4\n#### 4",
                    "#### 7",
                    "3+3=6\n#### 6",
                ],
            }
        )
        out = order_by_curriculum(ds)
        difficulties = [reasoning_difficulty(a) for a in out["answer"]]
        self.assertEqual(difficulties, sorted(difficulties))
        self.assertEqual(out["question"][0], "q3")
        ds2 = Dataset.from_dict({"prompt": ["x"]})
        self.assertIs(order_by_curriculum(ds2), ds2)


class CompareResultsTests(unittest.TestCase):
    @staticmethod
    def _payload(summary):
        """An artifact shaped like the evaluator writes: accuracy is the ratio
        of the counts, never a free-standing number."""
        summary_block = {
            task: {"accuracy": acc, "correct": round(acc * 100), "total": 100}
            for task, acc in summary.items()
        }
        return {"summary": summary_block}

    def _run_main(self, argv):
        from nanoreason import compare_results

        buffer = io.StringIO()
        old_argv = sys.argv
        sys.argv = ["compare_results"] + argv
        try:
            with contextlib.redirect_stdout(buffer):
                compare_results.main()
        finally:
            sys.argv = old_argv
        return buffer.getvalue()

    def test_exact_threshold_and_na_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "base.json"
            cand = Path(tmp) / "cand.json"
            base.write_text(json.dumps(self._payload({"gsm8k": 0.45})), encoding="utf-8")
            cand.write_text(json.dumps(self._payload({"gsm8k": 0.5, "mmlu": 0.6})), encoding="utf-8")
            out = self._run_main(["--baseline", str(base), "--candidate", str(cand)])
        self.assertIn("gsm8k,0.4500,0.5000,0.0500,True", out)
        self.assertIn("mmlu,NA,0.6000,NA,NA", out)
        self.assertIn("passes_plus_5pt", out)

    def test_column_name_tracks_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "base.json"
            cand = Path(tmp) / "cand.json"
            base.write_text(json.dumps(self._payload({"gsm8k": 0.40})), encoding="utf-8")
            cand.write_text(json.dumps(self._payload({"gsm8k": 0.43})), encoding="utf-8")
            out = self._run_main(
                ["--baseline", str(base), "--candidate", str(cand), "--min-improvement", "0.02"]
            )
        self.assertIn("passes_plus_2pt", out)
        self.assertIn("gsm8k,0.4000,0.4300,0.0300,True", out)

    def test_accuracy_comes_from_the_counts_not_the_recorded_field(self):
        """The artifact stores accuracy alongside correct/total, and the CLI's
        --ci interval is built from the counts. If the two ever disagree, the
        counts are the primary record: they are what the evaluator tallied, and
        the accuracy field is only a derived convenience that a hand-edited or
        half-written file can carry stale."""
        corrupt = {"summary": {"gsm8k": {"accuracy": 0.90, "correct": 1, "total": 100}}}
        honest = {"summary": {"gsm8k": {"accuracy": 0.50, "correct": 50, "total": 100}}}
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "base.json"
            cand = Path(tmp) / "cand.json"
            base.write_text(json.dumps(corrupt), encoding="utf-8")
            cand.write_text(json.dumps(honest), encoding="utf-8")
            out = self._run_main(["--baseline", str(base), "--candidate", str(cand)])
        self.assertIn("gsm8k,0.0100,0.5000,0.4900,True", out)


@unittest.skipUnless(HAS_TRL and HAS_TORCH, "trl/torch not installed")
class TrainerShimTests(unittest.TestCase):
    def test_sft_targets_every_phi3_linear_layer(self):
        from nanoreason.train_sft import LORA_TARGETS

        self.assertEqual(LORA_TARGETS, "all-linear")

    def test_grpo_reuses_loaded_adapter_for_reference_logits(self):
        import trl.trainer.grpo_trainer as grpo_module

        from nanoreason import train_grpo

        class FakePeftModel:
            active_adapter = "default"
            peft_config = {"default": object()}

        model = FakePeftModel()
        original = grpo_module.get_peft_model
        with mock.patch.object(train_grpo, "PeftModel", FakePeftModel):
            with train_grpo._reuse_loaded_adapter(model) as config:
                self.assertIs(config, model.peft_config["default"])
                self.assertIs(grpo_module.get_peft_model(model, config), model)
        self.assertIs(grpo_module.get_peft_model, original)

    def test_grpo_config_passthrough_and_loud_drop(self):
        from nanoreason.train_grpo import _grpo_config

        cfg = _grpo_config(output_dir="x", beta=0.07, num_generations=4, temperature=0.8)
        self.assertEqual(cfg.beta, 0.07)
        self.assertEqual(cfg.num_generations, 4)
        self.assertEqual(cfg.temperature, 0.8)

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            cfg = _grpo_config(output_dir="x", not_a_real_field=1)
        self.assertIn("not_a_real_field", buffer.getvalue())

    def test_sft_config_shim_preserves_eval_strategy_and_seq_length(self):
        from nanoreason.train_sft import _sft_config

        cfg = _sft_config(output_dir="x", evaluation_strategy="steps", eval_steps=50, max_seq_length=1024)
        strategy = getattr(cfg, "eval_strategy", getattr(cfg, "evaluation_strategy", None))
        self.assertEqual(str(strategy).split(".")[-1].lower().strip("'\""), "steps")
        self.assertEqual(cfg.eval_steps, 50)
        self.assertEqual(cfg.max_seq_length, 1024)

    def test_curriculum_sampler_is_sequential(self):
        from nanoreason.train_grpo import CurriculumGRPOTrainer

        trainer = object.__new__(CurriculumGRPOTrainer)
        trainer.train_dataset = list(range(5))
        self.assertEqual(list(trainer._get_train_sampler()), [0, 1, 2, 3, 4])

    def test_supported_trl_version(self):
        import trl

        from nanoreason.train_grpo import _require_supported_trl

        major_minor = tuple(int(p) for p in trl.__version__.split(".")[:2])
        if major_minor < (0, 15):
            _require_supported_trl()
        else:
            with self.assertRaises(RuntimeError):
                _require_supported_trl()


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class EvaluateTests(unittest.TestCase):
    def test_token_id_sets_with_stub_tokenizer(self):
        from nanoreason.evaluate import token_id_sets

        class StubTokenizer:
            TABLE = {
                "A": [1], " A": [2], "a": [3], " a": [4],
                "yes": [10], " yes": [11], "Yes": [12], " Yes": [13],
                "YES": [14, 15], " YES": [16, 17],
            }

            def encode(self, text, add_special_tokens=False):
                return self.TABLE.get(text, [99, 98])

        ids = token_id_sets(StubTokenizer(), ["A"])
        self.assertEqual(ids["A"], {1, 2, 3, 4})
        ids = token_id_sets(StubTokenizer(), ["yes"])
        self.assertEqual(ids["yes"], {10, 11, 12, 13})
        with self.assertRaises(ValueError):
            token_id_sets(StubTokenizer(), ["zz"])


class StatsTests(unittest.TestCase):
    @staticmethod
    def _payload(summary, per_item=None):
        per_item = per_item or {}
        return {
            "run_name": "unit",
            "summary": {
                task: {"accuracy": acc, "correct": correct, "total": total}
                for task, (acc, correct, total) in summary.items()
            },
            "results": [
                {"task": task, "accuracy": summary[task][0], "per_item": per_item.get(task)}
                for task in summary
            ],
        }

    def _run_main(self, argv):
        from nanoreason import compare_results

        buffer = io.StringIO()
        old_argv = sys.argv
        sys.argv = ["compare_results"] + argv
        try:
            with contextlib.redirect_stdout(buffer):
                compare_results.main()
        finally:
            sys.argv = old_argv
        return buffer.getvalue()

    def test_wilson_interval_brackets_the_point_estimate(self):
        from nanoreason.stats import wilson_interval

        low, high = wilson_interval(50, 100)
        self.assertLess(low, 0.5)
        self.assertGreater(high, 0.5)
        self.assertAlmostEqual(low, 0.404, places=3)
        self.assertAlmostEqual(high, 0.596, places=3)

    def test_wilson_interval_narrows_with_more_samples(self):
        from nanoreason.stats import wilson_interval

        widths = []
        for total in (100, 1000, 10000):
            low, high = wilson_interval(total // 2, total)
            widths.append(high - low)
        self.assertGreater(widths[0], widths[1])
        self.assertGreater(widths[1], widths[2])

    def test_wilson_interval_degenerate_cases(self):
        from nanoreason.stats import wilson_interval

        self.assertEqual(wilson_interval(0, 0), (0.0, 0.0))
        low, high = wilson_interval(20, 20)
        self.assertEqual(high, 1.0)
        self.assertLess(low, 1.0)
        low, high = wilson_interval(0, 50)
        self.assertEqual(low, 0.0)
        self.assertGreater(high, 0.0)

    def test_mcnemar_exact_known_values(self):
        import math

        from nanoreason.stats import mcnemar_exact

        self.assertEqual(mcnemar_exact(0, 0), 1.0)
        self.assertLess(mcnemar_exact(10, 0), 0.05)
        self.assertAlmostEqual(mcnemar_exact(10, 0), 2 / 1024)
        self.assertEqual(mcnemar_exact(5, 5), 1.0)
        big = mcnemar_exact(600, 600)
        self.assertTrue(math.isfinite(big))
        self.assertGreaterEqual(big, 0.0)
        self.assertLessEqual(big, 1.0)
        self.assertLess(mcnemar_exact(700, 600), 0.05)

    def test_mcnemar_exact_is_symmetric(self):
        from nanoreason.stats import mcnemar_exact

        for a, b in ((10, 0), (7, 3), (1, 4), (0, 6), (12, 11)):
            self.assertEqual(mcnemar_exact(a, b), mcnemar_exact(b, a))

    def test_paired_comparison_counts_discordant_pairs(self):
        from nanoreason.stats import paired_comparison

        base = [
            {"index": 0, "correct": True},
            {"index": 1, "correct": False},
            {"index": 2, "correct": False},
            {"index": 3, "correct": True},
        ]
        cand = [
            {"index": 0, "correct": True},
            {"index": 1, "correct": True},
            {"index": 2, "correct": True},
            {"index": 3, "correct": False},
        ]
        result = paired_comparison(base, cand)
        self.assertEqual(result["n_both"], 4)
        self.assertEqual(result["only_candidate"], 2)
        self.assertEqual(result["only_baseline"], 1)
        self.assertEqual(result["p_value"], 1.0)
        self.assertFalse(result["significant"])

    def test_paired_comparison_detects_a_real_win(self):
        from nanoreason.stats import paired_comparison

        base = [{"index": i, "correct": False} for i in range(12)]
        cand = [{"index": i, "correct": i < 10} for i in range(12)]
        result = paired_comparison(base, cand)
        self.assertEqual((result["only_candidate"], result["only_baseline"]), (10, 0))
        self.assertTrue(result["significant"])

    def test_paired_comparison_returns_none_without_pairs(self):
        from nanoreason.stats import paired_comparison

        items = [{"index": 0, "correct": True}]
        self.assertIsNone(paired_comparison(None, items))
        self.assertIsNone(paired_comparison(items, None))
        self.assertIsNone(paired_comparison([], items))
        self.assertIsNone(paired_comparison(items, [{"index": 99, "correct": True}]))

    def test_load_payload_and_per_item_by_task_round_trip(self):
        from nanoreason.compare_results import load_payload, load_summary, per_item_by_task

        per_item = {"gsm8k": [{"index": 0, "correct": True}], "mmlu": None}
        payload = self._payload({"gsm8k": (0.5, 5, 10), "mmlu": (0.6, 6, 10)}, per_item)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = load_payload(path)
            self.assertEqual(loaded["run_name"], "unit")
            self.assertEqual(load_summary(path), loaded["summary"])
            by_task = per_item_by_task(loaded)
        self.assertEqual(by_task["gsm8k"], [{"index": 0, "correct": True}])
        self.assertIsNone(by_task["mmlu"])
        self.assertEqual(per_item_by_task({}), {})

    def test_default_output_is_unchanged_by_the_new_flags(self):
        base = self._payload({"gsm8k": (0.45, 45, 100)})
        cand = self._payload({"gsm8k": (0.50, 50, 100), "mmlu": (0.60, 60, 100)})
        with tempfile.TemporaryDirectory() as tmp:
            base_path = Path(tmp) / "base.json"
            cand_path = Path(tmp) / "cand.json"
            base_path.write_text(json.dumps(base), encoding="utf-8")
            cand_path.write_text(json.dumps(cand), encoding="utf-8")
            out = self._run_main(["--baseline", str(base_path), "--candidate", str(cand_path)])
        expected = (
            "task,baseline,candidate,delta,passes_plus_5pt\n"
            "gsm8k,0.4500,0.5000,0.0500,True\n"
            "mmlu,NA,0.6000,NA,NA\n"
        )
        self.assertEqual(out, expected)

    def test_ci_and_paired_flags_append_columns_and_keep_rows_rectangular(self):
        per_item = {"gsm8k": [{"index": i, "correct": i < 45} for i in range(100)]}
        base = self._payload({"gsm8k": (0.45, 45, 100)}, per_item)
        cand_items = {"gsm8k": [{"index": i, "correct": i < 50} for i in range(100)]}
        cand = self._payload({"gsm8k": (0.50, 50, 100), "mmlu": (0.60, 60, 100)}, cand_items)
        with tempfile.TemporaryDirectory() as tmp:
            base_path = Path(tmp) / "base.json"
            cand_path = Path(tmp) / "cand.json"
            base_path.write_text(json.dumps(base), encoding="utf-8")
            cand_path.write_text(json.dumps(cand), encoding="utf-8")
            argv = ["--baseline", str(base_path), "--candidate", str(cand_path)]
            ci_out = self._run_main(argv + ["--ci"])
            both_out = self._run_main(argv + ["--ci", "--paired"])
        ci_rows = [row.split(",") for row in ci_out.strip().splitlines()]
        self.assertEqual(ci_rows[0][5:], ["baseline_ci_low", "baseline_ci_high",
                                          "candidate_ci_low", "candidate_ci_high"])
        self.assertEqual(ci_rows[1][5:], ["0.3561", "0.5476", "0.4038", "0.5962"])
        both_rows = [row.split(",") for row in both_out.strip().splitlines()]
        self.assertEqual(both_rows[0][9:], ["n_both", "only_candidate", "only_baseline",
                                            "p_value", "significant"])
        self.assertEqual(both_rows[1][4], "True")
        self.assertEqual(both_rows[1][9:12], ["100", "5", "0"])
        self.assertEqual(both_rows[1][12], "0.0625")
        self.assertEqual(both_rows[1][13], "False")
        self.assertEqual({len(row) for row in both_rows}, {14})


if __name__ == "__main__":
    unittest.main()
