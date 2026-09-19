"""Tests for the serving layer's truth core: analysis and the inference engines.

CPU-only, no network, no model. The point of most of these is not that the
numbers are *pleasant* but that they are the SAME numbers the GRPO trainer sees:
the UI's central claim is that it shows the real training signal, and the only
thing standing between that claim and a comfortable lie is this file.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nanoreason.metrics import iter_arithmetic_steps, verify_arithmetic_steps
from nanoreason.rewards import diversity_reward, format_reward, outcome_reward, process_reward
from nanoreason.serve import engine as engine_module
from nanoreason.serve.analysis import analyze
from nanoreason.serve.engine import (
    DemoEngine,
    EngineInfo,
    ReasoningEngine,
    TransformersEngine,
    get_engine,
    reset_engine,
)

GOOD_COT = (
    "Jane starts with 5 apples and buys 7 more.\n"
    "5 + 7 = 12\n"
    "She then triples that pile.\n"
    "12 * 3 = 36\n"
    "Finally she eats four.\n"
    "36 - 4 = 32\n"
    "#### 32"
)

FAKED_COT = (
    "First combine the piles.\n"
    "5 + 7 = 99\n"
    "Then scale it up.\n"
    "99 * 3 = 400\n"
    "#### 400"
)

PARITY_CASES = [
    "",
    "no arithmetic anywhere in this sentence",
    "5 + 7 = 12\n12 * 2 = 24",
    "5 + 7 = 99\n2 * 3 = 6",
    "1,000 + 500 = 1,500 then 2,500 - 1,200 = 1,300",
    "5 - -3 = 8",
    "1.5e3 * 2 = 3000",
    "5 / 0 = 1",
    "0 / 0 = 0\n7 / 0 = 7\n8 / 2 = 4",
    "10 / 3 = 3.33",
    "10 / 3 = 3.4",
    "5 + 7 = 12 * 2 = 24",
    "1 + 1 = 2\n" * 10,
    "3 * 3 = 9\n3 * 3 = 9\n3 * 3 = 10\n3 * 3 = 9",
]


class AnalysisTests(unittest.TestCase):
    def test_good_chain_of_thought(self):
        result = analyze(GOOD_COT, gold="32")
        self.assertEqual(result["answer"], "32")
        self.assertEqual(result["answer_source"], "marker")
        self.assertTrue(result["format_ok"])
        self.assertEqual(result["steps_total"], 3)
        self.assertEqual(result["steps_correct"], 3)
        self.assertEqual(result["steps_unique"], 3)
        self.assertTrue(result["correct"])
        self.assertTrue(all(step["ok"] for step in result["steps"]))
        self.assertEqual([step["index"] for step in result["steps"]], [0, 1, 2])

    def test_step_records_carry_the_contract_fields(self):
        step = analyze(GOOD_COT)["steps"][0]
        self.assertEqual(
            set(step),
            {"index", "lhs", "op", "rhs", "stated", "computed", "ok", "raw", "duplicate"},
        )
        self.assertEqual(step["lhs"], "5")
        self.assertEqual(step["op"], "+")
        self.assertEqual(step["rhs"], "7")
        self.assertEqual(step["stated"], "12")
        self.assertEqual(step["raw"], "5 + 7 = 12")
        self.assertFalse(step["duplicate"])

    def test_rewards_are_the_real_training_functions(self):
        for text, gold in ((GOOD_COT, "32"), (FAKED_COT, "32"), ("", "32")):
            with self.subTest(text=text[:20]):
                rewards = analyze(text, gold=gold)["rewards"]
                self.assertEqual(rewards["outcome"], outcome_reward([""], [text], answer=[gold])[0])
                self.assertEqual(rewards["format"], format_reward([""], [text])[0])
                self.assertEqual(
                    rewards["process"], process_reward([""], [text], answer=[gold])[0]
                )
                self.assertEqual(rewards["diversity"], diversity_reward([""], [text])[0])
                self.assertAlmostEqual(
                    rewards["total"],
                    rewards["outcome"] + rewards["format"] + rewards["process"]
                    + rewards["diversity"],
                    places=6,
                )

    def test_faked_chain_is_punished(self):
        result = analyze(FAKED_COT, gold="32")
        self.assertEqual(result["steps_total"], 2)
        self.assertEqual(result["steps_correct"], 0)
        self.assertLess(result["rewards"]["process"], 0.0)
        self.assertFalse(result["correct"])
        self.assertEqual(result["rewards"]["outcome"], 0.0)

    def test_empty_text(self):
        result = analyze("")
        self.assertIsNone(result["answer"])
        self.assertIsNone(result["answer_source"])
        self.assertFalse(result["format_ok"])
        self.assertEqual(result["steps"], [])
        self.assertEqual(result["steps_total"], 0)
        self.assertEqual(result["token_diversity"], 0.0)
        self.assertIsNone(result["gold"])

    def test_outcome_is_null_and_excluded_without_gold(self):
        result = analyze(GOOD_COT)
        self.assertIsNone(result["rewards"]["outcome"])
        self.assertIsNone(result["correct"])
        self.assertIsNone(result["gold"])
        self.assertAlmostEqual(
            result["rewards"]["total"],
            result["rewards"]["format"]
            + result["rewards"]["process"]
            + result["rewards"]["diversity"],
            places=6,
        )

    def test_gold_accepts_bare_number_and_full_answer_string(self):
        bare = analyze(GOOD_COT, gold="32")
        full = analyze(GOOD_COT, gold="She had a dozen, then some.\n#### 32")
        for result in (bare, full):
            self.assertEqual(result["gold"], "32")
            self.assertTrue(result["correct"])
            self.assertEqual(result["rewards"]["outcome"], 2.0)
        self.assertEqual(bare["rewards"], full["rewards"])

    def test_process_reward_scales_against_the_reference_solution(self):
        """The substance floor must come from the gold rationale, as in training.

        ``process_reward`` derives its floor from the reference solution's own
        equation count and only falls back to the constant when the gold carries
        none. A one-step completion answering a one-step problem is the case that
        tells the two apart: full credit against the rationale, scaled down to
        the three-step fallback against a bare number. If ``analyze`` ever stops
        forwarding the raw gold, the first assertion drops to 0.3333 and the UI
        starts printing a reward the trainer never gave.
        """
        completion = "5 + 7 = 12\n#### 12"
        rationale = "Jane starts with 5 apples and buys 7 more.\n5 + 7 = 12\n#### 12"
        against_rationale = analyze(completion, gold=rationale)["rewards"]["process"]
        self.assertEqual(against_rationale, process_reward([""], [completion], answer=[rationale])[0])
        self.assertEqual(against_rationale, 1.0)
        self.assertEqual(analyze(completion, gold="12")["rewards"]["process"], 0.3333)

    def test_unparseable_gold_does_not_raise(self):
        result = analyze(GOOD_COT, gold="forty two")
        self.assertFalse(result["correct"])
        self.assertEqual(result["rewards"]["outcome"], 0.0)

    def test_answer_source_falls_back_without_a_marker(self):
        result = analyze("She counted them up and reached 12 apples.")
        self.assertEqual(result["answer"], "12")
        self.assertEqual(result["answer_source"], "fallback")
        self.assertFalse(result["format_ok"])

    def test_duplicate_steps_are_flagged_not_dropped(self):
        result = analyze("2 + 2 = 4\n2 + 2 = 4\n3 * 3 = 9")
        self.assertEqual(result["steps_total"], 3)
        self.assertEqual(result["steps_unique"], 2)
        self.assertEqual([step["duplicate"] for step in result["steps"]], [False, True, False])


class StepVerifierParityTests(unittest.TestCase):
    def test_generator_matches_counting_wrapper(self):
        for text in PARITY_CASES:
            for dedupe in (False, True):
                with self.subTest(text=text[:32], dedupe=dedupe):
                    steps = list(iter_arithmetic_steps(text))
                    if dedupe:
                        steps = [step for step in steps if not step["duplicate"]]
                    expected = (sum(1 for step in steps if step["ok"]), len(steps))
                    self.assertEqual(verify_arithmetic_steps(text, dedupe=dedupe), expected)

    def test_division_by_zero_counts_as_attempted_but_never_correct(self):
        step = next(iter_arithmetic_steps("5 / 0 = 1"))
        self.assertIsNone(step["computed"])
        self.assertFalse(step["ok"])
        self.assertEqual(verify_arithmetic_steps("5 / 0 = 1"), (0, 1))


class DemoEngineTests(unittest.TestCase):
    QUESTIONS = [
        "Jane has 5 apples and buys 7 more, then triples the pile. How many?",
        "Why is the sky blue?",
        "",
        "0 zero and 100000 huge",
        "A cart holds 1,250 items at $3 each.",
    ]

    def test_deterministic(self):
        engine = DemoEngine()
        for question in self.QUESTIONS:
            with self.subTest(question=question[:24]):
                self.assertEqual(engine.solve(question), engine.solve(question))
                self.assertEqual(DemoEngine().solve(question), engine.solve(question))

    def test_solve_is_exactly_the_stream(self):
        engine = DemoEngine()
        for question in self.QUESTIONS:
            with self.subTest(question=question[:24]):
                chunks = list(engine.stream(question))
                self.assertGreater(len(chunks), 5)
                self.assertEqual("".join(chunks), engine.solve(question))

    def test_output_is_always_well_formed_and_verifiable(self):
        engine = DemoEngine()
        for question in self.QUESTIONS + ["trapme"]:
            with self.subTest(question=question[:24]):
                result = analyze(engine.solve(question))
                self.assertTrue(result["format_ok"])
                self.assertIsNotNone(result["answer"])
                self.assertEqual(result["answer_source"], "marker")
                self.assertEqual(result["steps_total"], 3)
                self.assertEqual(result["steps_unique"], 3)
                self.assertEqual(result["rewards"]["diversity"], 0.0)

    def test_clean_questions_produce_only_correct_steps(self):
        engine = DemoEngine()
        for question in self.QUESTIONS:
            with self.subTest(question=question[:24]):
                result = analyze(engine.solve(question))
                self.assertEqual(result["steps_correct"], result["steps_total"])
                self.assertEqual(result["rewards"]["process"], 1.0)

    def test_trapme_plants_exactly_one_failing_step(self):
        engine = DemoEngine()
        for question in [
            "trapme",
            "TRAPME please",
            "A TrapMe case with 5 and 7 and 3",
            "trapme with 9000 and 9000 and 9000",
        ]:
            with self.subTest(question=question):
                result = analyze(engine.solve(question))
                failed = [step for step in result["steps"] if not step["ok"]]
                self.assertEqual(len(failed), 1, failed)
                self.assertEqual(result["steps_total"], 3)
                self.assertTrue(result["format_ok"])

    def test_info_is_honest_about_being_a_demo(self):
        info = DemoEngine().info()
        self.assertIsInstance(info, EngineInfo)
        self.assertEqual(info.kind, "demo")
        self.assertTrue(info.is_demo)
        self.assertIsNone(info.model)
        self.assertIsNone(info.adapter)
        self.assertEqual(info.device, "cpu")

    def test_delay_defaults_to_zero(self):
        self.assertEqual(DemoEngine().delay, 0.0)


class TransformersEngineTests(unittest.TestCase):
    def test_construction_loads_nothing(self):
        engine = TransformersEngine("configs/default.toml", adapter=None)
        self.assertIsNone(engine._model)
        self.assertIsNone(engine._tokenizer)
        self.assertIsNone(engine._config)

    def test_info_before_load_reads_the_config_only(self):
        engine = TransformersEngine(str(ROOT / "configs" / "default.toml"))
        info = engine.info()
        self.assertEqual(info.kind, "transformers")
        self.assertFalse(info.is_demo)
        self.assertIsNotNone(info.model)
        self.assertIsNone(engine._model)

    def test_info_survives_a_missing_config(self):
        info = TransformersEngine("configs/does-not-exist.toml").info()
        self.assertIsNone(info.model)
        self.assertFalse(info.is_demo)

    def test_concurrent_first_requests_load_the_model_once(self):
        """FastAPI runs plain handlers in a threadpool, so the first burst of
        requests can all find the engine unloaded at the same moment. Each one
        that proceeds to load pulls several gigabytes; only one may."""
        import threading
        import time

        calls = []

        def slow_load(model_cfg, adapter=None):
            calls.append(threading.get_ident())
            time.sleep(0.05)
            return object(), mock.MagicMock()

        engine = TransformersEngine(str(ROOT / "configs" / "default.toml"))
        with mock.patch("nanoreason.modeling.load_causal_lm", side_effect=slow_load):
            threads = [threading.Thread(target=engine._ensure_loaded) for _ in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        self.assertEqual(len(calls), 1, f"model loaded {len(calls)} times across 8 threads")
        self.assertIsNotNone(engine._model)

    def test_closing_the_stream_early_stops_generation(self):
        """A browser tab closed mid-answer must not leave the GPU decoding the
        remaining hundreds of tokens for nobody. Starlette closes the generator
        on disconnect; that close has to reach ``model.generate``."""
        import time

        import torch

        produced = []

        class FakeInputs(dict):
            def to(self, device):
                return self

        class FakeTokenizer:
            eos_token_id = 0

            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
                return "prompt"

            def __call__(self, text, return_tensors="pt"):
                return FakeInputs(input_ids=torch.tensor([[1, 2, 3]]))

            def decode(self, ids, **kwargs):
                return "x" * len(ids)

        class FakeModel:
            device = "cpu"

            def generate(self, input_ids, max_new_tokens, streamer, stopping_criteria=None, **kw):
                for step in range(max_new_tokens):
                    if stopping_criteria is not None and stopping_criteria(input_ids, None).all():
                        break
                    produced.append(step)
                    streamer.put(torch.tensor([[step + 1]]))
                    time.sleep(0.002)
                streamer.end()

        engine = TransformersEngine(str(ROOT / "configs" / "default.toml"))
        engine._tokenizer, engine._model = FakeTokenizer(), FakeModel()

        stream = engine.stream("q", max_new_tokens=400)
        next(stream)
        stream.close()
        time.sleep(0.1)

        self.assertLess(
            len(produced), 20, f"generated {len(produced)} tokens after the client left"
        )


class EngineSelectionTests(unittest.TestCase):
    ENV_KEYS = (
        "NANOREASON_ENGINE",
        "NANOREASON_CONFIG",
        "NANOREASON_ADAPTER",
        "NANOREASON_MODEL_READY",
    )

    def setUp(self):
        reset_engine()
        patcher = mock.patch.dict(os.environ, {}, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(reset_engine)
        for key in self.ENV_KEYS:
            os.environ.pop(key, None)

    def test_defaults_to_demo(self):
        self.assertIsInstance(get_engine(), DemoEngine)

    def test_explicit_demo(self):
        os.environ["NANOREASON_ENGINE"] = "demo"
        self.assertIsInstance(get_engine(), DemoEngine)

    def test_explicit_transformers_is_still_lazy(self):
        os.environ["NANOREASON_ENGINE"] = "transformers"
        os.environ["NANOREASON_CONFIG"] = "configs/smoke.toml"
        selected = get_engine()
        self.assertIsInstance(selected, TransformersEngine)
        self.assertEqual(selected.config_path, "configs/smoke.toml")
        self.assertIsNone(selected._model)

    def test_auto_picks_transformers_when_the_model_is_declared_ready(self):
        os.environ["NANOREASON_MODEL_READY"] = "1"
        self.assertIsInstance(get_engine(), TransformersEngine)

    def test_auto_ignores_a_non_exact_ready_flag(self):
        os.environ["NANOREASON_MODEL_READY"] = "true"
        self.assertIsInstance(get_engine(), DemoEngine)

    def test_auto_ignores_an_adapter_path_that_does_not_exist(self):
        os.environ["NANOREASON_ADAPTER"] = str(ROOT / "artifacts" / "nope")
        self.assertIsInstance(get_engine(), DemoEngine)

    def test_auto_picks_transformers_for_an_existing_adapter(self):
        os.environ["NANOREASON_ADAPTER"] = str(ROOT)
        selected = get_engine()
        self.assertIsInstance(selected, TransformersEngine)
        self.assertEqual(selected.adapter, str(ROOT))

    def test_engine_is_cached_until_reset(self):
        first = get_engine()
        self.assertIs(first, get_engine())
        reset_engine()
        second = get_engine()
        self.assertIsNot(first, second)

    def test_reset_clears_the_module_global(self):
        get_engine()
        self.assertIsNotNone(engine_module._ENGINE)
        reset_engine()
        self.assertIsNone(engine_module._ENGINE)


class BaseEngineTests(unittest.TestCase):
    def test_solve_is_defined_as_the_joined_stream(self):
        class Fake(ReasoningEngine):
            def stream(self, question, max_new_tokens=384):
                yield from ["a", "b", "c"]

        self.assertEqual(Fake().solve("q"), "abc")

    def test_base_stream_and_info_are_abstract(self):
        base = ReasoningEngine()
        with self.assertRaises(NotImplementedError):
            base.info()
        with self.assertRaises(NotImplementedError):
            list(base.stream("q"))


if __name__ == "__main__":
    unittest.main()
