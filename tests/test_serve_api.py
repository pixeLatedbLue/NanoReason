import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from nanoreason.serve import engine as engine_mod  # noqa: E402
from nanoreason.serve import runs as runs_mod  # noqa: E402
from nanoreason.serve.app import create_app  # noqa: E402

GOOD_COT = "Jane starts with 5.\n5 + 7 = 12\n12 * 2 = 24\n#### 24"


def _run_payload(name, accuracy, correct, total, per_item=None):
    """A minimal but structurally real evaluation artifact."""
    return {
        "run_name": name,
        "base_model": "microsoft/Phi-3-mini-4k-instruct",
        "adapter": None if name == "baseline" else "artifacts/grpo-final",
        "config_path": "configs/default.toml",
        "created_at_unix": 1_700_000_000 if name == "baseline" else 1_700_000_100,
        "environment": {},
        "settings": {},
        "results": [
            {
                "task": "gsm8k",
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
                "split": "test",
                "dataset": "gsm8k",
                "subset": "main",
                "max_samples": total,
                "seed": 42,
                "examples": [],
                "per_item": per_item,
            }
        ],
        "summary": {"gsm8k": {"accuracy": accuracy, "correct": correct, "total": total}},
    }


class ServeApiTests(unittest.TestCase):
    def setUp(self):
        self.env = mock.patch.dict(os.environ, {"NANOREASON_ENGINE": "demo"})
        self.env.start()
        engine_mod.reset_engine()
        self.client = TestClient(create_app())

    def tearDown(self):
        self.env.stop()
        engine_mod.reset_engine()

    def test_health_reports_demo_engine(self):
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(body["status"], "ok")
        self.assertTrue(body["engine"]["is_demo"])
        self.assertEqual(body["engine"]["kind"], "demo")
        self.assertTrue(body["version"])

    def test_analyze_returns_verified_steps_and_rewards(self):
        res = self.client.post("/api/analyze", json={"text": GOOD_COT, "gold": "24"})
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(body["answer"], "24")
        self.assertEqual(body["answer_source"], "marker")
        self.assertEqual((body["steps_correct"], body["steps_total"]), (2, 2))
        self.assertTrue(body["correct"])
        self.assertEqual(body["rewards"]["outcome"], 2.0)
        self.assertEqual(body["rewards"]["format"], 0.5)

    def test_analyze_without_gold_leaves_outcome_null(self):
        body = self.client.post("/api/analyze", json={"text": GOOD_COT}).json()
        self.assertIsNone(body["rewards"]["outcome"])
        self.assertIsNone(body["correct"])

    def test_analyze_rejects_empty_text(self):
        self.assertEqual(self.client.post("/api/analyze", json={"text": ""}).status_code, 422)

    def test_solve_returns_completion_and_analysis(self):
        res = self.client.post("/api/solve", json={"question": "A pen costs 3 dollars. 4 pens?"})
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertTrue(body["completion"].strip())
        self.assertTrue(body["analysis"]["format_ok"])
        self.assertEqual(body["engine"], "demo")
        self.assertGreaterEqual(body["latency_ms"], 0.0)

    def test_solve_validates_token_budget(self):
        res = self.client.post("/api/solve", json={"question": "x", "max_new_tokens": 99999})
        self.assertEqual(res.status_code, 422)

    def test_trapme_question_produces_exactly_one_bad_step(self):
        body = self.client.post("/api/solve", json={"question": "trapme 5 and 7 times 2"}).json()
        steps = body["analysis"]["steps"]
        self.assertEqual(sum(1 for s in steps if not s["ok"]), 1)

    def test_stream_emits_token_frames_then_done(self):
        with self.client.stream("GET", "/api/solve/stream?question=5 apples and 7 more") as res:
            self.assertEqual(res.status_code, 200)
            self.assertIn("text/event-stream", res.headers["content-type"])
            raw = "".join(res.iter_text())
        frames = [json.loads(b[len("data: "):]) for b in raw.strip().split("\n\n") if b.strip()]
        self.assertGreaterEqual(len(frames), 2)
        self.assertEqual(frames[-1]["type"], "done")
        self.assertTrue(frames[-1]["completion"].strip())
        self.assertTrue(any(f["type"] == "token" for f in frames[:-1]))
        joined = "".join(f["text"] for f in frames if f["type"] == "token")
        self.assertEqual(joined, frames[-1]["completion"])

    def test_runs_compare_and_traversal_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            base_items = [{"index": i, "correct": i % 4 != 0} for i in range(100)]
            cand_items = [{"index": i, "correct": i % 10 != 0} for i in range(100)]
            (tmp_path / "baseline.json").write_text(
                json.dumps(_run_payload("baseline", 0.75, 75, 100, base_items)), encoding="utf-8"
            )
            (tmp_path / "grpo_final.json").write_text(
                json.dumps(_run_payload("grpo_final", 0.90, 90, 100, cand_items)), encoding="utf-8"
            )
            (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")

            with mock.patch.dict(os.environ, {"NANOREASON_RESULTS": str(tmp_path)}):
                runs = self.client.get("/api/runs").json()["runs"]
                self.assertEqual({r["name"] for r in runs}, {"baseline", "grpo_final"})
                self.assertEqual(runs[0]["name"], "grpo_final")

                self.assertEqual(self.client.get("/api/runs/baseline").status_code, 200)
                self.assertEqual(self.client.get("/api/runs/nope").status_code, 404)
                for evil in ("../../pyproject", "..%2F..%2Fpyproject", "C:pyproject"):
                    self.assertEqual(self.client.get(f"/api/runs/{evil}").status_code, 404)

                cmp_body = self.client.get(
                    "/api/compare?baseline=baseline&candidate=grpo_final"
                ).json()
                row = cmp_body["rows"][0]
                self.assertEqual(row["task"], "gsm8k")
                self.assertLessEqual(row["baseline"]["ci_low"], row["baseline"]["accuracy"])
                self.assertGreaterEqual(row["baseline"]["ci_high"], row["baseline"]["accuracy"])
                self.assertAlmostEqual(row["delta"], 0.15, places=6)
                self.assertTrue(row["passes"])
                self.assertIsNotNone(row["paired"])
                self.assertEqual(row["paired"]["n_both"], 100)

                bad = self.client.get("/api/compare?baseline=baseline&candidate=ghost")
                self.assertEqual(bad.status_code, 400)
                self.assertIn("ghost", bad.json()["detail"])

    def test_runs_empty_when_directory_missing(self):
        with mock.patch.dict(os.environ, {"NANOREASON_RESULTS": "does/not/exist"}):
            self.assertEqual(self.client.get("/api/runs").json()["runs"], [])

    def test_configs_lists_shipped_experiments(self):
        body = self.client.get("/api/configs").json()
        names = {c["name"] for c in body["configs"]}
        self.assertTrue({"default", "smoke", "kaggle_t4", "rtx4060_8gb"} <= names)

    def test_root_serves_the_single_page_app(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("text/html", res.headers["content-type"])
        self.assertIn("NanoReason", res.text)


class ServeDegradationTests(unittest.TestCase):
    """What the API does when its inputs are hostile or corrupt.

    The dashboard reads a directory that a training job may be writing to right
    now, and its run names arrive from a URL. Both are covered here, because
    "skipped a bad row" and "500ed the whole page" look identical in a test that
    only ever feeds the happy path.
    """

    MIXED_COT = "Step 1: 5 + 7 = 12\nStep 2: 12 * 3 = 36\nStep 3: 36 - 6 = 31\nStep 4: 8 / 0 = 4\n#### 30"

    def setUp(self):
        self.env = mock.patch.dict(os.environ, {"NANOREASON_ENGINE": "demo"})
        self.env.start()
        engine_mod.reset_engine()
        self.client = TestClient(create_app())

    def tearDown(self):
        self.env.stop()
        engine_mod.reset_engine()

    def test_analyze_reports_an_undefined_step_without_crediting_it(self):
        body = self.client.post(
            "/api/analyze", json={"text": self.MIXED_COT, "gold": "30"}
        ).json()

        self.assertEqual((body["steps_correct"], body["steps_total"]), (2, 4))
        self.assertEqual(body["steps_unique"], 4)
        self.assertIsNone(body["steps"][3]["computed"])
        self.assertFalse(body["steps"][3]["ok"])
        self.assertFalse(body["steps"][2]["ok"])
        self.assertEqual(body["rewards"]["process"], 0.25)
        self.assertEqual(body["rewards"]["outcome"], 2.0)
        self.assertAlmostEqual(body["rewards"]["total"], 2.75)

    def test_solve_rejects_an_empty_question(self):
        self.assertEqual(self.client.post("/api/solve", json={"question": ""}).status_code, 422)
        self.assertEqual(self.client.get("/api/solve/stream?question=").status_code, 422)

    def test_stream_still_closes_with_done_when_generation_fails(self):
        class Boom(engine_mod.ReasoningEngine):
            def info(self):
                return engine_mod.EngineInfo("demo", None, None, "cpu", True)

            def stream(self, question, max_new_tokens=384):
                yield "partial "
                raise RuntimeError("generation exploded")

        engine_mod._ENGINE = Boom()
        try:
            raw = self.client.get("/api/solve/stream?question=x").text
        finally:
            engine_mod.reset_engine()
        frames = [json.loads(b[len("data: "):]) for b in raw.strip().split("\n\n") if b.strip()]

        types = [f["type"] for f in frames]
        self.assertIn("error", types)
        self.assertEqual(types[-1], "done")
        self.assertEqual(frames[-1]["completion"], "partial ")
        self.assertIsNotNone(frames[-1]["analysis"])

    def test_stream_done_frame_carries_an_analysis_even_when_nothing_is_generated(self):
        class Silent(engine_mod.ReasoningEngine):
            def info(self):
                return engine_mod.EngineInfo("demo", None, None, "cpu", True)

            def stream(self, question, max_new_tokens=384):
                return iter(())

        engine_mod._ENGINE = Silent()
        try:
            raw = self.client.get("/api/solve/stream?question=x").text
        finally:
            engine_mod.reset_engine()
        final = [json.loads(b[len("data: "):]) for b in raw.strip().split("\n\n") if b.strip()][-1]

        self.assertEqual(final["type"], "done")
        self.assertEqual(final["completion"], "")
        self.assertIsNotNone(final["analysis"])
        self.assertEqual(final["analysis"]["steps_total"], 0)
        self.assertIsNone(final["analysis"]["answer"])

    def test_corrupt_artifacts_are_skipped_rather_than_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / "results"
            results.mkdir()
            (results / "good.json").write_text(
                json.dumps(_run_payload("good", 0.5, 5, 10)), encoding="utf-8"
            )
            (results / "truncated.json").write_text('{"run_name": "x', encoding="utf-8")
            (results / "garbage.json").write_text(
                json.dumps({"run_name": "g", "summary": {"gsm8k": "not-a-stat"}}), encoding="utf-8"
            )
            (results / "list.json").write_text("[1, 2, 3]", encoding="utf-8")
            (results / "forgetting_probe.json").write_text(
                json.dumps({"summary": {"mmlu": {"accuracy": 0.6, "correct": 6, "total": 10}}}),
                encoding="utf-8",
            )

            with mock.patch.dict(os.environ, {"NANOREASON_RESULTS": str(results)}):
                response = self.client.get("/api/runs")

                self.assertEqual(response.status_code, 200)
                names = [r["name"] for r in response.json()["runs"]]
                self.assertEqual(names, ["good"])
                self.assertNotIn("forgetting_probe", names)

    def test_compare_survives_unusable_per_item_records(self):
        broken = {
            "run_name": "b",
            "created_at_unix": 1,
            "summary": {"gsm8k": {"accuracy": 0.5, "correct": 5, "total": 10}},
            "results": [{"task": "gsm8k", "per_item": [{"idx": 0, "correct": True}]}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            for stem in ("a", "b"):
                (Path(tmp) / f"{stem}.json").write_text(json.dumps(broken), encoding="utf-8")

            with mock.patch.dict(os.environ, {"NANOREASON_RESULTS": tmp}):
                response = self.client.get("/api/compare?baseline=a&candidate=b")

        self.assertEqual(response.status_code, 200)
        row = response.json()["rows"][0]
        self.assertIsNone(row["paired"])
        self.assertIsNotNone(row["baseline"])

    def test_traversal_cannot_reach_a_real_file_outside_the_results_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = root / "results"
            results.mkdir()
            (results / "inside.json").write_text(
                json.dumps(_run_payload("inside", 0.5, 5, 10)), encoding="utf-8"
            )
            (root / "outside.json").write_text(json.dumps({"secret": "leaked"}), encoding="utf-8")

            with mock.patch.dict(os.environ, {"NANOREASON_RESULTS": str(results)}):
                for evil in ("../outside", "..%2Foutside", "..\\outside"):
                    with self.subTest(name=evil):
                        response = self.client.get(f"/api/runs/{evil}")
                        self.assertEqual(response.status_code, 404)
                        self.assertNotIn("leaked", response.text)

                self.assertIsNone(runs_mod.load_run("../outside"))
                self.assertIsNone(runs_mod.load_run("..\\outside"))
                self.assertIsNone(runs_mod.load_run(".."))
                self.assertIsNone(runs_mod.load_run(""))
                self.assertIsNone(runs_mod.load_run("C:/Windows/win"))
                self.assertIsNotNone(runs_mod.load_run("inside"))

    def test_empty_results_env_does_not_fall_back_to_the_working_directory(self):
        with mock.patch.dict(os.environ, {"NANOREASON_RESULTS": ""}):
            self.assertEqual(runs_mod.runs_dir(), Path("results"))


class ServeConfigsTests(unittest.TestCase):
    def setUp(self):
        self._cwd = Path.cwd()
        os.chdir(ROOT)
        self.client = TestClient(create_app())

    def tearDown(self):
        os.chdir(self._cwd)

    def test_lists_exactly_the_shipped_configs(self):
        configs = self.client.get("/api/configs").json()["configs"]

        self.assertEqual(
            [c["name"] for c in configs], ["default", "kaggle_t4", "rtx4060_8gb", "smoke"]
        )
        default = next(c for c in configs if c["name"] == "default")
        self.assertEqual(default["base_model"], "microsoft/Phi-3-mini-4k-instruct")
        self.assertTrue(default["load_in_4bit"])
        self.assertEqual(default["tasks"], ["aqua", "gsm8k", "mmlu", "strategyqa"])
        self.assertEqual(default["num_generations"], 8)
        self.assertTrue(default["path"].endswith("configs/default.toml"))


if __name__ == "__main__":
    unittest.main()
