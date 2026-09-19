"""Training callbacks for GRPO.

``ForgettingProbe`` periodically re-evaluates the policy on a held-out task
(MMLU by default) during RL.  A large drop versus the pre-RL baseline is the
signature of catastrophic forgetting, which is one of the failure modes the
project is explicitly designed to catch and stop early.
"""

from __future__ import annotations

from pathlib import Path

from transformers import TrainerCallback

from .config import EvalTaskConfig
from .evaluate import EVALUATORS
from .metrics import write_json

TASK_SPECS = {
    "mmlu": ("cais/mmlu", "all", "test"),
    "gsm8k": ("gsm8k", "main", "test"),
    "strategyqa": ("tasksource/strategy-qa", None, "train"),
    "aqua": ("aqua_rat", "raw", "test"),
}


class ForgettingProbe(TrainerCallback):
    def __init__(
        self,
        tokenizer,
        *,
        task: str = "mmlu",
        max_samples: int = 200,
        every_steps: int = 100,
        output_dir: str = "results",
        seed: int = 42,
        few_shot: bool = True,
    ):
        if task not in TASK_SPECS:
            raise ValueError(f"Unknown forgetting-probe task: {task}")
        if task not in EVALUATORS:
            raise ValueError(f"No evaluator registered for task: {task}")
        dataset, subset, split = TASK_SPECS[task]
        self.tokenizer = tokenizer
        self.task = task
        self.every_steps = every_steps
        self.few_shot = few_shot
        self.output_path = Path(output_dir) / "forgetting_probe.json"
        self.history: list[dict] = []
        self.task_cfg = EvalTaskConfig(
            dataset=dataset, subset=subset, split=split, max_samples=max_samples, seed=seed
        )

    def _run(self, step: int, model) -> None:
        was_training = model.training
        model.eval()
        try:
            result = EVALUATORS[self.task](self.tokenizer, model, self.task_cfg, self.few_shot)
        finally:
            if was_training:
                model.train()
        entry = {"step": step, "task": self.task, "accuracy": result.accuracy}
        self.history.append(entry)
        write_json(self.output_path, {"task": self.task, "history": self.history})
        print(f"[forgetting-probe] step {step} {self.task} acc={result.accuracy:.4f}")

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is None or self.every_steps <= 0:
            return
        if state.global_step > 0 and state.global_step % self.every_steps == 0:
            self._run(state.global_step, model)

    def on_train_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        if self.history and self.history[-1]["step"] == state.global_step:
            return
        self._run(state.global_step, model)
