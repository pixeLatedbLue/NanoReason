from __future__ import annotations

import re

from .metrics import (
    extract_final_number,
    gold_number,
    numeric_equal,
    token_diversity,
    verify_arithmetic_steps,
)


def _completion_text(completion) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        return "".join(part.get("content", "") for part in completion if isinstance(part, dict))
    return str(completion)


def outcome_reward(prompts, completions, answer, **kwargs) -> list[float]:
    rewards = []
    for completion, gold in zip(completions, answer):
        pred = extract_final_number(_completion_text(completion))
        rewards.append(2.0 if numeric_equal(pred, gold_number(gold)) else 0.0)
    return rewards

correctness_reward = outcome_reward

_SUBSTANCE_STEPS = 3


def reference_steps(answer) -> int | None:
    if answer is None:
        return None
    _, total = verify_arithmetic_steps(str(answer), dedupe=True)
    return total or None


def process_reward(prompts, completions, answer=None, **kwargs) -> list[float]:
    rewards = []
    for index, completion in enumerate(completions):
        text = _completion_text(completion)
        correct, total = verify_arithmetic_steps(text, dedupe=True)
        if total == 0:
            rewards.append(0.0)
            continue
        wrong = total - correct
        score = (correct / total) - 0.5 * (wrong / total)
        if score > 0:
            gold = None
            if answer is not None:
                try:
                    gold = answer[index]
                except (IndexError, TypeError, KeyError):
                    gold = None
            floor = reference_steps(gold) or _SUBSTANCE_STEPS
            score *= min(total, floor) / floor
        rewards.append(round(score, 4))
    return rewards


def format_reward(prompts, completions, **kwargs) -> list[float]:
    pattern = re.compile(r"####\s*\$?-?\d+(?:,\d{3})*(?:\.\d+)?[.!]?\s*$")
    return [0.5 if pattern.search(_completion_text(c).strip()) else 0.0 for c in completions]


def diversity_reward(prompts, completions, min_ratio: float = 0.35, **kwargs) -> list[float]:
    rewards = []
    for completion in completions:
        ratio = token_diversity(_completion_text(completion))
        rewards.append(0.0 if ratio >= min_ratio else -0.5)
    return rewards


class ModelProcessReward:

    __name__ = "model_process_reward"

    def __init__(self, model_name: str, scale: float = 1.0, device: str | None = None):
        self.model_name = model_name
        self.scale = scale
        self.device = device
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, torch_dtype=torch.float16
        )
        if self.device:
            self._model = self._model.to(self.device)
        self._model.eval()

    def __call__(self, prompts, completions, **kwargs):
        import torch

        self._ensure_loaded()
        texts = [_completion_text(c) for c in completions]
        inputs = self._tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=2048
        )
        if self.device:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self._model(**inputs).logits.squeeze(-1)
        return [float(x) * self.scale for x in logits.tolist()]


def build_reward_funcs(
    *,
    use_process: bool = True,
    use_diversity: bool = True,
    model_prm: str | None = None,
    model_prm_scale: float = 1.0,
):
    funcs = [outcome_reward, format_reward]
    if use_process:
        funcs.append(process_reward)
    if use_diversity:
        funcs.append(diversity_reward)
    if model_prm:
        funcs.append(ModelProcessReward(model_prm, scale=model_prm_scale))
    return funcs

reasoning_reward = process_reward
