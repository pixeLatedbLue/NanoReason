"""Hybrid reward functions for GRPO.

The proposal's central novelty is a *hybrid* reward that combines an
outcome-level signal (ORM) with a dense process-level signal (PRM) so the
policy cannot earn reward by gaming surface patterns alone.

This module implements that with deterministic, hack-proof verifiers:

* ``outcome_reward`` (ORM)  -- did the final answer match the gold answer?
* ``process_reward``  (PRM) -- are the intermediate arithmetic steps actually
  correct?  Each ``a op b = c`` equation is numerically verified, so fake or
  wrong equations are penalised instead of rewarded.
* ``format_reward``         -- is the required ``#### <number>`` present?
* ``diversity_reward``      -- entropy-style anti-collapse term that punishes
  the degenerate repetition GRPO can drift into on small models.

An optional model-based PRM (e.g. a teacher-distilled Skywork reward model) can
be plugged in through ``ModelProcessReward`` without changing the trainer.
"""

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


def outcome_reward(prompts, completions, answer, **kwargs):
    """+2.0 when the extracted final answer matches the gold answer."""
    rewards = []
    for completion, gold in zip(completions, answer):
        pred = extract_final_number(_completion_text(completion))
        rewards.append(2.0 if numeric_equal(pred, gold_number(gold)) else 0.0)
    return rewards


correctness_reward = outcome_reward


_SUBSTANCE_STEPS = 3


def reference_steps(answer) -> int | None:
    """How many distinct verified equations the reference solution contains.

    GSM8K rationales carry their own arithmetic, so the dataset already states
    how much work a problem is worth. Returns ``None`` when the gold answer
    carries no equations (a bare ``#### 42``), leaving the caller to fall back.
    """
    if answer is None:
        return None
    _, total = verify_arithmetic_steps(str(answer), dedupe=True)
    return total or None


def process_reward(prompts, completions, answer=None, **kwargs):
    """Dense step-level signal based on verified arithmetic.

    Base score = ``correct_fraction - 0.5 * wrong_fraction`` over the *unique*
    equations in the completion, so repeating an equation never adds credit.
    Wrong equations are penalised at full weight and a response with no
    equations earns 0. Every step is recomputed, so fabricated arithmetic is
    punished rather than rewarded.

    Positive scores are then scaled by substance, ``min(steps, floor) / floor``.
    The floor is the *reference solution's own* step count rather than a
    constant. A fixed floor of three was the original design and it is
    exploitable in the opposite direction from the one it was guarding: a
    genuinely correct two-step solution to a two-step problem was scaled to
    0.67 while three padded trivial equations reached the full 1.0, so the
    reward preferred padding to honest work (see the paper's Table II).
    Measuring substance against what the problem actually requires removes that
    incentive: an honest solution reaches the ceiling, and padding past the
    reference earns nothing extra.
    """
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


def format_reward(prompts, completions, **kwargs):
    """+0.5 when the response ends with the required ``#### <number>`` marker.

    Trailing sentence punctuation after the number is tolerated ("#### 42.")
    so this stays consistent with what ``extract_final_number`` accepts;
    substantive trailing text still voids the reward.
    """
    pattern = re.compile(r"####\s*\$?-?\d+(?:,\d{3})*(?:\.\d+)?[.!]?\s*$")
    return [0.5 if pattern.search(_completion_text(c).strip()) else 0.0 for c in completions]


def diversity_reward(prompts, completions, min_ratio: float = 0.35, **kwargs):
    """Entropy-style anti-collapse term.

    Penalises degenerate repetition (unique-token ratio below ``min_ratio``)
    with a small negative reward; healthy generations get 0.  Keeps the policy
    from collapsing onto a single repeated string during RL.
    """
    rewards = []
    for completion in completions:
        ratio = token_diversity(_completion_text(completion))
        rewards.append(0.0 if ratio >= min_ratio else -0.5)
    return rewards


class ModelProcessReward:
    """Wraps a HuggingFace sequence-classification reward model as a PRM.

    Lets you swap the deterministic verifier for a teacher-distilled reward
    model (e.g. ``Skywork/Skywork-Reward-Llama-3.1-8B``) without touching the
    trainer.  The model is loaded lazily so importing this module stays cheap
    and CPU-only environments are unaffected.
    """

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
    """Return the list of reward callables for the GRPO trainer.

    ``outcome_reward`` (ORM) and ``format_reward`` are always included; the
    process reward (PRM) and entropy/diversity term are toggleable so reward
    shaping can be ablated (ORM-only first, then add PRM) exactly as the
    project's reward-shaping methodology describes.
    """
    funcs = [outcome_reward, format_reward]
    if use_process:
        funcs.append(process_reward)
    if use_diversity:
        funcs.append(diversity_reward)
    if model_prm:
        funcs.append(ModelProcessReward(model_prm, scale=model_prm_scale))
    return funcs


reasoning_reward = process_reward
