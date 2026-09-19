"""The shared truth layer between the training reward and the browser.

Everything the UI shows about a completion — the parsed answer, the verified
arithmetic steps, the reward breakdown — is produced here, and it is produced by
calling the *same* functions the GRPO trainer calls. Nothing in this module
reimplements a reward or a verifier.

That constraint is the whole point of the serving layer. A demo that recomputed
"roughly what the reward does" would be a plausible-looking lie: it could show a
green checkmark on a step the trainer scored as wrong, and nobody would notice.
Routing the browser through ``rewards.outcome_reward`` and friends means the
numbers on screen are, by construction, the numbers that shaped the policy.
"""

from __future__ import annotations

from typing import Any

from ..metrics import (
    NUMBER_RE,
    extract_final_number,
    gold_number,
    iter_arithmetic_steps,
    numeric_equal,
    token_diversity,
)
from ..rewards import diversity_reward, format_reward, outcome_reward, process_reward


def normalise_gold(gold: str) -> str:
    """Reduce a gold answer to its bare numeric string.

    Callers hand us gold in two shapes: a bare ``"42"`` typed into the UI, or a
    full GSM8K answer string ending in ``"#### 42"``. ``gold_number`` handles
    both, and when it finds no number at all we fall back to the literal text so
    a nonsense gold grades as "not matched" instead of raising into a 500.
    """
    try:
        return gold_number(gold)
    except ValueError:
        return gold.strip()


def analyze(text: str, gold: str | None = None) -> dict[str, Any]:
    """Produce the full Analysis payload for one completion.

    ``gold`` is optional because the UI must stay useful on a free-form question
    nobody has a reference answer for; in that case ``outcome`` is null and is
    excluded from the reward total rather than being silently counted as zero,
    which would understate the total against a graded run.

    The two rewards that take a gold answer want *different shapes of it*, so
    they are deliberately given different values. ``outcome_reward`` compares
    final answers and gets ``gold_value``, the bare number. ``process_reward``
    scales substance by the reference solution's own step count, so it gets the
    raw ``gold`` string with its arithmetic still in it; passing ``gold_value``
    there would strip the equations, silently drop the floor back to the
    constant fallback, and print a process reward the trainer never gave.
    """
    steps = [
        {"index": index, **step} for index, step in enumerate(iter_arithmetic_steps(text))
    ]
    answer = extract_final_number(text)
    if NUMBER_RE.findall(text):
        answer_source: str | None = "marker"
    elif answer is not None:
        answer_source = "fallback"
    else:
        answer_source = None

    prompts = [""]
    completions = [text]
    gold_value = normalise_gold(gold) if gold is not None else None

    outcome: float | None = None
    correct: bool | None = None
    if gold_value is not None:
        correct = numeric_equal(answer, gold_value)
        try:
            outcome = outcome_reward(prompts, completions, answer=[gold_value])[0]
        except ValueError:
            outcome = 0.0

    fmt = format_reward(prompts, completions)[0]
    process = process_reward(prompts, completions, answer=[gold] if gold is not None else None)[0]
    diversity = diversity_reward(prompts, completions)[0]
    total = fmt + process + diversity + (outcome or 0.0)

    return {
        "answer": answer,
        "answer_source": answer_source,
        "format_ok": fmt > 0,
        "steps": steps,
        "steps_correct": sum(1 for step in steps if step["ok"]),
        "steps_total": len(steps),
        "steps_unique": sum(1 for step in steps if not step["duplicate"]),
        "token_diversity": token_diversity(text),
        "rewards": {
            "outcome": outcome,
            "format": fmt,
            "process": process,
            "diversity": diversity,
            "total": round(total, 4),
        },
        "gold": gold_value,
        "correct": correct,
    }
