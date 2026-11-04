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
    try:
        return gold_number(gold)
    except ValueError:
        return gold.strip()


def analyze(text: str, gold: str | None = None) -> dict[str, Any]:
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
