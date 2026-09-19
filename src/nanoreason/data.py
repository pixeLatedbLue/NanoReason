from __future__ import annotations

import re
from typing import TYPE_CHECKING

from .config import EvalTaskConfig

if TYPE_CHECKING:
    from datasets import Dataset


def load_eval_dataset(task: EvalTaskConfig) -> "Dataset":
    from datasets import load_dataset

    if task.subset:
        data = load_dataset(task.dataset, task.subset)[task.split]
    else:
        data = load_dataset(task.dataset)[task.split]

    if task.offset < 0:
        raise ValueError(f"Dataset offset must be non-negative, got {task.offset}")
    if task.max_samples is not None and task.max_samples <= 0:
        raise ValueError(f"max_samples must be positive or null, got {task.max_samples}")

    if task.seed is not None:
        data = data.shuffle(seed=task.seed)
    if task.offset:
        if task.offset >= len(data):
            raise ValueError(f"Offset {task.offset} leaves no samples in split of size {len(data)}")
        data = data.select(range(task.offset, len(data)))
    if task.max_samples is not None:
        data = data.select(range(min(task.max_samples, len(data))))
    if len(data) == 0:
        raise ValueError(f"Evaluation dataset is empty for {task.dataset}/{task.split}")
    return data


_EQ_RE = re.compile(r"\d\s*[+\-*/=](?=\s*-?\d)")


def reasoning_difficulty(answer: str) -> int:
    """Cheap proxy for problem difficulty used to order the curriculum.

    Counts digit-adjacent arithmetic operators and non-empty reasoning lines in
    the reference solution; more steps == harder.  Deterministic and
    dataset-agnostic, and immune to prose hyphens/markdown bullets.
    """
    if not answer:
        return 0
    operators = len(_EQ_RE.findall(answer))
    lines = len([ln for ln in answer.splitlines() if ln.strip()])
    return operators + lines


def order_by_curriculum(data: "Dataset", answer_field: str = "answer") -> "Dataset":
    """Return ``data`` sorted easy -> hard by :func:`reasoning_difficulty`.

    Stable sort, so it is reproducible.  Shuffle must be disabled by the caller
    for the ordering to reach the trainer intact.
    """
    if answer_field not in data.column_names:
        return data
    difficulties = [reasoning_difficulty(str(a)) for a in data[answer_field]]
    order = sorted(range(len(data)), key=lambda i: difficulties[i])
    return data.select(order)


def _normalise_gsm8k(data: "Dataset") -> "Dataset":
    """Already in ``{question, answer}`` form; keep only those columns."""
    keep = [c for c in data.column_names if c not in {"question", "answer"}]
    return data.remove_columns(keep) if keep else data


def aqua_to_qa(example: dict) -> dict:
    """Convert an AQuA-RAT row into GSM8K-style ``{question, answer}``.

    The answer keeps the chain-of-thought rationale and ends with the required
    ``#### <letter>`` marker so the same SFT/format machinery applies.
    Raises a diagnosable error on malformed rows instead of a bare KeyError
    deep inside ``dataset.map``.
    """
    missing = [key for key in ("question", "correct") if key not in example]
    if missing:
        raise ValueError(f"AQuA row missing required keys {missing}: {str(example)[:200]}")
    options = example.get("options", [])
    listed = "\n".join(str(opt) for opt in options)
    question = f"{example['question']}\nOptions:\n{listed}"
    rationale = str(example.get("rationale", "")).strip()
    answer = f"{rationale}\n#### {example['correct']}"
    return {"question": question, "answer": answer}


def load_training_dataset(dataset: str, subset: str | None, split: str) -> "Dataset":
    """Load one training source as a uniform ``{question, answer}`` dataset."""
    from datasets import load_dataset

    if dataset in {"aqua_rat", "deepmind/aqua_rat"}:
        raw = load_dataset(dataset, subset) if subset else load_dataset(dataset)
        data = raw[split]
        return data.map(aqua_to_qa, remove_columns=data.column_names)
    raw = load_dataset(dataset, subset) if subset else load_dataset(dataset)
    return _normalise_gsm8k(raw[split])


def load_combined_training(
    dataset: str,
    subset: str | None,
    split: str,
    extra_datasets: list[str] | None = None,
) -> "Dataset":
    """Load the primary training set plus any ``extra_datasets`` and concat.

    Extra datasets are given as ``"name"`` or ``"name:subset"`` strings and are
    coerced to the same ``{question, answer}`` schema.
    """
    from datasets import concatenate_datasets

    parts = [load_training_dataset(dataset, subset, split)]
    for spec in extra_datasets or []:
        name, _, sub = spec.partition(":")
        parts.append(load_training_dataset(name, sub or None, split))
    if len(parts) == 1:
        return parts[0]
    return concatenate_datasets(parts)
