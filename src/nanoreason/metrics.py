from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

NUMBER_RE = re.compile(r"####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)")
ANY_NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")

_EQ_NUM = r"-?\d+(?:,\d{3})*(?:\.\d+)?"
EQUATION_RE = re.compile(
    rf"(?<![\d.,eE])({_EQ_NUM})\s*([+\-*/])\s*({_EQ_NUM})\s*=\s*({_EQ_NUM})(?![\d,])"
)


def extract_final_number(text: str) -> str | None:
    marker_matches = NUMBER_RE.findall(text)
    if marker_matches:
        return marker_matches[-1].replace(",", "")
    numbers = ANY_NUMBER_RE.findall(text)
    return numbers[-1].replace(",", "") if numbers else None


LETTER_RE = re.compile(r"####\s*\(?([A-Ea-e])\)?")
_ANSWER_PHRASE_RE = re.compile(
    r"(?:answer|option|choice)\s*(?:is|:)\s*\(?([A-Ea-e])\)?\b", re.IGNORECASE
)


def extract_final_letter(text: str, choices: str = "ABCDE") -> str | None:
    """Extract the multiple-choice letter from a completion.

    Order of preference: the ``#### B`` marker, then an explicit answer phrase
    ("the answer is B"), then a bare letter *alone on the last line*. There is
    deliberately no free-prose fallback: scanning arbitrary text for standalone
    letters matches the English article "a" and silently corrupts accuracy.
    Returns ``None`` when no confident extraction exists so the item counts as
    incorrect for every model symmetrically.
    """
    markers = LETTER_RE.findall(text)
    if markers:
        return markers[-1].upper()
    phrases = _ANSWER_PHRASE_RE.findall(text)
    if phrases:
        return phrases[-1].upper()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        bare = re.match(rf"^\(?([{choices}{choices.lower()}])\)?[.!\s]*$", lines[-1])
        if bare:
            return bare.group(1).upper()
    return None


def numeric_equal(left: str | None, right: str | None, tolerance: float = 1e-6) -> bool:
    if left is None or right is None:
        return False
    try:
        return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=1e-9)
    except ValueError:
        return False


def gold_number(answer: str) -> str:
    value = extract_final_number(answer)
    if value is None:
        raise ValueError(f"Could not extract GSM8K gold answer from: {answer[:120]}")
    return value


def _step_result_matches(computed: float, stated_text: str, tolerance: float) -> bool:
    """True when the stated result matches the computed one.

    Accepts honest human rounding: ``10 / 3 = 3.33`` is a correct step because
    3.33 is the computed value rounded to the stated number of decimals.
    """
    stated = float(stated_text.replace(",", ""))
    if math.isclose(computed, stated, rel_tol=tolerance, abs_tol=tolerance):
        return True
    if "." in stated_text:
        decimals = len(stated_text.split(".")[1])
        return math.isclose(round(computed, decimals), stated, abs_tol=10 ** -(decimals + 6))
    return False


def iter_arithmetic_steps(text: str, tolerance: float = 1e-4) -> Iterator[dict[str, Any]]:
    """Yield one verified-step record per ``a op b = c`` equation in ``text``.

    This is the single source of truth for step verification: both the training
    signal (``verify_arithmetic_steps`` -> ``rewards.process_reward``) and the
    browser UI read the *same* records. Two implementations would eventually
    drift, and the moment they did, the UI's claim that it is showing what the
    reward actually scored would become a lie.

    Each record carries ``lhs``/``op``/``rhs``/``stated`` exactly as they appear
    in the text (thousands separators intact, so the UI can echo the model's own
    words), plus the derived ``computed``, ``ok`` and ``duplicate`` fields.
    ``duplicate`` marks an equation whose *normalised* ``(lhs, op, rhs, stated)``
    tuple already appeared earlier, which is what lets callers count repeated
    equations once. ``computed`` is ``None`` for division by zero: the step is
    real (it counts as attempted) but no value can be checked, so ``ok`` is
    False and it can never earn credit.
    """
    seen: set[tuple[str, str, str, str]] = set()
    for left, op, right, stated in EQUATION_RE.findall(text):
        key = (left.replace(",", ""), op, right.replace(",", ""), stated.replace(",", ""))
        duplicate = key in seen
        seen.add(key)
        try:
            a, b = float(key[0]), float(key[2])
            float(key[3])
        except ValueError:
            continue
        computed: float | None
        if op == "+":
            computed = a + b
        elif op == "-":
            computed = a - b
        elif op == "*":
            computed = a * b
        elif b == 0:
            computed = None
        else:
            computed = a / b
        ok = computed is not None and _step_result_matches(computed, stated, tolerance)
        yield {
            "lhs": left,
            "op": op,
            "rhs": right,
            "stated": stated,
            "computed": computed,
            "ok": ok,
            "raw": f"{left} {op} {right} = {stated}",
            "duplicate": duplicate,
        }


def verify_arithmetic_steps(
    text: str, tolerance: float = 1e-4, dedupe: bool = False
) -> tuple[int, int]:
    """Numerically verify every ``a op b = c`` equation found in ``text``.

    Returns ``(correct_steps, total_steps)``. This is the deterministic core of
    the process reward: an equation only counts as correct if the stated result
    actually matches the computed result, so a model cannot earn process reward
    by printing plausible-looking but wrong arithmetic. With ``dedupe=True``
    identical equations are counted once, so repeating ``1 + 1 = 2`` cannot
    inflate the step count.
    """
    correct = 0
    total = 0
    for step in iter_arithmetic_steps(text, tolerance):
        if dedupe and step["duplicate"]:
            continue
        total += 1
        if step["ok"]:
            correct += 1
    return correct, total


def token_diversity(text: str) -> float:
    """Unique-token ratio in [0, 1]; an entropy-style anti-collapse proxy.

    Degenerate repetition (a known SLM RL failure mode) drives this toward 0.
    """
    tokens = text.split()
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


@dataclass
class EvalResult:
    task: str
    accuracy: float
    correct: int
    total: int
    split: str
    dataset: str
    subset: str | None
    max_samples: int | None
    seed: int
    examples: list[dict[str, Any]]
    few_shot: bool = False
    unparsed: int = 0
    marker_rate: float | None = None
    per_item: list[dict[str, Any]] | None = None


def environment_info() -> dict[str, Any]:
    """Versions of everything that can move an eval number, for the payload."""
    import importlib.metadata as importlib_metadata
    import platform

    packages = {}
    for pkg in ("torch", "transformers", "trl", "peft", "datasets", "accelerate", "bitsandbytes"):
        try:
            packages[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            packages[pkg] = None
    return {"python": platform.python_version(), "platform": platform.system(), "packages": packages}


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def result_payload(
    *,
    run_name: str,
    base_model: str,
    adapter: str | None,
    results: list[EvalResult],
    config_path: str,
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rows = [asdict(result) for result in results]
    return {
        "run_name": run_name,
        "base_model": base_model,
        "adapter": adapter,
        "config_path": config_path,
        "created_at_unix": int(time.time()),
        "environment": environment_info(),
        "settings": settings or {},
        "results": rows,
        "summary": {
            result.task: {
                "accuracy": result.accuracy,
                "correct": result.correct,
                "total": result.total,
            }
            for result in results
        },
    }
