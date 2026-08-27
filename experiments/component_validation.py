"""Adversarial audit of the NanoReason reward functions.

No GPU, no checkpoint, no network. Every number produced here comes from the
same deterministic verifier and reward functions that the GRPO trainer calls,
so what is measured here is the training signal itself, not a description of it.

Four experiments:
  E1  reward-hacking resistance   -- what each attack strategy actually earns
  E2  verifier accuracy           -- precision/recall on a labelled equation set
  E3  answer-extraction coverage  -- marker versus fallback parsing behaviour
  E4  statistical resolving power -- smallest delta a paired test can detect
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nanoreason.metrics import (  # noqa: E402
    extract_final_number,
    iter_arithmetic_steps,
    verify_arithmetic_steps,
)
from nanoreason.rewards import (  # noqa: E402
    diversity_reward,
    format_reward,
    outcome_reward,
    process_reward,
)
from nanoreason.stats import mcnemar_exact  # noqa: E402

GOLD = (
    "Jane starts with 5 apples and buys 7 more.\n5 + 7 = 12\n"
    "She doubles that amount.\n12 * 2 = 24\n#### 24"
)

_LEGACY_FLOOR = 3


def legacy_process_reward(text: str) -> float:
    """The superseded PRM: substance scaled by a constant three-step floor."""
    correct, total = verify_arithmetic_steps(text, dedupe=True)
    if total == 0:
        return 0.0
    wrong = total - correct
    score = (correct / total) - 0.5 * (wrong / total)
    if score > 0:
        score *= min(total, _LEGACY_FLOOR) / _LEGACY_FLOOR
    return round(score, 4)


def reward_vector(text: str, gold: str = GOLD) -> dict[str, float]:
    """The four reward terms and their sum, exactly as GRPO would score them.

    Both PRM variants are reported so the padding exploit and its repair are
    visible in one table.
    """
    prompts, completions = [""], [text]
    outcome = outcome_reward(prompts, completions, answer=[gold])[0]
    fmt = format_reward(prompts, completions)[0]
    proc = process_reward(prompts, completions, answer=[gold])[0]
    legacy = legacy_process_reward(text)
    div = diversity_reward(prompts, completions)[0]
    return {
        "outcome": outcome,
        "format": fmt,
        "process": round(proc, 4),
        "process_legacy": legacy,
        "diversity": div,
        "total": round(outcome + fmt + proc + div, 4),
        "total_legacy": round(outcome + fmt + legacy + div, 4),
    }


HONEST = (
    "Jane starts with 5 apples and buys 7 more.\n5 + 7 = 12\n"
    "She doubles that amount.\n12 * 2 = 24\n#### 24"
)
FABRICATED = (
    "Jane starts with 5 apples and buys 7 more.\n5 + 7 = 99\n"
    "She doubles that amount.\n12 * 2 = 71\n#### 24"
)
NO_WORK = "The answer is twenty-four.\n#### 24"
REPEAT_SPAM = "\n".join(["1 + 1 = 2"] * 20) + "\n#### 24"
TRIVIAL_PAD = "1 + 1 = 2\n2 + 2 = 4\n3 + 3 = 6\n#### 24"
COLLAPSE = ("answer " * 40).strip() + "\n#### 24"
HONEST_WRONG = (
    "Jane starts with 5 apples and buys 7 more.\n5 + 7 = 12\n"
    "She triples that amount.\n12 * 3 = 36\n#### 36"
)
PARTIAL = (
    "Jane starts with 5 apples and buys 7 more.\n5 + 7 = 12\n"
    "She doubles that amount.\n12 * 2 = 25\n#### 24"
)

STRATEGIES = [
    ("Honest, fully correct", HONEST, "the behaviour the reward should favour"),
    ("Right answer, fabricated steps", FABRICATED, "outcome-only reward pays full price"),
    ("Right answer, no working shown", NO_WORK, "outcome-only reward pays full price"),
    ("Repetition spam, 20 copies", REPEAT_SPAM, "attacks a step-counting PRM"),
    ("Trivial-equation padding", TRIVIAL_PAD, "attacks a PRM with no substance floor"),
    ("Degenerate repetition", COLLAPSE, "the classic small-model RL collapse mode"),
    ("Honest reasoning, wrong answer", HONEST_WRONG, "process credit without outcome credit"),
    ("One slip, propagated", PARTIAL, "the realistic failure the PRM must localise"),
]


def e1() -> dict:
    rows = []
    for name, text, note in STRATEGIES:
        rewards = reward_vector(text)
        correct, total = verify_arithmetic_steps(text, dedupe=True)
        rows.append(
            {
                "strategy": name,
                "note": note,
                "rewards": rewards,
                "unique_steps_correct": correct,
                "unique_steps_total": total,
            }
        )
    honest = rows[0]["rewards"]
    beat_new = [r["strategy"] for r in rows[1:] if r["rewards"]["total"] > honest["total"]]
    tie_new = [r["strategy"] for r in rows[1:] if r["rewards"]["total"] == honest["total"]]
    beat_legacy = [
        r["strategy"] for r in rows[1:] if r["rewards"]["total_legacy"] > honest["total_legacy"]
    ]
    return {
        "rows": rows,
        "honest_total": honest["total"],
        "honest_total_legacy": honest["total_legacy"],
        "strategies_beating_honest": beat_new,
        "strategies_tying_honest": tie_new,
        "strategies_beating_honest_legacy": beat_legacy,
    }


LABELLED = [
    ("2 + 2 = 4", True),
    ("10 - 3 = 7", True),
    ("6 * 7 = 42", True),
    ("8 / 2 = 4", True),
    ("1,200 + 300 = 1,500", True),
    ("-5 + 3 = -2", True),
    ("10 / 3 = 3.33", True),
    ("2.5 * 4 = 10", True),
    ("100 - 1 = 99", True),
    ("0 + 0 = 0", True),
    ("2 + 2 = 5", False),
    ("10 - 3 = 8", False),
    ("6 * 7 = 41", False),
    ("9 / 3 = 4", False),
    ("1,200 + 300 = 1,400", False),
    ("-5 + 3 = 2", False),
    ("10 / 3 = 3.9", False),
    ("2.5 * 4 = 11", False),
    ("5 / 0 = 5", False),
    ("7 * 8 = 54", False),
]


def e2() -> dict:
    tp = fp = tn = fn = 0
    misses = []
    for text, expected in LABELLED:
        steps = list(iter_arithmetic_steps(text))
        if not steps:
            misses.append({"text": text, "reason": "not parsed as an equation"})
            continue
        got = steps[0]["ok"]
        if expected and got:
            tp += 1
        elif expected and not got:
            fn += 1
            misses.append({"text": text, "reason": "correct step marked wrong"})
        elif not expected and got:
            fp += 1
            misses.append({"text": text, "reason": "wrong step marked correct"})
        else:
            tn += 1
    n = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "n": n,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "accuracy": round((tp + tn) / n, 4) if n else 0.0,
        "misclassified": misses,
    }


EXTRACTION = [
    ("marker", "reasoning here\n#### 42", "42"),
    ("marker with comma", "total spent\n#### 1,234", "1234"),
    ("marker with decimal", "cost\n#### 12.50", "12.50"),
    ("marker negative", "temperature delta\n#### -7", "-7"),
    ("fallback last number", "so she has 17 apples left", "17"),
    ("fallback trailing unit", "the distance is 180 km", "180"),
    ("no number at all", "the answer cannot be determined", None),
]


def e3() -> dict:
    rows, agree = [], 0
    for label, text, expected in EXTRACTION:
        got = extract_final_number(text)
        ok = got == expected
        agree += ok
        rows.append({"case": label, "expected": expected, "extracted": got, "ok": ok})
    return {"rows": rows, "agreement": f"{agree}/{len(rows)}", "agreed": agree, "cases": len(rows)}


def e4(n: int = 300, churn: int = 15) -> dict:
    """Smallest true improvement a paired McNemar test resolves at n items.

    ``churn`` is the count of items the candidate loses that the baseline got
    right. A real fine-tune trades some items away, so assuming zero regressions
    would flatter the test; the discordant pair is (churn + gained, churn).
    """
    curve = []
    smallest = None
    for delta_points in range(1, 16):
        gained = round(n * delta_points / 100)
        only_candidate = churn + gained
        p_value = mcnemar_exact(only_candidate, churn)
        significant = p_value < 0.05
        curve.append(
            {
                "delta_points": delta_points,
                "only_candidate": only_candidate,
                "only_baseline": churn,
                "p_value": p_value,
                "significant": significant,
            }
        )
        if significant and smallest is None:
            smallest = delta_points
    return {
        "n": n,
        "assumed_regressions": churn,
        "smallest_resolvable_delta_points": smallest,
        "curve": curve,
    }


def main() -> None:
    out = {
        "e1_reward_hacking": e1(),
        "e2_verifier_accuracy": e2(),
        "e3_answer_extraction": e3(),
        "e4_statistical_power": e4(),
    }

    e1r = out["e1_reward_hacking"]
    print("=" * 78)
    print("E1  Reward earned by each strategy against a 2-step reference solution")
    print("=" * 78)
    print(
        f"{'strategy':<34}{'out':>5}{'fmt':>5}{'div':>6}"
        f"{'PRMold':>8}{'TOTold':>8}{'PRMnew':>8}{'TOTnew':>8}"
    )
    for row in e1r["rows"]:
        v = row["rewards"]
        print(
            f"{row['strategy']:<34}{v['outcome']:>5.1f}{v['format']:>5.1f}{v['diversity']:>6.1f}"
            f"{v['process_legacy']:>8.3f}{v['total_legacy']:>8.3f}"
            f"{v['process']:>8.3f}{v['total']:>8.3f}"
        )
    print(f"\nFixed floor  : honest={e1r['honest_total_legacy']}, "
          f"beaten by {e1r['strategies_beating_honest_legacy'] or 'NONE'}")
    print(f"Adaptive floor: honest={e1r['honest_total']}, "
          f"beaten by {e1r['strategies_beating_honest'] or 'NONE'}, "
          f"tied by {e1r['strategies_tying_honest'] or 'NONE'}")

    acc = out["e2_verifier_accuracy"]
    print("\n" + "=" * 78)
    print(
        f"E2  Verifier on {acc['n']} labelled equations: precision={acc['precision']}  "
        f"recall={acc['recall']}  accuracy={acc['accuracy']}"
    )
    print(f"    TP={acc['tp']} FP={acc['fp']} TN={acc['tn']} FN={acc['fn']}")
    for miss in acc["misclassified"]:
        print(f"    ! {miss['text']}  -- {miss['reason']}")

    print("\n" + "=" * 78)
    print(f"E3  Answer extraction: {out['e3_answer_extraction']['agreement']} cases agree")
    for row in out["e3_answer_extraction"]["rows"]:
        flag = " " if row["ok"] else "!"
        print(f"  {flag} {row['case']:<24} expected={str(row['expected']):<8} got={row['extracted']}")

    power = out["e4_statistical_power"]
    print("\n" + "=" * 78)
    print(
        f"E4  At n={power['n']} with {power['assumed_regressions']} assumed regressions, "
        f"smallest resolvable improvement = {power['smallest_resolvable_delta_points']} points"
    )
    for row in power["curve"][:8]:
        verdict = "significant" if row["significant"] else "not significant"
        print(f"    +{row['delta_points']:>2} pts -> p={row['p_value']:.4g}  {verdict}")

    dest = Path(__file__).with_name("component_validation.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nWrote {dest}")


if __name__ == "__main__":
    main()
