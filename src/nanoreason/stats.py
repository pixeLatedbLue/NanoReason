from __future__ import annotations

import math


def wilson_interval(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    n = float(total)
    p_hat = correct / n
    denom = 1.0 + (z * z) / n
    centre = (p_hat + (z * z) / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt((p_hat * (1.0 - p_hat)) / n + (z * z) / (4.0 * n * n))
    low = centre - margin
    high = centre + margin
    if correct <= 0:
        low = 0.0
    if correct >= total:
        high = 1.0
    return (min(1.0, max(0.0, low)), min(1.0, max(0.0, high)))


def mcnemar_exact(only_a: int, only_b: int) -> float:
    n = only_a + only_b
    if n <= 0:
        return 1.0
    k_max = min(only_a, only_b)
    tail = sum(math.comb(n, k) for k in range(k_max + 1))
    return min(1.0, (2 * tail) / (1 << n))


def paired_comparison(
    baseline_items: list[dict] | None, candidate_items: list[dict] | None
) -> dict | None:
    if not baseline_items or not candidate_items:
        return None
    baseline_map = {item["index"]: bool(item["correct"]) for item in baseline_items}
    candidate_map = {item["index"]: bool(item["correct"]) for item in candidate_items}
    shared = baseline_map.keys() & candidate_map.keys()
    if not shared:
        return None
    only_candidate = sum(1 for i in shared if candidate_map[i] and not baseline_map[i])
    only_baseline = sum(1 for i in shared if baseline_map[i] and not candidate_map[i])
    p_value = mcnemar_exact(only_candidate, only_baseline)
    return {
        "n_both": len(shared),
        "only_candidate": only_candidate,
        "only_baseline": only_baseline,
        "p_value": p_value,
        "significant": p_value < 0.05,
    }


def accuracy(correct: int, total: int) -> float:
    return correct / total if total else 0.0
