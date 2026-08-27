"""Small-sample statistics for the reported evaluation numbers.

A fine-tuning run is judged on a few hundred seeded samples, and at that size a
headline delta is easy to over-read: +5 points on 300 GSM8K items is well inside
the sampling noise of two coin flips, so a bare accuracy table can announce a
"win" that a different seed would erase. This module supplies the two pieces of
rigour that make such a claim defensible, using only the standard library
(``math``) so it imports on a CPU box with no scientific stack installed:

* ``wilson_interval`` -- how precisely each single accuracy is known.
* ``mcnemar_exact`` / ``paired_comparison`` -- whether the *difference* between
  two runs is bigger than chance, exploiting the fact that both runs are scored
  on the identical seeded subset.
"""

from __future__ import annotations

import math


def wilson_interval(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion, clamped to [0, 1].

    The textbook normal approximation (``p +/- z*sqrt(p(1-p)/n)``) is wrong in
    exactly the regime this project lives in: small ``n`` and proportions near 0
    or 1. It produces intervals that run off the end of the probability scale
    (a 4/50 accuracy gets a negative lower bound) and it collapses to zero width
    at ``p = 0`` or ``p = 1``, implying perfect certainty from a handful of
    samples. The Wilson interval inverts the score test instead, so it stays
    inside [0, 1] and keeps sensible width at the extremes.

    ``total == 0`` returns ``(0.0, 0.0)``: no evidence, and callers render an
    empty run without a special case.
    """
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
    """Two-sided exact binomial (sign) test on the discordant pairs.

    Only the items the two models *disagree* on carry information about which is
    better; the ones both get right and both get wrong are shared difficulty and
    cancel. Under the null hypothesis "the models are equally good", each
    disagreement is a fair coin, so the p-value is the two-sided binomial tail.
    The exact form is used rather than the chi-square approximation because the
    discordant count here is routinely in the single digits, where the
    approximation is unreliable (and undefined without a continuity correction).

    ``n == 0`` (no disagreements at all) returns 1.0: identical behaviour is not
    evidence of a difference.
    """
    n = only_a + only_b
    if n <= 0:
        return 1.0
    k_max = min(only_a, only_b)
    tail = sum(math.comb(n, k) for k in range(k_max + 1))
    return min(1.0, (2 * tail) / (1 << n))


def paired_comparison(
    baseline_items: list[dict] | None, candidate_items: list[dict] | None
) -> dict | None:
    """McNemar comparison of two runs scored on the identical seeded subset.

    This is why ``EvalResult.per_item`` exists. Two independent accuracy numbers
    only support an unpaired test, which throws away the strongest fact we have:
    both runs answered the *same* questions, so the per-question pairing removes
    item difficulty from the noise and makes a real improvement detectable at a
    sample size where an unpaired test would shrug.

    Items are joined on ``"index"``, so a truncated or re-ordered run still
    compares like with like. Returns ``None`` when either side is missing (older
    result files predate ``per_item``) or the two share no indices -- there is
    nothing to pair, and a fabricated p-value would be worse than silence.
    """
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
