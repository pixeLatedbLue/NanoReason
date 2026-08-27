from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .stats import paired_comparison, wilson_interval


def load_payload(path: str | Path) -> dict:
    """The whole run JSON, not just the summary block.

    ``load_summary`` deliberately narrows to the accuracy table because that is
    all the plain comparison needs; the confidence-interval and paired tests
    need the per-item detail that lives further down the payload, so they read
    the file through here instead of re-opening it.
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_summary(path: str | Path) -> dict:
    return load_payload(path)["summary"]


def per_item_by_task(payload: dict) -> dict[str, list | None]:
    """Map task -> its ``per_item`` list, or ``None`` when the run lacks one.

    Result files written before ``per_item`` existed simply have no such field,
    and rerunning old evaluations is expensive; tolerating the gap here means
    ``--paired`` degrades to "NA" on those rows rather than crashing.
    """
    by_task: dict[str, list | None] = {}
    for row in payload.get("results") or []:
        task = row.get("task")
        if task is not None:
            by_task[task] = row.get("per_item")
    return by_task


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline and trained evaluation JSON files.")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--min-improvement", type=float, default=0.05)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when any task present in both files misses the threshold.",
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="Append Wilson 95%% confidence bounds for each accuracy.",
    )
    parser.add_argument(
        "--paired",
        action="store_true",
        help="Append an exact McNemar test over the per-item records of both runs.",
    )
    args = parser.parse_args()

    baseline_payload = load_payload(args.baseline)
    candidate_payload = load_payload(args.candidate)
    baseline = baseline_payload["summary"]
    candidate = candidate_payload["summary"]
    baseline_items = per_item_by_task(baseline_payload) if args.paired else {}
    candidate_items = per_item_by_task(candidate_payload) if args.paired else {}

    threshold_col = f"passes_plus_{args.min_improvement * 100:g}pt"
    header = ["task", "baseline", "candidate", "delta", threshold_col]
    if args.ci:
        header += ["baseline_ci_low", "baseline_ci_high", "candidate_ci_low", "candidate_ci_high"]
    if args.paired:
        header += ["n_both", "only_candidate", "only_baseline", "p_value", "significant"]
    extra_columns = len(header) - 5
    print(",".join(header))

    all_pass = True
    for task in sorted(set(baseline) | set(candidate)):
        base_stat = baseline.get(task, {})
        cand_stat = candidate.get(task, {})
        base_acc = base_stat.get("accuracy")
        cand_acc = cand_stat.get("accuracy")
        if base_acc is None or cand_acc is None:
            base_s = "NA" if base_acc is None else f"{base_acc:.4f}"
            cand_s = "NA" if cand_acc is None else f"{cand_acc:.4f}"
            print(",".join([task, base_s, cand_s, "NA", "NA"] + ["NA"] * extra_columns))
            continue
        delta = cand_acc - base_acc
        passes = round(delta, 9) >= round(args.min_improvement, 9) - 1e-9
        all_pass = all_pass and passes
        fields = [task, f"{base_acc:.4f}", f"{cand_acc:.4f}", f"{delta:.4f}", str(passes)]
        if args.ci:
            base_low, base_high = wilson_interval(base_stat.get("correct", 0), base_stat.get("total", 0))
            cand_low, cand_high = wilson_interval(cand_stat.get("correct", 0), cand_stat.get("total", 0))
            fields += [f"{base_low:.4f}", f"{base_high:.4f}", f"{cand_low:.4f}", f"{cand_high:.4f}"]
        if args.paired:
            pair = paired_comparison(baseline_items.get(task), candidate_items.get(task))
            if pair is None:
                fields += ["NA"] * 5
            else:
                fields += [
                    str(pair["n_both"]),
                    str(pair["only_candidate"]),
                    str(pair["only_baseline"]),
                    f"{pair['p_value']:.6g}",
                    str(pair["significant"]),
                ]
        print(",".join(fields))

    if args.strict and not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
