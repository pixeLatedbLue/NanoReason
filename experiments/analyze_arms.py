from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

MARKUP = "<<"


def load_items(directory: Path, arm: str) -> dict[int, dict]:
    path = directory / "progress" / f"{arm}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist; has the {arm} arm run?")
    return {item["index"]: item for item in json.loads(path.read_text(encoding="utf-8"))["items"]}


def failure_breakdown(items: dict[int, dict], keys: list[int]) -> dict[str, int]:
    wrong = [k for k in keys if not items[k]["correct"]]
    return {
        "wrong": len(wrong),
        "no_answer_marker": sum(not items[k]["has_answer_marker"] for k in wrong),
        "token_cap_hit": sum(items[k]["token_cap_hit"] for k in wrong),
        "parsed_but_wrong": sum(
            items[k]["has_answer_marker"] and items[k]["prediction"] not in (None, "")
            for k in wrong),
    }


def describe(items: dict[int, dict], keys: list[int]) -> dict[str, float]:
    tokens = [items[k]["generated_tokens"] for k in keys]
    correct = [k for k in keys if items[k]["correct"]]
    return {
        "n": len(keys),
        "accuracy": len(correct) / len(keys) if keys else 0.0,
        "median_tokens": statistics.median(tokens) if tokens else 0,
        "markup_rate": sum(MARKUP in items[k]["completion"] for k in keys) / len(keys) if keys else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default="artifacts/slm/evaluation-full")
    parser.add_argument("--baseline", default="base")
    parser.add_argument("--candidate", default="sft")
    parser.add_argument("--examples", type=int, default=0,
                        help="print this many items the baseline got right and the candidate did not")
    args = parser.parse_args()

    directory = Path(args.directory)
    baseline = load_items(directory, args.baseline)
    candidate = load_items(directory, args.candidate)
    shared = sorted(set(baseline) & set(candidate))
    if not shared:
        raise ValueError("the two arms share no items, so they cannot be compared")

    only_baseline = [k for k in shared if baseline[k]["correct"] and not candidate[k]["correct"]]
    only_candidate = [k for k in shared if candidate[k]["correct"] and not baseline[k]["correct"]]

    report = {
        "n_shared": len(shared),
        "discordant": {
            f"only_{args.baseline}": len(only_baseline),
            f"only_{args.candidate}": len(only_candidate),
        },
        "overall": {
            args.baseline: describe(baseline, shared),
            args.candidate: describe(candidate, shared),
        },
        "failures": {
            args.baseline: failure_breakdown(baseline, shared),
            args.candidate: failure_breakdown(candidate, shared),
        },
    }
    print(json.dumps(report, indent=2))

    for key in only_baseline[: args.examples]:
        print("=" * 72)
        print("question:", baseline[key]["question"])
        print("gold:", baseline[key]["gold"])
        for name, items in ((args.baseline, baseline), (args.candidate, candidate)):
            item = items[key]
            print(f"-- {name}: predicted {item['prediction']} in {item['generated_tokens']} tokens")
            print(item["completion"])
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
