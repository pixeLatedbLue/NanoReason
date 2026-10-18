"""Build the original-project results note from complete matched evaluations."""

import argparse
import json
from pathlib import Path

from nanoreason.matched_eval import atomic_json, compare_complete, digest

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, help="Report date, YYYY-MM-DD")
    args = parser.parse_args()
    run_root = ROOT / "artifacts/slm/evaluation"
    sample = json.loads((run_root / "sample.json").read_text(encoding="utf-8"))
    evidence = {"date": args.date, "sample_sha256": digest(run_root / "sample.json"),
                "source_sha256": sample["source_sha256"], "source_url": sample["source_url"],
                "sample_indices": [r["index"] for r in sample["rows"]], "runs": {}, "paired": {}}
    payloads = {}
    for arm in ("base", "sft", "grpo"):
        path = run_root / "runs" / f"{arm}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = payload["results"][0]
        items = result["per_item"]
        if [r["index"] for r in items] != evidence["sample_indices"]:
            raise ValueError(f"{arm}: incomplete or mismatched sample")
        if payload["provenance"]["sample_sha256"] != evidence["sample_sha256"]:
            raise ValueError(f"{arm}: different sample provenance")
        if result["total"] != len(items) or result["correct"] != sum(r["correct"] for r in items):
            raise ValueError(f"{arm}: inconsistent counts")
        payloads[arm] = payload
        evidence["runs"][arm] = {
            "path": path.relative_to(ROOT).as_posix(), "sha256": digest(path),
            "correct": result["correct"], "total": result["total"],
            "accuracy": result["correct"] / result["total"],
            "diagnostics": payload["diagnostics"], "provenance": payload["provenance"],
            "mean_generated_tokens": sum(r["generated_tokens"] for r in items) / len(items),
        }
    for baseline, candidate in (("base", "sft"), ("base", "grpo"), ("sft", "grpo")):
        if baseline in payloads and candidate in payloads:
            evidence["paired"][f"{baseline}-vs-{candidate}"] = compare_complete(
                payloads[baseline], payloads[candidate])
    lines = ["# Original SLM comparison", "", f"Measured report: {args.date}.", "",
             "64 fixed questions sampled with seed 42 from all 1,319 GSM8K test questions. "
             "All arms use the pinned Phi-3-mini base, the same base tokenizer and two-shot prompt, "
             "greedy decoding, a 512-token answer limit, and NF4 4-bit loading on the RTX 4060 laptop.", "",
             "| Model | Correct | Accuracy | Wilson 95% interval | Token-cap rate | Answer-marker coverage |",
             "|---|---:|---:|---:|---:|---:|"]
    for arm in ("base", "sft", "grpo"):
        row = evidence["runs"].get(arm)
        if row is None:
            lines.append(f"| {arm.upper()} | Pending | — | — | — | — |")
            continue
        d = row["diagnostics"]
        low, high = d["accuracy_ci95"]
        lines.append(f"| {arm.upper()} | {row['correct']}/{row['total']} | {row['accuracy']:.2%} | "
                     f"{low:.2%}–{high:.2%} | {d['token_cap_rate']:.2%} | {d['marker_rate']:.2%} |")
    lines += ["", "## Paired comparisons", ""]
    for name, pair in evidence["paired"].items():
        baseline, candidate = name.split("-vs-")
        delta = evidence["runs"][candidate]["accuracy"] - evidence["runs"][baseline]["accuracy"]
        lines.append(f"- {candidate.upper()} minus {baseline.upper()}: {delta * 100:+.2f} percentage points; "
                     f"{pair['only_candidate']} candidate-only correct, {pair['only_baseline']} "
                     f"baseline-only correct; exact McNemar p = {pair['p_value']:.6g}.")
    lines += ["", "## Performance on this laptop", ""]
    for arm, row in evidence["runs"].items():
        d = row["diagnostics"]
        lines.append(f"- {arm.upper()}: mean {d['mean_latency_seconds']:.2f} seconds per question, "
                     f"{row['mean_generated_tokens']:.1f} generated tokens; "
                     f"peak PyTorch allocated memory {d['peak_allocated_bytes'] / 2**30:.2f} GiB.")
    verification = ROOT / "docs/evidence/original_grpo_verification.json"
    if verification.exists():
        training = json.loads(verification.read_text(encoding="utf-8"))
        first, last = training["reward_records"][0], training["reward_records"][-1]
        lines += ["", "## Original GRPO training evidence", "",
                  f"Recovered private Kaggle version {training['source_version']} completed "
                  f"{training['global_step']}/{training['max_steps']} optimizer steps. All recorded "
                  "file hashes were independently checked. The adapter has "
                  f"{training['adapter_parameter_count']:,} LoRA parameters.", "",
                  f"Logged outcome reward fell from {first['rewards/outcome_reward']:.3f} to "
                  f"{last['rewards/outcome_reward']:.3f}; mean completion length ended at "
                  f"{last['completion_length']:.0f}, the training cap. Truncation is a plausible "
                  "contributor to falling reward, not an established causal diagnosis. The local "
                  "evaluation above uses a shared 512-token cap, so its scores must not be confused "
                  "with rewards measured during 96-token stochastic training.", ""]
    lines += ["", "## Interpretation and limitations", "",
              "This is a small local evaluation, not the full GSM8K benchmark. It establishes neither "
              "general superiority nor improvement on other tasks. The test sample is used for this "
              "comparison and should not subsequently be used to tune training settings.", "",
              "Final-answer parsing prefers the #### marker and otherwise takes the last number. "
              "Answer-marker coverage is a formatting diagnostic, not a proof of valid reasoning. "
              "Token-cap hits exclude answers ending in EOS at the limit. Latency excludes loading; "
              "GPU memory is PyTorch allocated memory, not total device usage.", ""]
    if "grpo" not in evidence["runs"]:
        lines += ["Original GRPO recovery and evaluation are pending. The later local-final and "
                  "graph-final pilot adapters are not substitutes. The app defaults to preserved "
                  "SFT while the original comparison remains incomplete.", ""]
    elif all(arm in evidence["runs"] for arm in ("base", "sft")):
        trained = max(("sft", "grpo"), key=lambda arm: evidence["runs"][arm]["accuracy"])
        delta = evidence["runs"][trained]["accuracy"] - evidence["runs"]["base"]["accuracy"]
        evidence["selected_trained_demo"] = trained
        lines += [f"The trained demo uses {trained.upper()}, the higher-scoring trained arm on this "
                  "sample (SFT wins a tie). This is a practical demo selection, not evidence of "
                  "general superiority. The base model remains available as an explicit launcher option.", ""]
        if delta <= 0:
            lines += ["Neither trained arm outscored the base on this sample. The project demonstrates "
                      "a working fine-tuning and evaluation pipeline; improved accuracy is not established.",
                      ""]
    live = ROOT / "docs/evidence/live_slm_app.json"
    if live.exists():
        check = json.loads(live.read_text(encoding="utf-8"))
        terminal = check["stream_frames"][-1]
        if check["batch"]["analysis"]["correct"] and terminal.get("analysis", {}).get("correct"):
            lines += ["## Local application verification", "",
                      "The real CUDA adapter served two arithmetic checks through the batch and "
                      "streaming HTTP APIs. Health reported the adapter active; all three measured "
                      "runs and the GUI assets were available. Responses and latency are saved in "
                      "`docs/evidence/live_slm_app.json`. These are integration checks, not benchmark "
                      "samples. Browser layout was not visually inspected because the computer-control "
                      "tool could not initialize.", ""]
    lines += ["Raw completions, sample identities, hashes, exact settings and package versions are "
              "preserved under `artifacts/slm/evaluation`. Concise evidence is committed in "
              "`docs/evidence/slm_comparison.json`. See [the runbook](SLM_RUNBOOK.md) for reproduction.", ""]
    atomic_json(ROOT / "docs/evidence/slm_comparison.json", evidence)
    (ROOT / "docs/SLM_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print("Updated docs/SLM_RESULTS.md and docs/evidence/slm_comparison.json")


if __name__ == "__main__":
    main()
