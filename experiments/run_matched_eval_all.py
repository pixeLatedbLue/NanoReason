"""Run the matched base/SFT/GRPO comparison end to end in one session.

The n=64 comparison could not resolve the deltas it measured: 1.6 and 3.1
points against a sign test that needs roughly 10 at that size. This runs the
same protocol over a larger sample so the result means something either way.

All three arms must run in one process group on one machine. ``compare_complete``
refuses to pair runs whose recorded ``environment`` differs, which is the point:
a baseline measured elsewhere is not a baseline.

Each arm is a separate subprocess so its CUDA context, and the ~2.2 GB of 4-bit
base weights plus adapter, is released before the next arm loads. On a 16 GB T4
loading three in one process is how you run out of memory at arm three, six
hours in.

Arms resume: ``matched_eval`` checkpoints per item, so a killed session
continues rather than restarting.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("base", "sft", "grpo")


def build_commands(
    *,
    config: str,
    source: str,
    output: str,
    samples: int,
    seed: int,
    max_new_tokens: int,
    sft_adapter: str | None,
    grpo_adapter: str | None,
) -> list[list[str]]:
    """One command per arm, base first.

    Base must run first: the trained arms compare themselves against
    ``base.json`` as they finish, so it has to exist by then.
    """
    adapters = {"base": None, "sft": sft_adapter, "grpo": grpo_adapter}
    for arm in ("sft", "grpo"):
        if not adapters[arm]:
            raise ValueError(f"The {arm} arm needs an explicit --{arm}-adapter")

    commands = []
    for arm in ARMS:
        command = [
            sys.executable, "-m", "nanoreason.matched_eval",
            "--arm", arm,
            "--config", config,
            "--source", source,
            "--output", output,
            "--samples", str(samples),
            "--seed", str(seed),
            "--max-new-tokens", str(max_new_tokens),
        ]
        if adapters[arm]:
            command += ["--adapter", adapters[arm]]
        commands.append(command)
    return commands


def ensure_source(source: Path) -> None:
    """Fetch the official GSM8K test split if it is not already present.

    ``download_gsm8k_test.py`` verifies the file against a pinned SHA-256 and
    refuses to proceed on a mismatch, so this cannot silently evaluate against
    different questions than the earlier run did.
    """
    if source.exists():
        print(f"source present: {source}", flush=True)
        return
    print(f"fetching {source}", flush=True)
    subprocess.run(
        [sys.executable, str(ROOT / "experiments" / "download_gsm8k_test.py")],
        check=True,
        cwd=ROOT,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=1319,
                        help="1319 is the whole official GSM8K test split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", default="configs/local_inference.toml")
    parser.add_argument("--source", default="artifacts/slm/data/gsm8k-test.jsonl")
    parser.add_argument("--output", default="artifacts/slm/evaluation-full",
                        help="kept separate from the n=64 run so its evidence survives")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--sft-adapter", required=True)
    parser.add_argument("--grpo-adapter", required=True)
    args = parser.parse_args()

    ensure_source(ROOT / args.source)

    commands = build_commands(
        config=args.config, source=args.source, output=args.output,
        samples=args.samples, seed=args.seed, max_new_tokens=args.max_new_tokens,
        sft_adapter=args.sft_adapter, grpo_adapter=args.grpo_adapter,
    )
    for command in commands:
        arm = command[command.index("--arm") + 1]
        print(f"\n=== {arm} ===\n> {' '.join(command)}", flush=True)
        subprocess.run(command, check=True, cwd=ROOT)

    out = ROOT / args.output
    print(f"\nwrote {out}")
    for name in sorted(p.name for p in out.glob("*.json")):
        print(f"  {name}")
    for name in sorted(p.name for p in (out / "runs").glob("*.json")):
        print(f"  runs/{name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
