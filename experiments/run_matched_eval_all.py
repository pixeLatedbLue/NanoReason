from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("base", "sft", "grpo")


def selected_arms(names: str | None) -> tuple[str, ...]:
    if not names:
        return ARMS
    chosen = {name.strip() for name in names.split(",") if name.strip()}
    unknown = sorted(chosen - set(ARMS))
    if unknown:
        raise ValueError(f"Unknown arm(s): {', '.join(unknown)}")
    return tuple(arm for arm in ARMS if arm in chosen)


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
    arms: tuple[str, ...] = ARMS,
) -> list[list[str]]:
    adapters = {"base": None, "sft": sft_adapter, "grpo": grpo_adapter}
    for arm in arms:
        if arm != "base" and not adapters[arm]:
            raise ValueError(f"The {arm} arm needs an explicit --{arm}-adapter")

    commands = []
    for arm in arms:
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


def verify_adapter(adapter: str, expected_sha256: str | None) -> None:
    if not expected_sha256:
        return
    weights = Path(adapter) / "adapter_model.safetensors"
    with weights.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected_sha256:
        raise ValueError(
            f"{weights} does not match the expected adapter; "
            f"expected {expected_sha256}, found {actual}")
    print(f"adapter verified: {weights} is {actual}", flush=True)


def ensure_source(source: Path) -> None:
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
    parser.add_argument("--arms", default=None,
                        help="comma-separated subset, e.g. base or sft,grpo; "
                             "all three by default")
    parser.add_argument("--sft-adapter", default=None)
    parser.add_argument("--grpo-adapter", default=None)
    parser.add_argument("--sft-sha256", default=None,
                        help="SHA-256 the SFT adapter_model.safetensors must have")
    parser.add_argument("--grpo-sha256", default=None,
                        help="SHA-256 the GRPO adapter_model.safetensors must have")
    args = parser.parse_args()

    arms = selected_arms(args.arms)
    for arm, adapter, expected in (("sft", args.sft_adapter, args.sft_sha256),
                                   ("grpo", args.grpo_adapter, args.grpo_sha256)):
        if arm in arms and adapter:
            verify_adapter(adapter, expected)

    ensure_source(ROOT / args.source)

    commands = build_commands(
        config=args.config, source=args.source, output=args.output,
        samples=args.samples, seed=args.seed, max_new_tokens=args.max_new_tokens,
        sft_adapter=args.sft_adapter, grpo_adapter=args.grpo_adapter,
        arms=arms,
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
