"""Resumable, fixed-sample GSM8K evaluation of existing base/SFT/GRPO models."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

from .config import load_config
from .metrics import (
    EvalResult,
    environment_info,
    extract_final_number,
    gold_number,
    numeric_equal,
    result_payload,
)
from .stats import paired_comparison, wilson_interval

SOURCE_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
REPO_ROOT = Path(__file__).resolve().parents[2]


def adapter_label(adapter: Path | None) -> str | None:
    """How an adapter is named in recorded provenance.

    Relative to the repository when it lives inside it, absolute otherwise. The
    label is only a name: integrity comes from ``adapter_files_sha256``, which
    hashes the adapter's actual contents. An absolute path here would publish
    the operator's home directory in committed evidence files and buy nothing
    in verifiability.
    """
    if adapter is None:
        return None
    try:
        return adapter.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(adapter)


def reset_peak_memory() -> None:
    """Start the peak-memory probe over, where there is a GPU to probe.

    torch.cuda.reset_peak_memory_stats raises AttributeError on a CPU-only
    build of torch rather than returning harmlessly, so the guard is needed to
    run this protocol on CPU at all. Running it on CPU with a tiny model is how
    a wrong flag gets caught before it wastes a multi-hour GPU session.
    """
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_allocated_bytes() -> int:
    """Peak GPU bytes allocated so far, or zero when running on CPU."""
    import torch

    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.max_memory_allocated())


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def prepare_sample(source: Path, destination: Path, size: int = 64, seed: int = 42) -> dict:
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not 0 < size <= len(rows):
        raise ValueError("Sample size must be positive and no larger than the source split")
    indices = random.Random(seed).sample(range(len(rows)), size)
    sample = {
        "dataset": "gsm8k", "subset": "main", "split": "test", "source_url": SOURCE_URL,
        "source_sha256": digest(source), "population": len(rows), "seed": seed,
        "selection": "Python random.Random(seed).sample across the complete source split",
        "rows": [{"index": index, "question": rows[index]["question"],
                  "answer": rows[index]["answer"]} for index in indices],
    }
    if destination.exists():
        if json.loads(destination.read_text(encoding="utf-8")) != sample:
            raise ValueError("Existing sample differs; use another output directory, do not overwrite it")
    else:
        atomic_json(destination, sample)
    return sample


def resume_items(path: Path, identity: dict, sample: list[dict]) -> list[dict]:
    if not path.exists():
        return []
    saved = json.loads(path.read_text(encoding="utf-8"))
    if saved["identity"] != identity:
        raise ValueError("Run settings or checkpoint changed; refusing to mix results")
    rows = saved["items"]
    if len(rows) > len(sample) or [r["index"] for r in rows] != [r["index"] for r in sample[:len(rows)]]:
        raise ValueError("Progress is not an ordered prefix of the fixed sample")
    return rows


def compare_complete(baseline: dict, candidate: dict) -> dict:
    for key in ("sample_sha256", "protocol", "prompt_ids_sha256", "environment"):
        if baseline["provenance"][key] != candidate["provenance"][key]:
            raise ValueError(f"Cannot pair evaluations with different {key}")
    a = baseline["results"][0]["per_item"]
    b = candidate["results"][0]["per_item"]
    if [r["index"] for r in a] != [r["index"] for r in b]:
        raise ValueError("Cannot pair different sample indices")
    return paired_comparison(a, b)


def run(args) -> None:
    import torch

    from .modeling import load_causal_lm, load_tokenizer
    from .prompts import gsm8k_prompt

    root = Path(args.output)
    sample_path = root / "sample.json"
    sample = prepare_sample(Path(args.source), sample_path, args.samples, args.seed)
    cfg = load_config(args.config)
    if not cfg.model.revision:
        raise ValueError("A pinned base model revision is required")
    adapter = Path(args.adapter).resolve() if args.adapter else None
    if (args.arm == "base") != (adapter is None):
        raise ValueError("Base must have no adapter; SFT and GRPO require an explicit adapter")
    adapter_files = {}
    if adapter:
        for required in ("adapter_config.json", "adapter_model.safetensors"):
            if not (adapter / required).is_file():
                raise FileNotFoundError(adapter / required)
        adapter_files = {p.name: digest(p) for p in sorted(adapter.iterdir())
                         if p.is_file() and p.suffix in {".json", ".model", ".safetensors"}}

    protocol = {
        "model": asdict(cfg.model), "max_new_tokens": args.max_new_tokens,
        "few_shot": True, "decoding": "greedy", "num_beams": 1,
        "tokenizer": "pinned base tokenizer for every arm",
        "code_sha256": {name: digest(Path(__file__).with_name(name)) for name in
                        ("matched_eval.py", "prompts.py", "modeling.py", "metrics.py")},
    }
    # Render every arm with exactly the same base tokenizer, independent of adapter packaging.
    tokenizer = load_tokenizer(cfg.model.base_model, revision=cfg.model.revision)
    prompts = [gsm8k_prompt(tokenizer, row["question"], few_shot=True) for row in sample["rows"]]
    token_ids = [tokenizer(prompt)["input_ids"] for prompt in prompts]
    identity = {
        "arm": args.arm, "adapter": adapter_label(adapter),
        "adapter_files_sha256": adapter_files, "sample_sha256": digest(sample_path),
        "protocol": protocol, "environment": environment_info(),
        "prompt_ids_sha256": hashlib.sha256(json.dumps(token_ids).encode()).hexdigest(),
    }
    progress_path = root / "progress" / f"{args.arm}.json"
    completed = resume_items(progress_path, identity, sample["rows"])
    if len(completed) < len(prompts):
        print(f"Loading {args.arm}; resuming at {len(completed)}/{len(prompts)}", flush=True)
        _, model = load_causal_lm(cfg.model, adapter=str(adapter) if adapter else None)
        reset_peak_memory()
        for offset in range(len(completed), len(prompts)):
            row = sample["rows"][offset]
            inputs = tokenizer(prompts[offset], return_tensors="pt").to(model.device)
            if inputs["input_ids"].shape[-1] + args.max_new_tokens > model.config.max_position_embeddings:
                raise ValueError(f"Sample {row['index']} exceeds the model context")
            started = time.perf_counter()
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                        do_sample=False, num_beams=1, use_cache=True,
                                        pad_token_id=tokenizer.eos_token_id)
            generated = output[0][inputs["input_ids"].shape[-1]:].tolist()
            text = tokenizer.decode(generated, skip_special_tokens=True)
            prediction, gold = extract_final_number(text), gold_number(row["answer"])
            eos = model.generation_config.eos_token_id
            eos_ids = eos if isinstance(eos, list) else [eos]
            completed.append({
                **row, "completion": text, "prediction": prediction, "gold": gold,
                "correct": numeric_equal(prediction, gold), "generated_tokens": len(generated),
                "token_cap_hit": len(generated) >= args.max_new_tokens and generated[-1] not in eos_ids,
                "has_answer_marker": "####" in text, "latency_seconds": time.perf_counter() - started,
                "peak_allocated_bytes": peak_allocated_bytes(),
            })
            atomic_json(progress_path, {"identity": identity, "items": completed})
            print(f"{args.arm}: {len(completed)}/{len(prompts)}, "
                  f"correct={sum(r['correct'] for r in completed)}", flush=True)

    n = len(completed)
    correct = sum(r["correct"] for r in completed)
    result = EvalResult(
        task="gsm8k", accuracy=correct/n, correct=correct, total=n, split="test",
        dataset="gsm8k", subset="main", max_samples=n, seed=args.seed,
        examples=completed[:20], few_shot=True,
        unparsed=sum(r["prediction"] is None for r in completed),
        marker_rate=sum(r["has_answer_marker"] for r in completed)/n, per_item=completed,
    )
    payload = result_payload(run_name=args.arm, base_model=cfg.model.base_model,
                             adapter=adapter_label(adapter), results=[result],
                             config_path=args.config, settings=protocol)
    payload["provenance"] = identity
    payload["diagnostics"] = {
        "accuracy_ci95": wilson_interval(correct, n),
        "token_cap_count": sum(r["token_cap_hit"] for r in completed),
        "token_cap_rate": sum(r["token_cap_hit"] for r in completed)/n,
        "marker_rate": result.marker_rate,
        "mean_latency_seconds": sum(r["latency_seconds"] for r in completed)/n,
        "peak_allocated_bytes": max(r["peak_allocated_bytes"] for r in completed),
    }
    atomic_json(root / "runs" / f"{args.arm}.json", payload)
    baseline_path = root / "runs/base.json"
    if args.arm != "base" and baseline_path.exists():
        comparison = compare_complete(json.loads(baseline_path.read_text(encoding="utf-8")), payload)
        atomic_json(root / f"base-vs-{args.arm}.json", comparison)
    print(json.dumps({"arm": args.arm, "correct": correct, "total": n,
                      "diagnostics": payload["diagnostics"]}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=["base", "sft", "grpo"])
    parser.add_argument("--adapter")
    parser.add_argument("--config", default="configs/local_inference.toml")
    parser.add_argument("--source", default="artifacts/slm/data/gsm8k-test.jsonl")
    parser.add_argument("--output", default="artifacts/slm/evaluation")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, choices=range(16, 1025), default=512)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
