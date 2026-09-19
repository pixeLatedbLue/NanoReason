from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from .config import EvalTaskConfig, load_config
from .data import load_eval_dataset
from .metrics import (
    EvalResult,
    extract_final_letter,
    extract_final_number,
    gold_number,
    numeric_equal,
    result_payload,
    write_json,
)
from .modeling import load_causal_lm
from .prompts import aqua_prompt, gsm8k_prompt, mmlu_prompt, strategyqa_prompt

MAX_NEW_TOKENS = 512


def token_id_sets(tokenizer, labels: list[str]) -> dict[str, set[int]]:
    label_ids: dict[str, set[int]] = {}
    for label in labels:
        ids = set()
        variants = {label, label.lower(), label.upper(), label.capitalize()}
        for value in variants:
            for text in (value, " " + value):
                encoded = tokenizer.encode(text, add_special_tokens=False)
                if len(encoded) == 1:
                    ids.add(encoded[0])
        if not ids:
            raise ValueError(f"No single-token ids found for label {label!r}")
        label_ids[label] = ids
    return label_ids


def next_token_choice(
    tokenizer, model, prompt: str, choices: list[str], ids_by_choice: dict[str, set[int]] | None = None
) -> str:
    if ids_by_choice is None:
        ids_by_choice = token_id_sets(tokenizer, choices)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1]
    scores = {
        choice: max(logits[token_id].item() for token_id in token_ids)
        for choice, token_ids in ids_by_choice.items()
    }
    return max(scores, key=scores.get)


def eval_strategyqa(tokenizer, model, cfg: EvalTaskConfig, few_shot: bool) -> EvalResult:
    data = load_eval_dataset(cfg)
    ids_by_choice = token_id_sets(tokenizer, ["yes", "no"])
    correct = 0
    examples: list[dict[str, Any]] = []
    per_item: list[dict[str, Any]] = []
    for index, sample in enumerate(data):
        answer = sample["answer"]
        if not isinstance(answer, bool):
            raise TypeError(f"strategyqa answer must be bool, got {type(answer)}: {answer!r}")
        pred = next_token_choice(
            tokenizer,
            model,
            strategyqa_prompt(tokenizer, sample["question"], few_shot=few_shot),
            ["yes", "no"],
            ids_by_choice,
        )
        gold = "yes" if answer else "no"
        ok = pred == gold
        correct += int(ok)
        per_item.append({"index": index, "correct": ok})
        if len(examples) < 20:
            examples.append({"index": index, "prediction": pred, "gold": gold, "correct": ok})
    return EvalResult(
        "strategyqa", correct / len(data), correct, len(data), cfg.split, cfg.dataset,
        cfg.subset, cfg.max_samples, cfg.seed, examples, few_shot=few_shot, per_item=per_item,
    )


def eval_mmlu(tokenizer, model, cfg: EvalTaskConfig, few_shot: bool) -> EvalResult:
    """Score MMLU by comparing the A/B/C/D next-token logits.

    Nothing is generated here, so there is no place to put demonstrations:
    ``mmlu_prompt`` takes no few-shot examples. The ``few_shot`` argument is
    therefore discarded rather than threaded through as the other evaluators do,
    and the result records ``few_shot=False`` so the run JSON states the
    condition that actually applied.
    """
    del few_shot
    data = load_eval_dataset(cfg)
    labels = ["A", "B", "C", "D"]
    ids_by_choice = token_id_sets(tokenizer, labels)
    correct = 0
    examples: list[dict[str, Any]] = []
    per_item: list[dict[str, Any]] = []
    for index, sample in enumerate(data):
        pred = next_token_choice(tokenizer, model, mmlu_prompt(tokenizer, sample), labels, ids_by_choice)
        pred_idx = labels.index(pred)
        gold_idx = int(sample["answer"])
        ok = pred_idx == gold_idx
        correct += int(ok)
        per_item.append({"index": index, "correct": ok})
        if len(examples) < 20:
            examples.append({"index": index, "prediction": pred, "gold": labels[gold_idx], "correct": ok})
    return EvalResult(
        "mmlu", correct / len(data), correct, len(data), cfg.split, cfg.dataset,
        cfg.subset, cfg.max_samples, cfg.seed, examples, few_shot=False, per_item=per_item,
    )


def _generate(tokenizer, model, prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)


def eval_gsm8k(tokenizer, model, cfg: EvalTaskConfig, few_shot: bool) -> EvalResult:
    data = load_eval_dataset(cfg)
    correct = 0
    unparsed = 0
    marker_hits = 0
    examples: list[dict[str, Any]] = []
    per_item: list[dict[str, Any]] = []
    for index, sample in enumerate(data):
        prompt = gsm8k_prompt(tokenizer, sample["question"], few_shot=few_shot)
        generated = _generate(tokenizer, model, prompt)
        pred = extract_final_number(generated)
        unparsed += int(pred is None)
        marker_hits += int("####" in generated)
        gold = gold_number(sample["answer"])
        ok = numeric_equal(pred, gold)
        correct += int(ok)
        per_item.append({"index": index, "correct": ok})
        if len(examples) < 20:
            examples.append(
                {"index": index, "prediction": pred, "gold": gold, "correct": ok,
                 "completion": generated[:500]}
            )
    return EvalResult(
        "gsm8k", correct / len(data), correct, len(data), cfg.split, cfg.dataset,
        cfg.subset, cfg.max_samples, cfg.seed, examples, few_shot=few_shot,
        unparsed=unparsed, marker_rate=round(marker_hits / len(data), 4), per_item=per_item,
    )


def eval_aqua(tokenizer, model, cfg: EvalTaskConfig, few_shot: bool) -> EvalResult:
    data = load_eval_dataset(cfg)
    correct = 0
    unparsed = 0
    marker_hits = 0
    examples: list[dict[str, Any]] = []
    per_item: list[dict[str, Any]] = []
    for index, sample in enumerate(data):
        options = "\n".join(str(opt) for opt in sample["options"])
        question = f"{sample['question']}\nOptions:\n{options}"
        generated = _generate(tokenizer, model, aqua_prompt(tokenizer, question, few_shot=few_shot))
        pred = extract_final_letter(generated)
        unparsed += int(pred is None)
        marker_hits += int("####" in generated)
        gold = str(sample["correct"]).upper()
        ok = pred == gold
        correct += int(ok)
        per_item.append({"index": index, "correct": ok})
        if len(examples) < 20:
            examples.append(
                {"index": index, "prediction": pred, "gold": gold, "correct": ok,
                 "completion": generated[:500]}
            )
    return EvalResult(
        "aqua", correct / len(data), correct, len(data), cfg.split, cfg.dataset,
        cfg.subset, cfg.max_samples, cfg.seed, examples, few_shot=few_shot,
        unparsed=unparsed, marker_rate=round(marker_hits / len(data), 4), per_item=per_item,
    )


EVALUATORS = {
    "strategyqa": eval_strategyqa,
    "gsm8k": eval_gsm8k,
    "mmlu": eval_mmlu,
    "aqua": eval_aqua,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate NanoReason models.")
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--adapter", default=None, help="Optional PEFT adapter path.")
    parser.add_argument("--merge-adapter", action="store_true")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--tasks", nargs="*", default=None, choices=sorted(EVALUATORS))
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_name = args.run_name or cfg.evaluation.run_name
    task_names = args.tasks or list(cfg.evaluation.tasks)
    missing = [name for name in task_names if name not in cfg.evaluation.tasks]
    if missing:
        raise ValueError(
            f"Tasks {missing} are not configured in {args.config}; "
            f"available: {sorted(cfg.evaluation.tasks)}"
        )
    tokenizer, model = load_causal_lm(cfg.model, adapter=args.adapter, merge_adapter=args.merge_adapter)

    results = []
    for task_name in task_names:
        task_cfg = cfg.evaluation.tasks[task_name]
        result = EVALUATORS[task_name](tokenizer, model, task_cfg, cfg.evaluation.few_shot)
        results.append(result)
        print(f"{task_name}: {result.accuracy:.4f} ({result.correct}/{result.total})")

    payload = result_payload(
        run_name=run_name,
        base_model=cfg.model.base_model,
        adapter=args.adapter,
        results=results,
        config_path=str(Path(args.config)),
        settings={
            "few_shot": cfg.evaluation.few_shot,
            "load_in_4bit": cfg.model.load_in_4bit,
            "torch_dtype": cfg.model.torch_dtype,
            "max_new_tokens": MAX_NEW_TOKENS,
            "decoding": "greedy",
        },
    )
    output_path = Path(cfg.evaluation.output_dir) / f"{run_name}.json"
    write_json(output_path, payload)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
