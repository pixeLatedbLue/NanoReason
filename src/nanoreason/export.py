"""Stage 4 export: merge the trained adapter and serve a quantized model.

Training only updates the QLoRA adapter.  For deployment we:

1. load the base model in full precision,
2. attach the trained adapter and ``merge_and_unload`` it into the weights,
3. save the merged FP16 model, and
4. (optionally) reload it with 4-bit NF4 quantization for cheap INT4 inference.

This makes the "merge then requantize for inference" stage of the project a
real, runnable artifact instead of a claim.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from .config import load_config
from .modeling import load_tokenizer, torch_dtype


def merge_adapter(base_model: str, adapter: str, merged_dir: str, dtype: str = "float16") -> str:
    """Merge ``adapter`` into ``base_model`` and save the result to ``merged_dir``."""
    if not Path(adapter).exists():
        raise FileNotFoundError(f"Adapter path does not exist: {adapter}")
    tokenizer = load_tokenizer(adapter)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch_dtype(dtype), device_map="cpu"
    )
    model = PeftModel.from_pretrained(model, adapter)
    model = model.merge_and_unload()
    Path(merged_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    print(f"Merged model saved to {merged_dir}")
    return merged_dir


def load_for_inference(merged_dir: str, load_4bit: bool, dtype: str = "float16"):
    """Load the merged model, optionally quantized to 4-bit NF4 for inference."""
    quant = None
    if load_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype(dtype),
            bnb_4bit_use_double_quant=True,
        )
    tokenizer = load_tokenizer(merged_dir)
    model = AutoModelForCausalLM.from_pretrained(
        merged_dir,
        torch_dtype=torch_dtype(dtype),
        quantization_config=quant,
        device_map="auto",
    )
    model.eval()
    return tokenizer, model


def generate(tokenizer, model, prompt: str, max_new_tokens: int = 256) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge adapter and run quantized inference.")
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--adapter", default=None, help="Adapter path (defaults to grpo final_dir).")
    parser.add_argument("--merged-dir", default="artifacts/merged")
    parser.add_argument("--skip-merge", action="store_true", help="Reuse an existing merged dir.")
    parser.add_argument("--load-4bit", action="store_true", help="Quantize to INT4 for inference.")
    parser.add_argument("--prompt", default=None, help="Optional prompt to test generation.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    adapter = args.adapter or cfg.grpo.final_dir

    if not args.skip_merge:
        merge_adapter(cfg.model.base_model, adapter, args.merged_dir, cfg.model.torch_dtype)

    if args.prompt is not None:
        tokenizer, model = load_for_inference(args.merged_dir, args.load_4bit, cfg.model.torch_dtype)
        print(generate(tokenizer, model, args.prompt))


if __name__ == "__main__":
    main()
