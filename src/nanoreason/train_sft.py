from __future__ import annotations

import argparse
import inspect
import random
from typing import TYPE_CHECKING

import numpy as np
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM
from trl import SFTConfig as TRLSFTConfig
from trl import SFTTrainer

from .config import load_config
from .data import load_combined_training
from .modeling import load_tokenizer, quantization_config, torch_dtype
from .prompts import GSM8K_SYSTEM, chat_prompt

if TYPE_CHECKING:
    from trl import DataCollatorForCompletionOnlyLM

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_example(tokenizer, example: dict) -> dict[str, str]:
    messages = [
        {"role": "system", "content": GSM8K_SYSTEM},
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["answer"]},
    ]
    return {"text": chat_prompt(tokenizer, messages, add_generation_prompt=False)}


def build_completion_collator(
    tokenizer, response_template: str
) -> DataCollatorForCompletionOnlyLM:
    """Collator that masks the loss on everything before the assistant turn.

    Ensures SFT trains only on the reasoning + answer tokens, matching the
    project design ("loss is computed solely on the reasoning and answer
    tokens"). The template is passed as token ids (trl's recommendation) so
    context-dependent tokenization cannot silently break the match. Note trl's
    behavior when the template is missing from an example (e.g. truncated away
    by max_seq_length): that example's labels are fully masked with a
    UserWarning — it contributes zero loss, there is no LM-loss fallback.
    """
    from trl import DataCollatorForCompletionOnlyLM

    template_ids = tokenizer.encode(response_template, add_special_tokens=False)
    return DataCollatorForCompletionOnlyLM(template_ids, tokenizer=tokenizer)


def _assert_template_present(tokenizer, response_template: str) -> None:
    """Fail fast if the response template can't be found in a formatted sample.

    If the template never matches, the collator would mask 100% of examples and
    the run would 'succeed' while learning nothing.
    """
    sample = format_example(tokenizer, {"question": "check", "answer": "check"})
    text_ids = tokenizer.encode(sample["text"], add_special_tokens=False)
    template_ids = tokenizer.encode(response_template, add_special_tokens=False)
    n, m = len(text_ids), len(template_ids)
    found = any(text_ids[i : i + m] == template_ids for i in range(n - m + 1))
    if not found:
        raise ValueError(
            f"response_template {response_template!r} (ids {template_ids}) does not "
            "appear in a chat-formatted sample; completion-only masking would mask "
            "every example. Check sft.response_template against the model's chat template."
        )


def _sft_config(**kwargs):
    fields = getattr(TRLSFTConfig, "__dataclass_fields__", {})
    if "eval_strategy" in fields and "evaluation_strategy" in kwargs:
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")
    elif "evaluation_strategy" not in fields and "evaluation_strategy" in kwargs:
        kwargs.pop("evaluation_strategy")
    return TRLSFTConfig(**kwargs)


def _sft_trainer(**kwargs):
    params = inspect.signature(SFTTrainer.__init__).parameters
    if "processing_class" not in params and "processing_class" in kwargs:
        kwargs["tokenizer"] = kwargs.pop("processing_class")
    return SFTTrainer(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run QLoRA SFT warm-up.")
    parser.add_argument("--config", default="configs/default.toml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.sft.seed)

    tokenizer = load_tokenizer(cfg.model.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.base_model,
        torch_dtype=torch_dtype(cfg.model.torch_dtype),
        quantization_config=quantization_config(cfg.model),
        device_map=cfg.model.device_map,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    if cfg.model.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg.sft.gradient_checkpointing
        )
    elif cfg.sft.gradient_checkpointing:
        model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=cfg.sft.lora_r,
            lora_alpha=cfg.sft.lora_alpha,
            lora_dropout=cfg.sft.lora_dropout,
            target_modules=LORA_TARGETS,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    raw = load_combined_training(
        cfg.sft.dataset, cfg.sft.subset, cfg.sft.train_split, cfg.sft.extra_datasets
    )
    split = raw.train_test_split(test_size=cfg.sft.validation_fraction, seed=cfg.sft.seed)
    train_data = split["train"].map(lambda ex: format_example(tokenizer, ex), remove_columns=raw.column_names)
    valid_data = split["test"].map(lambda ex: format_example(tokenizer, ex), remove_columns=raw.column_names)

    collator = None
    if cfg.sft.completion_only:
        _assert_template_present(tokenizer, cfg.sft.response_template)
        collator = build_completion_collator(tokenizer, cfg.sft.response_template)

    trainer_args = _sft_config(
        output_dir=cfg.sft.output_dir,
        num_train_epochs=cfg.sft.num_train_epochs,
        per_device_train_batch_size=cfg.sft.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.sft.gradient_accumulation_steps,
        learning_rate=cfg.sft.learning_rate,
        warmup_ratio=cfg.sft.warmup_ratio,
        logging_steps=cfg.sft.logging_steps,
        evaluation_strategy="steps",
        eval_steps=cfg.sft.eval_steps,
        save_steps=cfg.sft.save_steps,
        bf16=cfg.model.torch_dtype.lower() in {"bf16", "bfloat16"},
        fp16=cfg.model.torch_dtype.lower() in {"fp16", "float16"},
        max_seq_length=cfg.sft.max_seq_length,
        dataset_text_field="text",
        packing=False,
        gradient_checkpointing=cfg.sft.gradient_checkpointing,
        optim=cfg.sft.optim,
        report_to="none",
        seed=cfg.sft.seed,
    )

    trainer = _sft_trainer(
        model=model,
        args=trainer_args,
        train_dataset=train_data,
        eval_dataset=valid_data,
        data_collator=collator,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(cfg.sft.final_dir)
    tokenizer.save_pretrained(cfg.sft.final_dir)
    print(f"SFT adapter saved to {cfg.sft.final_dir}")


if __name__ == "__main__":
    main()
