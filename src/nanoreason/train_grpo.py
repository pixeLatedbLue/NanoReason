from __future__ import annotations

import argparse
import inspect
from contextlib import contextmanager

from peft import PeftModel, prepare_model_for_kbit_training
from torch.utils.data import SequentialSampler
from transformers import AutoModelForCausalLM
from trl import GRPOConfig as TRLGRPOConfig
from trl import GRPOTrainer

from .callbacks import ForgettingProbe
from .config import load_config
from .data import load_combined_training, order_by_curriculum
from .modeling import load_tokenizer, quantization_config, torch_dtype
from .prompts import GSM8K_SYSTEM
from .rewards import build_reward_funcs
from .train_sft import set_seed


def to_grpo_example(example: dict) -> dict:
    return {
        "prompt": [
            {"role": "system", "content": GSM8K_SYSTEM},
            {"role": "user", "content": example["question"]},
        ],
        "answer": example["answer"],
    }


def _require_supported_trl() -> None:
    """Hard-stop on trl versions whose GRPO batching would silently break us.

    Verified against trl 0.14.0 source: each batch row is one prompt and the
    trainer generates ``num_generations`` completions per row internally
    (``num_return_sequences``), so a sequential sampler preserves the
    curriculum without touching group formation. trl >= 0.15 rewrote GRPO
    around a repeat-sampler (each prompt repeated G times across the batch,
    with a batch-divisibility requirement); under that scheme a plain
    sequential sampler would form advantage groups from *different* prompts —
    silent corruption, no crash. Refuse to run rather than train garbage.
    """
    import trl
    from packaging import version

    if version.parse(trl.__version__) >= version.parse("0.15.0"):
        raise RuntimeError(
            f"trl {trl.__version__} is not supported by train_grpo: GRPO batching "
            "changed in 0.15 (grouped repeat-sampler) and would silently break the "
            "curriculum sampler and the shipped batch-size configs. Install "
            "'trl>=0.14,<0.15' (the Kaggle-verified pin)."
        )


class CurriculumGRPOTrainer(GRPOTrainer):
    """GRPO trainer that consumes the training set in its given (sorted) order.

    The base ``Trainer`` reshuffles every epoch, which would destroy an
    easy-to-hard curriculum.  Swapping in a sequential sampler preserves the
    ordering produced by :func:`order_by_curriculum`. Safe on trl 0.14.x only
    (see :func:`_require_supported_trl`), where groups are formed per prompt
    row via ``num_return_sequences``, not across the batch dimension.
    """

    def _get_train_sampler(self, *args, **kwargs):
        return SequentialSampler(self.train_dataset)


def _grpo_config(**kwargs) -> TRLGRPOConfig:
    """Build a GRPOConfig, dropping keys unsupported by the installed TRL.

    Dropped keys are reported loudly: a hyperparameter that silently reverts
    to the library default (e.g. ``beta``) would change the whole RL run while
    every log claims the config was applied.
    """
    fields = getattr(TRLGRPOConfig, "__dataclass_fields__", {})
    if fields:
        dropped = {k: v for k, v in kwargs.items() if k not in fields and v is not None}
        if dropped:
            import trl

            print(
                f"[train_grpo] WARNING: settings not supported by trl {trl.__version__} "
                f"and IGNORED: {sorted(dropped)}"
            )
        kwargs = {k: v for k, v in kwargs.items() if k in fields}
    return TRLGRPOConfig(**kwargs)


@contextmanager
def _reuse_loaded_adapter(model):
    """Let TRL 0.14 reuse a loaded adapter without copying the base model.

    A PEFT policy obtains reference logits by disabling its adapter. TRL 0.14
    selects that memory-efficient path only when ``peft_config`` is passed, but
    its normal wrapper would nest a second adapter around an existing PeftModel.
    This compatibility shim makes that wrapper a no-op during construction.
    """
    if not isinstance(model, PeftModel):
        yield None
        return
    import trl.trainer.grpo_trainer as grpo_module

    original = grpo_module.get_peft_model
    grpo_module.get_peft_model = lambda loaded_model, config: loaded_model
    try:
        yield model.peft_config[model.active_adapter]
    finally:
        grpo_module.get_peft_model = original


def _grpo_trainer(curriculum: bool, **kwargs):
    cls = CurriculumGRPOTrainer if curriculum else GRPOTrainer
    params = inspect.signature(GRPOTrainer.__init__).parameters
    if "processing_class" not in params and "processing_class" in kwargs:
        kwargs["tokenizer"] = kwargs.pop("processing_class")
    model = kwargs.get("model")
    with _reuse_loaded_adapter(model) as peft_config:
        if peft_config is not None:
            kwargs["peft_config"] = peft_config
        return cls(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GRPO training from an SFT adapter.")
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Path to an artifacts/grpo/checkpoint-* directory to resume from.",
    )
    args = parser.parse_args()

    _require_supported_trl()
    cfg = load_config(args.config)
    set_seed(cfg.grpo.seed)

    tokenizer = load_tokenizer(cfg.grpo.sft_adapter)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.base_model,
        torch_dtype=torch_dtype(cfg.model.torch_dtype),
        quantization_config=quantization_config(cfg.model),
        device_map=cfg.model.device_map,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    if cfg.model.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    model = PeftModel.from_pretrained(model, cfg.grpo.sft_adapter, is_trainable=True)
    model.config.use_cache = False

    raw = load_combined_training(cfg.grpo.dataset, cfg.grpo.subset, cfg.grpo.train_split)
    if cfg.grpo.curriculum:
        raw = order_by_curriculum(raw, answer_field="answer")
    train_data = raw.map(to_grpo_example, remove_columns=raw.column_names)

    reward_funcs = build_reward_funcs(
        use_process=cfg.grpo.use_process_reward,
        use_diversity=cfg.grpo.use_diversity_reward,
        model_prm=cfg.grpo.model_prm,
        model_prm_scale=cfg.grpo.model_prm_scale,
    )

    trainer_args = _grpo_config(
        output_dir=cfg.grpo.output_dir,
        num_train_epochs=cfg.grpo.num_train_epochs,
        per_device_train_batch_size=cfg.grpo.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.grpo.gradient_accumulation_steps,
        learning_rate=cfg.grpo.learning_rate,
        num_generations=cfg.grpo.num_generations,
        max_prompt_length=cfg.grpo.max_prompt_length,
        max_completion_length=cfg.grpo.max_completion_length,
        beta=cfg.grpo.beta,
        temperature=cfg.grpo.temperature,
        reward_weights=cfg.grpo.reward_weights,
        gradient_checkpointing=cfg.grpo.gradient_checkpointing,
        optim=cfg.grpo.optim,
        logging_steps=cfg.grpo.logging_steps,
        save_steps=cfg.grpo.save_steps,
        bf16=cfg.model.torch_dtype.lower() in {"bf16", "bfloat16"},
        fp16=cfg.model.torch_dtype.lower() in {"fp16", "float16"},
        report_to="none",
        seed=cfg.grpo.seed,
        save_total_limit=2,
    )

    callbacks = []
    if cfg.grpo.forgetting_check:
        callbacks.append(
            ForgettingProbe(
                tokenizer,
                task=cfg.grpo.forgetting_task,
                max_samples=cfg.grpo.forgetting_max_samples,
                every_steps=cfg.grpo.forgetting_every_steps,
                output_dir=cfg.evaluation.output_dir,
                seed=cfg.grpo.seed,
                few_shot=cfg.evaluation.few_shot,
            )
        )

    trainer = _grpo_trainer(
        cfg.grpo.curriculum,
        model=model,
        reward_funcs=reward_funcs,
        args=trainer_args,
        train_dataset=train_data,
        processing_class=tokenizer,
        callbacks=callbacks,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(cfg.grpo.final_dir)
    tokenizer.save_pretrained(cfg.grpo.final_dir)
    print(f"GRPO adapter saved to {cfg.grpo.final_dir}")


if __name__ == "__main__":
    main()
