from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    base_model: str = "microsoft/Phi-3-mini-4k-instruct"
    torch_dtype: str = "float16"
    device_map: str = "auto"
    load_in_4bit: bool = True
    trust_remote_code: bool = False


@dataclass
class EvalTaskConfig:
    dataset: str
    subset: str | None = None
    split: str = "test"
    max_samples: int | None = None
    seed: int = 42
    offset: int = 0


@dataclass
class EvaluationConfig:
    output_dir: str = "results"
    run_name: str = "baseline"
    few_shot: bool = True
    tasks: dict[str, EvalTaskConfig] = field(default_factory=dict)


@dataclass
class SFTConfig:
    output_dir: str = "artifacts/sft"
    final_dir: str = "artifacts/sft-final"
    dataset: str = "gsm8k"
    subset: str = "main"
    train_split: str = "train"
    extra_datasets: list[str] = field(default_factory=list)
    validation_fraction: float = 0.05
    seed: int = 42
    max_seq_length: int = 1024
    num_train_epochs: float = 2.0
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    logging_steps: int = 20
    eval_steps: int = 200
    save_steps: int = 500
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    completion_only: bool = True
    response_template: str = "<|assistant|>"
    gradient_checkpointing: bool = False
    optim: str = "adamw_torch"


@dataclass
class GRPOConfig:
    sft_adapter: str = "artifacts/sft-final"
    output_dir: str = "artifacts/grpo"
    final_dir: str = "artifacts/grpo-final"
    dataset: str = "gsm8k"
    subset: str = "main"
    train_split: str = "train"
    seed: int = 42
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-6
    num_generations: int = 8
    max_prompt_length: int = 512
    max_completion_length: int = 512
    beta: float = 0.04
    temperature: float = 0.9
    logging_steps: int = 10
    save_steps: int = 200
    gradient_checkpointing: bool = False
    optim: str = "adamw_torch"
    use_process_reward: bool = True
    use_diversity_reward: bool = True
    model_prm: str | None = None
    model_prm_scale: float = 1.0
    reward_weights: list[float] | None = None
    curriculum: bool = True
    forgetting_check: bool = True
    forgetting_every_steps: int = 100
    forgetting_task: str = "mmlu"
    forgetting_max_samples: int = 200


@dataclass
class ProjectConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)


def _build(cls, values: dict[str, Any], section: str, path: str | Path, **extra):
    """Construct a config dataclass, naming the file and [section] on bad keys."""
    try:
        return cls(**values, **extra)
    except TypeError as exc:
        raise ValueError(f"Invalid key in [{section}] of {path}: {exc}") from exc


def _validate(cfg: ProjectConfig, path: str | Path) -> None:
    if not 0 < cfg.sft.validation_fraction < 1:
        raise ValueError(
            f"sft.validation_fraction must be in (0, 1), got {cfg.sft.validation_fraction} in {path}"
        )
    if cfg.grpo.num_generations < 2:
        raise ValueError(
            f"grpo.num_generations must be >= 2 (GRPO needs a group), "
            f"got {cfg.grpo.num_generations} in {path}"
        )


def load_config(path: str | Path) -> ProjectConfig:
    with Path(path).open("rb") as fh:
        raw = tomllib.load(fh)

    model = _build(ModelConfig, raw.get("model", {}), "model", path)
    eval_raw = raw.get("evaluation", {})
    tasks = {
        name: _build(EvalTaskConfig, values, f"evaluation.tasks.{name}", path)
        for name, values in eval_raw.pop("tasks", {}).items()
    }
    evaluation = _build(EvaluationConfig, eval_raw, "evaluation", path, tasks=tasks)
    sft = _build(SFTConfig, raw.get("sft", {}), "sft", path)
    grpo = _build(GRPOConfig, raw.get("grpo", {}), "grpo", path)
    cfg = ProjectConfig(model=model, evaluation=evaluation, sft=sft, grpo=grpo)
    _validate(cfg, path)
    return cfg
