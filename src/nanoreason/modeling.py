from __future__ import annotations

import importlib.util
import platform
import warnings
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .config import ModelConfig

DTYPES = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def torch_dtype(name: str):
    try:
        return DTYPES[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported torch dtype: {name}") from exc


def quantization_config(config: ModelConfig) -> BitsAndBytesConfig | None:
    if not config.load_in_4bit:
        return None
    if importlib.util.find_spec("bitsandbytes") is None:
        raise RuntimeError(
            "4-bit loading requires bitsandbytes. Install the linux-gpu extra on a supported GPU "
            "machine or set load_in_4bit = false in the config."
        )
    if platform.system() == "Windows":
        warnings.warn(
            "Running bitsandbytes 4-bit on native Windows (supported since 0.43 but "
            "less battle-tested than Linux/WSL2/Colab). If loading fails, run under "
            "WSL2 or set load_in_4bit = false for CPU-only checks.",
            stacklevel=2,
        )
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype(config.torch_dtype),
        bnb_4bit_use_double_quant=True,
    )


def load_tokenizer(path_or_model: str):
    tokenizer = AutoTokenizer.from_pretrained(path_or_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_causal_lm(config: ModelConfig, *, adapter: str | None = None, merge_adapter: bool = False):
    tokenizer_source = adapter if adapter else config.base_model
    tokenizer = load_tokenizer(tokenizer_source)
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=torch_dtype(config.torch_dtype),
        quantization_config=quantization_config(config),
        device_map=config.device_map,
        trust_remote_code=config.trust_remote_code,
    )
    if adapter:
        if not Path(adapter).exists():
            raise FileNotFoundError(f"Adapter path does not exist: {adapter}")
        model = PeftModel.from_pretrained(model, adapter)
        if merge_adapter:
            model = model.merge_and_unload()
    model.eval()
    return tokenizer, model
