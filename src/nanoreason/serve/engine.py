from __future__ import annotations

import os
import re
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MAX_NEW_TOKENS = 384


@dataclass
class EngineInfo:

    kind: str
    model: str | None
    adapter: str | None
    device: str
    is_demo: bool
    adapter_active: bool = False


class ReasoningEngine:

    def load(self) -> None:
        pass

    def info(self) -> EngineInfo:  # pragma: no cover - overridden
        raise NotImplementedError

    def stream(
        self, question: str, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    ) -> Iterator[str]:  # pragma: no cover - overridden
        raise NotImplementedError

    def solve(self, question: str, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS) -> str:
        return "".join(self.stream(question, max_new_tokens))

TRAP_MARKER = "trapme"

_INT_RE = re.compile(r"\d+")
_MAX_OPERAND = 9999
_CHUNK_RE = re.compile(r"\s*\S+|\s+")


def _demo_operands(question: str) -> tuple[int, int, int]:
    values = [
        value
        for value in (int(token) for token in _INT_RE.findall(question))
        if 0 < value <= _MAX_OPERAND
    ]
    while len(values) < 3:
        values.append((len(question.split()) + 3 * (len(values) + 1)) % 97 + 2)
    return values[0], values[1], values[2]


def _demo_script(question: str) -> str:
    a, b, c = _demo_operands(question)
    first = a + b
    scaled = first * c

    if TRAP_MARKER in question.lower():
        stated_scaled = scaled + max(2, abs(scaled) // 3 + 1)
    else:
        stated_scaled = scaled
    final = stated_scaled - a

    return (
        f"The quantities in play are {a}, {b} and {c}.\n"
        f"Start by putting the first two together.\n"
        f"{a} + {b} = {first}\n"
        f"Now scale that running subtotal by the third quantity.\n"
        f"{first} * {c} = {stated_scaled}\n"
        f"Then take back out the {a} that was already counted at the start.\n"
        f"{stated_scaled} - {a} = {final}\n"
        f"That leaves {final} as the result.\n"
        f"#### {final}"
    )


class DemoEngine(ReasoningEngine):

    def __init__(self, delay: float = 0.0):
        self.delay = delay

    def info(self) -> EngineInfo:
        return EngineInfo(kind="demo", model=None, adapter=None, device="cpu", is_demo=True)

    def stream(
        self, question: str, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    ) -> Iterator[str]:
        for chunk in _CHUNK_RE.findall(_demo_script(question)):
            if self.delay:
                time.sleep(self.delay)
            yield chunk


class _CancelSignal:

    def __init__(self) -> None:
        self._event = threading.Event()

    def set(self) -> None:
        self._event.set()

    def __call__(self, input_ids, scores, **kwargs):
        import torch

        return torch.full(
            (input_ids.shape[0],), self._event.is_set(), dtype=torch.bool, device=input_ids.device
        )


class TransformersEngine(ReasoningEngine):

    def __init__(
        self,
        config_path: str = "configs/default.toml",
        adapter: str | None = None,
        few_shot: bool = True,
    ):
        self.config_path = config_path
        self.adapter = adapter
        self.few_shot = few_shot
        self._config = None
        self._tokenizer = None
        self._model = None
        self._load_lock = threading.Lock()
        self._generation_lock = threading.Lock()

    def load(self) -> None:
        self._ensure_loaded()

    def _settings(self):
        if self._config is None:
            from ..config import load_config

            self._config = load_config(self.config_path)
        return self._config

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            from ..modeling import load_causal_lm

            self._tokenizer, self._model = load_causal_lm(
                self._settings().model, adapter=self.adapter
            )

    def info(self) -> EngineInfo:
        if self._model is not None:
            model_name = getattr(self._model.config, "_name_or_path", None)
            device = str(next(self._model.parameters()).device)
        else:
            device = "unloaded"
            try:
                model_name = self._settings().model.base_model
            except (OSError, ValueError):
                model_name = None
        return EngineInfo(
            kind="transformers",
            model=model_name,
            adapter=self.adapter,
            device=device,
            is_demo=False,
            adapter_active=bool(
                self._model is not None and self.adapter and getattr(self._model, "active_adapters", [])
            ),
        )

    def stream(
        self, question: str, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    ) -> Iterator[str]:
        from threading import Thread

        import torch
        from transformers import StoppingCriteriaList, TextIteratorStreamer

        from ..prompts import gsm8k_prompt

        self._ensure_loaded()
        tokenizer, model = self._tokenizer, self._model
        prompt = gsm8k_prompt(tokenizer, question, few_shot=self.few_shot)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        context_limit = getattr(getattr(model, "config", None), "max_position_embeddings", 4096)
        if inputs["input_ids"].shape[-1] + max_new_tokens > context_limit:
            raise ValueError(
                f"Question and requested answer exceed the model's {context_limit}-token context. "
                "Shorten the question or lower Max new tokens."
            )
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        cancel = _CancelSignal()
        errors: list[Exception] = []

        if not self._generation_lock.acquire(timeout=30):
            raise RuntimeError("The GPU is busy answering another question; retry when it finishes.")

        def _generate() -> None:
            try:
                with torch.no_grad():
                    model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                        streamer=streamer,
                        stopping_criteria=StoppingCriteriaList([cancel]),
                    )
            except Exception as exc:
                errors.append(exc)
            finally:
                streamer.end()
                self._generation_lock.release()

        thread = Thread(target=_generate, daemon=True)
        try:
            thread.start()
        except Exception:
            self._generation_lock.release()
            raise
        try:
            yield from streamer
            if errors:
                raise RuntimeError(f"Model generation failed: {errors[0]}") from errors[0]
        finally:
            cancel.set()
            thread.join(timeout=5.0)

_ENGINE: ReasoningEngine | None = None


def _build_engine() -> ReasoningEngine:
    kind = os.environ.get("NANOREASON_ENGINE", "auto").strip().lower()
    config_path = os.environ.get("NANOREASON_CONFIG", "configs/default.toml")
    adapter = os.environ.get("NANOREASON_ADAPTER") or None

    if kind == "demo":
        return DemoEngine()
    if kind == "transformers":
        return TransformersEngine(config_path, adapter)

    ready = os.environ.get("NANOREASON_MODEL_READY") == "1"
    if ready or (adapter and Path(adapter).exists()):
        return TransformersEngine(config_path, adapter)
    return DemoEngine()


def get_engine() -> ReasoningEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = _build_engine()
    return _ENGINE


def reset_engine() -> None:
    global _ENGINE
    _ENGINE = None
