"""Inference engines behind the web app.

Two implementations sit behind one interface. ``TransformersEngine`` is the real
thing: it loads the base model plus a trained adapter and decodes greedily.
``DemoEngine`` loads nothing at all and emits a scripted worked solution.

The demo engine exists because the interesting half of this project — the step
verifier that produces the process reward — is pure arithmetic and needs no GPU,
no checkpoint and no network. Without it the web app would be undemonstrable on
any machine that has not first spent an hour downloading Phi-3, which is most
machines.

The line the demo engine must not cross is scoring. It fabricates *text*; every
number rendered about that text still comes from the real verifier and the real
reward functions, and ``info().is_demo`` is True everywhere it surfaces, so a
viewer can never mistake its output for a model's.
"""

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
    """What is actually answering, surfaced to the UI so it can say so."""

    kind: str
    model: str | None
    adapter: str | None
    device: str
    is_demo: bool


class ReasoningEngine:
    """Streaming text interface shared by the real and the scripted engine."""

    def info(self) -> EngineInfo:  # pragma: no cover - overridden
        raise NotImplementedError

    def stream(
        self, question: str, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    ) -> Iterator[str]:  # pragma: no cover - overridden
        raise NotImplementedError

    def solve(self, question: str, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS) -> str:
        """Collect the stream into one string.

        The batch route is defined *as* the stream so the two can never diverge:
        if they could, the reward breakdown shown beside a streamed answer would
        be describing a different answer than the one on screen.
        """
        return "".join(self.stream(question, max_new_tokens))


TRAP_MARKER = "trapme"

_INT_RE = re.compile(r"\d+")
_MAX_OPERAND = 9999
_CHUNK_RE = re.compile(r"\s*\S+|\s+")


def _demo_operands(question: str) -> tuple[int, int, int]:
    """Three positive operands derived deterministically from the question.

    Integers actually present in the question are preferred so the worked chain
    visibly talks about what was asked. But the UI must stay demoable on *any*
    input — "Why is the sky blue?" has to produce a chain too — so a shortfall is
    filled from the question's own shape. Nothing here consults a RNG, a clock or
    the environment, which is what makes the same question yield the same bytes
    on every call and every machine.
    """
    values = [
        value
        for value in (int(token) for token in _INT_RE.findall(question))
        if 0 < value <= _MAX_OPERAND
    ]
    while len(values) < 3:
        values.append((len(question.split()) + 3 * (len(values) + 1)) % 97 + 2)
    return values[0], values[1], values[2]


def _demo_script(question: str) -> str:
    """The scripted chain of thought for ``question``.

    The arithmetic is real: every stated result is computed rather than canned,
    so the verifier lights up green on genuine (if trivial) work. Three distinct
    steps is exactly ``rewards._SUBSTANCE_STEPS``, so a clean demo run shows the
    process reward at its ceiling instead of at some arbitrary partial value.

    When the question contains ``trapme`` the *middle* step's stated result is
    corrupted and the final step continues from the corrupted value. That models
    the failure the process reward exists to punish — one slip, silently
    propagated — and it leaves exactly one step that fails verification, because
    the last step is still internally consistent with the wrong number it was
    handed. Only the arithmetic check can catch it: the format is fine and the
    answer parser is perfectly happy.
    """
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
    """Deterministic scripted solver: no model, no network, no randomness."""

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
    """A stopping criterion that flips when the consumer walks away.

    Duck-typed against ``transformers.StoppingCriteria`` so this module can stay
    free of a top-level transformers import; ``StoppingCriteriaList`` only calls
    its members, it never isinstance-checks them.
    """

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
    """Real greedy decoding from the base model plus an optional adapter.

    Construction is deliberately free — no imports, no config read, no weights —
    so importing the app, running the tests, or hitting ``/api/health`` never
    pulls several gigabytes onto a machine that only wanted to see the page.
    """

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

    def _settings(self):
        """The parsed TOML, read at most once. Cheap: no weights are involved."""
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
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        cancel = _CancelSignal()

        def _generate() -> None:
            with torch.no_grad():
                model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    streamer=streamer,
                    stopping_criteria=StoppingCriteriaList([cancel]),
                )

        thread = Thread(target=_generate, daemon=True)
        thread.start()
        try:
            yield from streamer
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
    """Process-wide engine, built once (loading a model per request is fatal)."""
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = _build_engine()
    return _ENGINE


def reset_engine() -> None:
    """Drop the cached engine so a test can flip the environment between cases."""
    global _ENGINE
    _ENGINE = None
