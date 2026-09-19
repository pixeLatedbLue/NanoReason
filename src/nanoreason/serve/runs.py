"""Read-only view over the evaluation artifacts in ``results/``.

The dashboard reads the same JSON files ``nanoreason.evaluate`` writes and the
same ones the reporting rules require numbers to come from — there is no second
store and no cache that could drift from the artifacts on disk.

Nothing here computes statistics of its own: intervals and paired tests come
from ``nanoreason.stats``, so the browser, the CLI and any report quote one
implementation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ..compare_results import per_item_by_task
from ..config import load_config
from ..stats import accuracy, paired_comparison, wilson_interval

_EXCLUDED = {"forgetting_probe.json"}


class UnknownRunError(LookupError):
    """Raised when a named run cannot be loaded.

    A dedicated type rather than a bare ``KeyError`` because the API turns this
    into a 400 that names the offending operand: any *incidental* ``KeyError``
    from deeper in the stack (a run whose ``per_item`` records are missing a
    key, say) would otherwise be reported to the user as "unknown run 'index'",
    blaming their query for a corrupt artifact.
    """

    def __init__(self, name: str) -> None:
        super().__init__(f"Unknown run: {name}")
        self.name = name


def runs_dir() -> Path:
    """Resolved at call time, not import time, so tests can repoint it.

    An empty value falls back to the default: compose files routinely pass
    ``NANOREASON_RESULTS=`` for an unset knob, and ``Path("")`` is ``Path(".")``,
    which would silently scan the working directory for stray JSON.
    """
    return Path(os.environ.get("NANOREASON_RESULTS", "").strip() or "results")


def configs_dir() -> Path:
    return Path(os.environ.get("NANOREASON_CONFIGS", "").strip() or "configs")


def _task_stat(entry: Any) -> dict[str, Any] | None:
    """Coerce one ``summary[task]`` entry, or ``None`` when it is not one."""
    if not isinstance(entry, dict):
        return None
    try:
        correct = int(entry.get("correct", 0))
        total = int(entry.get("total", 0))
    except (TypeError, ValueError):
        return None
    return {"accuracy": accuracy(correct, total), "correct": correct, "total": total}


def _summarise(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Reduce a run payload to the dashboard row shape.

    Every field is coerced rather than passed through. Skipping unparseable
    *JSON* is not enough on its own: a file can parse perfectly and still carry
    a half-written ``summary``, and handing that straight to the response model
    turns one bad artifact into a 500 for the whole dashboard — the failure this
    layer exists to prevent.
    """
    summary = payload.get("summary")
    tasks: dict[str, dict[str, Any]] = {}
    if isinstance(summary, dict):
        for task, entry in summary.items():
            stat = _task_stat(entry)
            if stat is not None:
                tasks[str(task)] = stat
    try:
        created = int(payload.get("created_at_unix") or 0)
    except (TypeError, ValueError):
        created = 0
    adapter = payload.get("adapter")
    return {
        "name": name,
        "run_name": str(payload.get("run_name") or name),
        "base_model": str(payload.get("base_model") or ""),
        "adapter": adapter if isinstance(adapter, str) else None,
        "created_at_unix": created,
        "config_path": str(payload.get("config_path") or ""),
        "tasks": tasks,
    }


def list_runs() -> list[dict[str, Any]]:
    """Every readable run artifact, newest first.

    A malformed file is skipped rather than raised: evaluation writes these
    during long GPU jobs, and a half-flushed file must not take the dashboard
    down while a run is still in progress.
    """
    directory = runs_dir()
    if not directory.is_dir():
        return []
    runs: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        if path.name in _EXCLUDED:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            continue
        if not isinstance(payload, dict) or "summary" not in payload:
            continue
        row = _summarise(path.stem, payload)
        if not row["tasks"]:
            continue
        runs.append(row)
    runs.sort(key=lambda r: r["created_at_unix"], reverse=True)
    return runs


def load_run(name: str) -> dict[str, Any] | None:
    """Load one run by stem.

    ``name`` arrives straight from the URL, so it is treated as hostile: any
    separator, parent reference or drive letter is rejected outright, and the
    resolved path is then confirmed to sit inside the results directory before
    anything is read.
    """
    if not name or any(ch in name for ch in ("/", "\\", ":")) or ".." in name:
        return None
    directory = runs_dir()
    path = (directory / f"{name}.json").resolve()
    try:
        if not path.is_relative_to(directory.resolve()):
            return None
    except (OSError, ValueError):
        return None
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _stat(entry: dict[str, Any] | None) -> dict[str, Any] | None:
    if not entry:
        return None
    correct = int(entry.get("correct", 0))
    total = int(entry.get("total", 0))
    low, high = wilson_interval(correct, total)
    return {
        "accuracy": accuracy(correct, total),
        "correct": correct,
        "total": total,
        "ci_low": low,
        "ci_high": high,
    }


def _paired(base_items: Any, cand_items: Any) -> dict[str, Any] | None:
    """Paired McNemar test, degrading to ``None`` on unusable per-item records.

    ``paired_comparison`` subscripts ``index`` and ``correct`` directly, which is
    right for the CLI (a malformed artifact should be loud there). Here the
    absence of a significance test is a missing cell in a table, not a reason to
    fail the request that also carries the accuracies.
    """
    try:
        return paired_comparison(base_items, cand_items)
    except (KeyError, TypeError, AttributeError):
        return None


def compare_runs(baseline: str, candidate: str, min_improvement: float = 0.05) -> dict[str, Any]:
    """Row-per-task comparison with an interval on each accuracy and, where the
    artifacts carry ``per_item``, a paired McNemar test on the difference.

    Raises ``UnknownRunError`` naming the missing side when a run cannot be
    loaded, so the API can turn it into a 400 that lists what is available.
    """
    base_payload = load_run(baseline)
    if base_payload is None:
        raise UnknownRunError(baseline)
    cand_payload = load_run(candidate)
    if cand_payload is None:
        raise UnknownRunError(candidate)

    base_summary = _summarise(baseline, base_payload)["tasks"]
    cand_summary = _summarise(candidate, cand_payload)["tasks"]
    base_items = per_item_by_task(base_payload)
    cand_items = per_item_by_task(cand_payload)

    rows: list[dict[str, Any]] = []
    for task in sorted(set(base_summary) | set(cand_summary)):
        base_stat = _stat(base_summary.get(task))
        cand_stat = _stat(cand_summary.get(task))
        delta: float | None = None
        passes: bool | None = None
        if base_stat and cand_stat:
            delta = cand_stat["accuracy"] - base_stat["accuracy"]
            passes = round(delta, 9) >= round(min_improvement, 9) - 1e-9
        rows.append(
            {
                "task": task,
                "baseline": base_stat,
                "candidate": cand_stat,
                "delta": delta,
                "passes": passes,
                "paired": _paired(base_items.get(task), cand_items.get(task)),
            }
        )

    return {
        "baseline": baseline,
        "candidate": candidate,
        "min_improvement": min_improvement,
        "rows": rows,
    }


def list_configs() -> list[dict[str, Any]]:
    """The shipped experiment configs, so the UI can show what can be run."""
    directory = configs_dir()
    if not directory.is_dir():
        return []
    configs: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.toml")):
        try:
            cfg = load_config(path)
        except Exception:  # noqa: BLE001 - one bad config must not hide the rest
            continue
        configs.append(
            {
                "name": path.stem,
                "path": str(path).replace("\\", "/"),
                "base_model": cfg.model.base_model,
                "load_in_4bit": cfg.model.load_in_4bit,
                "tasks": sorted(cfg.evaluation.tasks),
                "num_generations": cfg.grpo.num_generations,
            }
        )
    return configs
