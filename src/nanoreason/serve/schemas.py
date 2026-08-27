"""Request and response models for the HTTP API.

These are the contract the single-page app is written against, so they are
declared explicitly rather than left to duck typing: a renamed field here is a
silently blank panel in the browser, and FastAPI's generated OpenAPI schema is
what makes that contract checkable.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=1, description="A completion to verify.")
    gold: str | None = Field(default=None, description="Reference answer, bare or '#### n'.")


class SolveRequest(BaseModel):
    question: str = Field(min_length=1)
    max_new_tokens: int = Field(default=384, ge=16, le=2048)
    gold: str | None = None


class StepModel(BaseModel):
    index: int
    lhs: str
    op: str
    rhs: str
    stated: str
    computed: float | None
    ok: bool
    raw: str
    duplicate: bool


class RewardsModel(BaseModel):
    outcome: float | None
    format: float
    process: float
    diversity: float
    total: float


class AnalysisModel(BaseModel):
    answer: str | None
    answer_source: str | None
    format_ok: bool
    steps: list[StepModel]
    steps_correct: int
    steps_total: int
    steps_unique: int
    token_diversity: float
    rewards: RewardsModel
    gold: str | None
    correct: bool | None


class SolveResponse(BaseModel):
    question: str
    completion: str
    analysis: AnalysisModel
    latency_ms: float
    engine: str


class EngineModel(BaseModel):
    kind: str
    model: str | None
    adapter: str | None
    device: str
    is_demo: bool


class HealthResponse(BaseModel):
    status: str
    engine: EngineModel
    version: str


class TaskStat(BaseModel):
    accuracy: float
    correct: int
    total: int


class RunSummary(BaseModel):
    name: str
    run_name: str
    base_model: str
    adapter: str | None
    created_at_unix: int
    config_path: str
    tasks: dict[str, TaskStat]


class RunsResponse(BaseModel):
    runs: list[RunSummary]


class StatModel(BaseModel):
    accuracy: float
    correct: int
    total: int
    ci_low: float
    ci_high: float


class PairedModel(BaseModel):
    n_both: int
    only_candidate: int
    only_baseline: int
    p_value: float
    significant: bool


class CompareRow(BaseModel):
    task: str
    baseline: StatModel | None
    candidate: StatModel | None
    delta: float | None
    passes: bool | None
    paired: PairedModel | None


class CompareResponse(BaseModel):
    baseline: str
    candidate: str
    min_improvement: float
    rows: list[CompareRow]


class ConfigSummary(BaseModel):
    name: str
    path: str
    base_model: str
    load_in_4bit: bool
    tasks: list[str]
    num_generations: int


class ConfigsResponse(BaseModel):
    configs: list[ConfigSummary]


RunPayload = dict[str, Any]
