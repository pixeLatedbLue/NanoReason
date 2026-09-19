"""The NanoReason HTTP API and the single-page app it serves.

Endpoint handlers are declared ``def``, not ``async def``, on purpose: analysis
and generation are synchronous CPU/GPU work, and an ``async def`` handler doing
blocking work would stall the whole event loop for every other request. FastAPI
runs plain ``def`` handlers in a threadpool, which is the correct shape here.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Iterator
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from . import runs as runs_mod
from .analysis import analyze
from .engine import get_engine
from .schemas import (
    AnalysisModel,
    AnalyzeRequest,
    CompareResponse,
    ConfigsResponse,
    HealthResponse,
    RunsResponse,
    SolveRequest,
    SolveResponse,
)

STATIC_DIR = Path(__file__).parent / "static"


def _sse(payload: dict) -> str:
    """One server-sent-event frame. Separator is a blank line, per the spec."""
    return f"data: {json.dumps(payload)}\n\n"


def create_app() -> FastAPI:
    app = FastAPI(
        title="NanoReason",
        version=__version__,
        description="Serve a QLoRA/GRPO-tuned reasoning model and inspect the reward that trained it.",
    )

    origins = [o.strip() for o in os.environ.get("NANOREASON_CORS", "*").split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health", response_model=HealthResponse)
    def health():
        info = get_engine().info()
        return {
            "status": "ok",
            "engine": {
                "kind": info.kind,
                "model": info.model,
                "adapter": info.adapter,
                "device": info.device,
                "is_demo": info.is_demo,
            },
            "version": __version__,
        }

    @app.post("/api/analyze", response_model=AnalysisModel)
    def analyze_endpoint(req: AnalyzeRequest):
        if len(req.text) > 100_000:
            raise HTTPException(status_code=413, detail="Text too large (limit 100000 characters).")
        return analyze(req.text, req.gold)

    @app.post("/api/solve", response_model=SolveResponse)
    def solve(req: SolveRequest):
        engine = get_engine()
        started = time.perf_counter()
        completion = engine.solve(req.question, req.max_new_tokens)
        latency_ms = (time.perf_counter() - started) * 1000.0
        return {
            "question": req.question,
            "completion": completion,
            "analysis": analyze(completion, req.gold),
            "latency_ms": latency_ms,
            "engine": engine.info().kind,
        }

    @app.get("/api/solve/stream")
    def solve_stream(
        question: str = Query(min_length=1),
        max_new_tokens: int = Query(default=384, ge=16, le=2048),
        gold: str | None = Query(default=None),
    ):
        engine = get_engine()

        def frames() -> Iterator[str]:
            started = time.perf_counter()
            chunks: list[str] = []
            try:
                for chunk in engine.stream(question, max_new_tokens):
                    chunks.append(chunk)
                    yield _sse({"type": "token", "text": chunk})
            except Exception as exc:  # noqa: BLE001 - surfaced to the client, not swallowed
                yield _sse({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
            completion = "".join(chunks)
            try:
                analysis = analyze(completion, gold)
            except Exception as exc:  # noqa: BLE001 - reported, then degraded
                yield _sse({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
                analysis = analyze(completion)
            yield _sse(
                {
                    "type": "done",
                    "completion": completion,
                    "analysis": analysis,
                    "latency_ms": (time.perf_counter() - started) * 1000.0,
                }
            )

        return StreamingResponse(
            frames(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/runs", response_model=RunsResponse)
    def list_runs():
        return {"runs": runs_mod.list_runs()}

    @app.get("/api/runs/{name}")
    def get_run(name: str):
        payload = runs_mod.load_run(name)
        if payload is None:
            raise HTTPException(status_code=404, detail=f"No run named {name!r}.")
        return JSONResponse(payload)

    @app.get("/api/compare", response_model=CompareResponse)
    def compare(
        baseline: str = Query(min_length=1),
        candidate: str = Query(min_length=1),
        min_improvement: float = Query(default=0.05, ge=-1.0, le=1.0),
    ):
        try:
            return runs_mod.compare_runs(baseline, candidate, min_improvement)
        except runs_mod.UnknownRunError as exc:
            available = ", ".join(r["name"] for r in runs_mod.list_runs()) or "(none)"
            raise HTTPException(
                status_code=400,
                detail=f"Unknown run {exc.name!r}. Available runs: {available}.",
            ) from exc

    @app.get("/api/configs", response_model=ConfigsResponse)
    def list_configs():
        return {"configs": runs_mod.list_configs()}

    if STATIC_DIR.is_dir() and any(STATIC_DIR.iterdir()):
        app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
    else:  # pragma: no cover - only hit in a partially built tree

        @app.get("/")
        def index():
            return {"detail": "UI assets are not installed; the API is available under /api."}

    return app


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the NanoReason web app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    import uvicorn

    uvicorn.run("nanoreason.serve.app:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
