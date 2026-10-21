"""HTTP API for the product-attribute extractor.

    SERVE_BACKEND=replay uv run uvicorn pslm.serve.api:app --port 8000                     # no model (CI, Docker)
    SERVE_BACKEND=hf SERVE_MODEL=outputs/qwen2.5-1.5b-lora/adapter uv run --extra train uvicorn pslm.serve.api:app
    SERVE_BACKEND=vllm SERVE_MODEL=/models/adapter uvicorn pslm.serve.api:app              # GPU (Dockerfile.gpu)

Endpoints:
* `POST /extract {"titles": [...]}`: one result per title, in order. A bad title gets `attributes: null` plus an
  `error`, and never fails the whole batch.
* `GET /health`: backend and model; 503 until the model is loaded.
* `GET /metrics`: request/title/error counters, latency percentiles, throughput.

Batching: a request's titles go to the backend in one call. With vLLM that's one `generate` over the whole list,
which its scheduler batches on the GPU. Concurrent *requests* are serialized by a lock (vLLM's offline `LLM`
object isn't thread-safe). At higher concurrency you'd run vLLM's async engine or its OpenAI-compatible server
behind this API, which batches across requests too (docs/INTERVIEW.md: serving).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import Counter, deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from pslm.config import settings
from pslm.serve.backends import Backend, build_backend

log = logging.getLogger("pslm.serve")


class ExtractRequest(BaseModel):
    titles: list[str] = Field(min_length=1, max_length=settings.serve_max_batch)

    @field_validator("titles")
    @classmethod
    def _titles_reasonable(cls, v: list[str]) -> list[str]:
        cleaned = [" ".join(t.split()) for t in v]
        if any(not t for t in cleaned):
            raise ValueError("titles must be non-empty")
        if any(len(t) > 500 for t in cleaned):
            raise ValueError("titles longer than 500 characters are not product titles")
        return cleaned


class Result(BaseModel):
    title: str
    attributes: dict | None
    error: str | None = None


class ExtractResponse(BaseModel):
    results: list[Result]
    model: str
    latency_ms: float


class Metrics:
    def __init__(self, window: int = 1000):
        self.lock = threading.Lock()
        self.requests = 0
        self.titles = 0
        self.errors: Counter = Counter()
        self.batch_ms: deque[float] = deque(maxlen=window)
        self.per_title_ms: deque[float] = deque(maxlen=window)
        self.busy_ms = 0.0
        self.started = time.time()

    def record(self, n: int, ms: float, errors: list[str | None]) -> None:
        with self.lock:
            self.requests += 1
            self.titles += n
            self.errors.update(e for e in errors if e)
            self.batch_ms.append(ms)
            self.per_title_ms.append(ms / n)
            self.busy_ms += ms

    @staticmethod
    def _pct(xs, q: float) -> float | None:
        if not xs:
            return None
        s = sorted(xs)
        return round(s[min(len(s) - 1, int(q * len(s)))], 1)

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "requests": self.requests,
                "titles": self.titles,
                "errors": dict(self.errors),
                "batch_latency_ms": {
                    "p50": self._pct(self.batch_ms, 0.5),
                    "p95": self._pct(self.batch_ms, 0.95),
                },
                "per_title_ms": {
                    "p50": self._pct(self.per_title_ms, 0.5),
                    "p95": self._pct(self.per_title_ms, 0.95),
                },
                # titles per second of backend time (excludes idle), i.e. what the model sustains
                "titles_per_s": round(self.titles / (self.busy_ms / 1000), 1) if self.busy_ms else None,
                "uptime_s": round(time.time() - self.started, 1),
            }


def create_app(backend: Backend | None = None) -> FastAPI:
    """Tests inject a backend; otherwise one is built from SERVE_* settings when the app starts."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.backend is None:
            app.state.backend = build_backend(
                settings.serve_backend, settings.serve_model, settings.serve_guided
            )
        log.info("backend=%s model=%s", app.state.backend.name, app.state.backend.model)
        yield

    app = FastAPI(title="product-slm", version="0.1.0", lifespan=lifespan)
    app.state.backend = backend
    app.state.lock = threading.Lock()
    app.state.metrics = Metrics()

    @app.get("/health")
    def health(request: Request) -> dict:
        b = request.app.state.backend
        if b is None:
            raise HTTPException(503, "model not loaded")
        return {"status": "ok", "backend": b.name, "model": b.model}

    @app.get("/metrics")
    def metrics(request: Request) -> dict:
        return request.app.state.metrics.snapshot()

    @app.post("/extract", response_model=ExtractResponse)
    def extract(req: ExtractRequest, request: Request) -> ExtractResponse:
        state = request.app.state
        if state.backend is None:
            raise HTTPException(503, "model not loaded")
        with state.lock:
            out = state.backend.extract(req.titles)
        errors = [e.error for e in out.extractions]
        state.metrics.record(len(req.titles), out.latency_ms, errors)
        log.info(
            json.dumps(
                {
                    "event": "extract",
                    "n": len(req.titles),
                    "latency_ms": out.latency_ms,
                    "errors": sum(e is not None for e in errors),
                }
            )
        )
        return ExtractResponse(
            results=[Result(**e.public()) for e in out.extractions],
            model=out.model,
            latency_ms=out.latency_ms,
        )

    return app


app = create_app()
