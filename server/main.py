"""FastAPI application for Clarus OpenEnv server.

Endpoints:
    POST /reset  — start a new episode
    POST /step   — execute one action
    GET  /state  — current environment state

The server maintains a single global ClarusEnv instance.  The reference
DB is loaded once at startup and shared across all episodes.
"""

from __future__ import annotations

import sqlite3
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from data.setup import load_all
from server.env import ClarusEnv
from server.models import (
    ResetRequest,
    ResetResponse,
    StateResponse,
    StepRequest,
    StepResponse,
)
from server.schema import create_tables


# ------------------------------------------------------------------
# Application state (module-level singletons)
# ------------------------------------------------------------------

_ref_db: sqlite3.Connection | None = None
_env: ClarusEnv | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load reference data and initialise environment at startup."""
    global _ref_db, _env

    _ref_db = sqlite3.connect(":memory:", check_same_thread=False)
    _ref_db.row_factory = sqlite3.Row
    create_tables(_ref_db)
    counts = load_all(_ref_db)

    cpt_count = counts.get("cpt_codes", 0)
    ncci_count = counts.get("ncci_edits", 0)
    print(
        f"[Clarus] Ref DB ready: {cpt_count} CPT codes, {ncci_count} NCCI edits",
        flush=True,
    )

    _env = ClarusEnv(ref_db=_ref_db)
    print("[Clarus] Environment initialised.", flush=True)

    yield

    if _ref_db:
        _ref_db.close()


app = FastAPI(
    title="Clarus",
    description=(
        "Healthcare Billing Dispute & Patient Advocacy OpenEnv environment. "
        "Three tasks: deductive_liability, abductive_conflict, adversarial_fabrication."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------


@app.get("/")
async def root() -> Dict[str, Any]:
    """Health check — returns 200 so submission pings pass."""
    return {"status": "ok", "name": "clarus", "version": "1.0.0"}


@app.post("/reset", response_model=ResetResponse)
async def reset(request: ResetRequest = ResetRequest()) -> ResetResponse:
    """Start a new episode.

    Args:
        request: Optional task_name and seed.  Defaults to deductive_liability
                 with a random train seed.

    Returns:
        ResetResponse with episode_id, task_name, seed, and initial observation.
    """
    if _env is None:
        raise HTTPException(status_code=503, detail="Environment not initialised.")

    obs = await _env.reset(
        task_name=request.task_name,
        seed=request.seed,
    )
    return ResetResponse(
        episode_id=_env.episode_id,
        task_name=_env.task_name,
        seed=_env.seed,
        observation=obs,
    )


@app.post("/step", response_model=StepResponse)
async def step(request: StepRequest) -> StepResponse:
    """Execute one action in the current episode.

    Args:
        request: StepRequest containing the action.

    Returns:
        StepResponse with observation, reward, done, and info.
    """
    if _env is None:
        raise HTTPException(status_code=503, detail="Environment not initialised.")
    if _env.episode_id is None:
        raise HTTPException(
            status_code=400,
            detail="No active episode.  Call /reset first.",
        )

    result = await _env.step(request.action)

    # Convert CheckResult list to serialisable form
    info: Dict[str, Any] = {
        "action_error": result.info.get("action_error"),
        "rate_limited": result.info.get("rate_limited", False),
        "episode_score": result.info.get("episode_score"),
    }
    if result.info.get("check_results") is not None:
        info["check_results"] = [
            cr.model_dump() for cr in result.info["check_results"]
        ]

    return StepResponse(
        observation=result.observation,
        reward=result.reward,
        done=result.done,
        info=info,
    )


@app.get("/state", response_model=StateResponse)
async def state() -> StateResponse:
    """Return the current environment state.

    Returns:
        StateResponse with episode metadata and last observation.
    """
    if _env is None:
        raise HTTPException(status_code=503, detail="Environment not initialised.")

    return _env.state()


@app.get("/health")
async def health() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "service": "clarus"}
