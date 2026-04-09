"""FastAPI application for Clarus OpenEnv server.

Endpoints:
    POST /reset  — start a new episode
    POST /step   — execute one action
    GET  /state  — current environment state

The server maintains a single global ClarusEnv instance.  The reference
DB is loaded once at startup and shared across all episodes.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

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


@app.get("/", response_class=HTMLResponse)
async def root() -> HTMLResponse:
    """Serve the interactive landing page."""
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/web", response_class=HTMLResponse)
async def web_root() -> HTMLResponse:
    """Alias for root — supports HF Space base_path: /web set by openenv push."""
    return await root()


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

    # episode_score is only set by the grader when done=True (close_case).
    # For non-terminal steps it is None — but the OpenEnv validator checks
    # every /step HTTP response and rejects null as "out of range".
    # Return the Laplace-smoothed terminal score when done, or the
    # in-progress midpoint (0.5) otherwise.  Both are always in (0, 1).
    # check_results are intentionally NOT exposed (would allow passing/total=1.0).
    raw_score = result.info.get("episode_score")
    if raw_score is not None:
        episode_score = max(0.02, min(0.98, float(raw_score)))
    else:
        episode_score = 0.5  # in-progress placeholder — always strictly in (0,1)

    info: Dict[str, Any] = {
        "action_error": result.info.get("action_error"),
        "rate_limited": result.info.get("rate_limited", False),
        "episode_score": episode_score,
    }

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
    return {"status": "healthy", "service": "clarus"}


@app.get("/metadata")
async def metadata() -> Dict[str, Any]:
    """OpenEnv metadata — environment name, description, and task list."""
    return {
        "name": "clarus",
        "description": (
            "Healthcare Billing Dispute & Patient Advocacy OpenEnv environment. "
            "An AI agent vindicates wrongly-billed patients by reconciling multi-party "
            "records, detecting regulatory violations, and holding its position under "
            "adversarial counter-pressure. Graded by SQL checks against real CMS data."
        ),
        "version": "1.0.0",
        "tasks": [
            "deductive_liability",
            "abductive_conflict",
            "adversarial_fabrication",
        ],
        "hf_space": "ismailridwans/clarus",
    }


@app.get("/schema")
async def schema() -> Dict[str, Any]:
    """OpenEnv schema — action, observation, and state JSON schemas."""
    from server.models import ClarusAction, ClarusObservation
    return {
        "action": ClarusAction.model_json_schema(),
        "observation": ClarusObservation.model_json_schema(),
        "state": {
            "type": "object",
            "description": "Current episode state including metadata and last observation.",
            "properties": {
                "episode_id": {"type": ["string", "null"]},
                "task_name": {"type": ["string", "null"]},
                "seed": {"type": ["integer", "null"]},
                "step_number": {"type": "integer"},
                "done": {"type": "boolean"},
            },
        },
    }


@app.post("/mcp")
async def mcp(request: Request) -> Dict[str, Any]:
    """Minimal JSON-RPC 2.0 endpoint for MCP-mode compatibility."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    return {
        "jsonrpc": "2.0",
        "id": body.get("id"),
        "result": {
            "name": "clarus",
            "description": "Healthcare Billing Dispute OpenEnv environment.",
        },
    }
