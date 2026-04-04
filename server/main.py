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
from fastapi.responses import HTMLResponse, JSONResponse

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
)


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def root() -> HTMLResponse:
    """Landing page — rendered in the HuggingFace Space App tab."""
    html = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Clarus — Healthcare Billing Dispute RL Environment</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           background: #0f172a; color: #e2e8f0; min-height: 100vh;
           display: flex; align-items: center; justify-content: center; padding: 2rem; }
    .card { background: #1e293b; border-radius: 16px; padding: 2.5rem;
            max-width: 680px; width: 100%; box-shadow: 0 25px 50px rgba(0,0,0,0.4); }
    .badge { display: inline-block; background: #22c55e; color: #fff;
             font-size: 0.75rem; font-weight: 700; padding: 3px 10px;
             border-radius: 999px; margin-bottom: 1rem; letter-spacing: 0.05em; }
    h1 { font-size: 2rem; font-weight: 800; color: #f8fafc; margin-bottom: 0.25rem; }
    .subtitle { color: #94a3b8; margin-bottom: 2rem; font-size: 0.95rem; }
    .section-title { font-size: 0.7rem; font-weight: 700; color: #64748b;
                     letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.75rem; }
    .tasks { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.75rem; margin-bottom: 2rem; }
    .task { background: #0f172a; border-radius: 10px; padding: 1rem; text-align: center; }
    .task-name { font-size: 0.78rem; font-weight: 600; color: #38bdf8; margin-bottom: 0.25rem; }
    .task-diff { font-size: 0.7rem; color: #64748b; }
    .easy { color: #22c55e !important; }
    .medium { color: #f59e0b !important; }
    .hard { color: #ef4444 !important; }
    .endpoints { display: flex; flex-direction: column; gap: 0.5rem; margin-bottom: 2rem; }
    .ep { display: flex; align-items: center; gap: 0.75rem; background: #0f172a;
          border-radius: 8px; padding: 0.6rem 1rem; font-size: 0.85rem; }
    .method { font-weight: 700; font-size: 0.72rem; padding: 2px 8px;
              border-radius: 4px; min-width: 44px; text-align: center; }
    .post { background: #1d4ed8; color: #bfdbfe; }
    .get  { background: #15803d; color: #bbf7d0; }
    .path { color: #e2e8f0; font-family: monospace; }
    .desc { color: #64748b; font-size: 0.78rem; margin-left: auto; }
    .btn  { display: block; text-align: center; background: #3b82f6; color: #fff;
            text-decoration: none; padding: 0.85rem; border-radius: 10px;
            font-weight: 700; font-size: 0.95rem; transition: background 0.2s; }
    .btn:hover { background: #2563eb; }
  </style>
</head>
<body>
  <div class="card">
    <div class="badge">● RUNNING</div>
    <h1>Clarus</h1>
    <p class="subtitle">Healthcare Billing Dispute &amp; Patient Advocacy — OpenEnv RL Environment</p>

    <p class="section-title">Tasks</p>
    <div class="tasks">
      <div class="task">
        <div class="task-name">Deductive Liability</div>
        <div class="task-diff easy">Easy</div>
      </div>
      <div class="task">
        <div class="task-name">Abductive Conflict</div>
        <div class="task-diff medium">Medium</div>
      </div>
      <div class="task">
        <div class="task-name">Adversarial Fabrication</div>
        <div class="task-diff hard">Hard</div>
      </div>
    </div>

    <p class="section-title">Endpoints</p>
    <div class="endpoints">
      <div class="ep">
        <span class="method post">POST</span>
        <span class="path">/reset</span>
        <span class="desc">Start a new episode</span>
      </div>
      <div class="ep">
        <span class="method post">POST</span>
        <span class="path">/step</span>
        <span class="desc">Execute one action</span>
      </div>
      <div class="ep">
        <span class="method get">GET</span>
        <span class="path">/state</span>
        <span class="desc">Current environment state</span>
      </div>
      <div class="ep">
        <span class="method get">GET</span>
        <span class="path">/docs</span>
        <span class="desc">Interactive API documentation</span>
      </div>
    </div>

    <a class="btn" href="/docs">Open Interactive API Docs →</a>
  </div>
</body>
</html>
"""
    return HTMLResponse(content=html)


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
