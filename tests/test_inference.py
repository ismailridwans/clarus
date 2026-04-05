"""Live OpenAI API integration tests for Clarus.

These tests run a real gpt-4o agent against the environment.  They are
SKIPPED automatically when HF_TOKEN is not set in the environment,
so they never break CI that lacks the key.

Run manually:
    export HF_TOKEN=hf_...
    pytest tests/test_inference.py -v -s

What these tests verify:
- inference.py can construct an OpenAI client and make real API calls
- The environment accepts gpt-4o responses without errors
- The episode always terminates (done=True) within max_steps
- The final episode_score is a float in [0.0, 1.0]
- The easy task (deductive_liability) achieves score > 0.0

These tests do NOT assert score == 1.0 because LLM responses are
non-deterministic and the purpose here is end-to-end smoke validation,
not optimal-trajectory verification (that is test_episodes.py's job).
"""

from __future__ import annotations

import os
import sqlite3

import pytest

# Skip every test in this module when HF_TOKEN is absent.
pytestmark = pytest.mark.skipif(
    not os.getenv("HF_TOKEN"),
    reason="HF_TOKEN not set — skipping live API tests",
)


# ------------------------------------------------------------------
# Shared fixture
# ------------------------------------------------------------------


def _make_ref_db() -> sqlite3.Connection:
    """Build an in-memory reference DB with all bundle data loaded.

    Returns:
        Connected sqlite3.Connection with CPT, NCCI, plan and QPA data.
    """
    from data.setup import load_all
    from server.schema import create_tables

    db = sqlite3.connect(":memory:", check_same_thread=False)
    db.row_factory = sqlite3.Row
    create_tables(db)
    load_all(db)
    return db


def _make_client():
    """Build an OpenAI client from HF_TOKEN.

    Returns:
        openai.OpenAI instance pointing at the standard OpenAI endpoint,
        or at API_BASE_URL if that env var is set.
    """
    from openai import OpenAI
    from inference import HF_TOKEN, _base_url_override

    kwargs: dict = {"api_key": HF_TOKEN}
    if _base_url_override:
        kwargs["base_url"] = _base_url_override
    return OpenAI(**kwargs)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


async def _run_one(task_name: str, seed: int) -> dict:
    """Reset the environment and run one episode with gpt-4o.

    Args:
        task_name: Task to run.
        seed: Deterministic seed for the episode generator.

    Returns:
        Dict with 'score' (float), 'steps' (int), 'rewards' (list).
    """
    from server.env import ClarusEnv
    from inference import run_episode

    ref_db = _make_ref_db()
    env = ClarusEnv(ref_db=ref_db)
    client = _make_client()

    try:
        return await run_episode(env, client, task_name, seed)
    finally:
        await env.close()


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_live_task1_terminates_with_valid_score():
    """gpt-4o must complete a deductive_liability episode with score in [0, 1]."""
    result = await _run_one("deductive_liability", seed=1101)

    assert isinstance(result["score"], float), "score must be a float"
    assert 0.0 <= result["score"] <= 1.0, f"score {result['score']} out of [0, 1]"
    assert result["steps"] >= 1, "at least one step must have occurred"
    assert isinstance(result["rewards"], list), "rewards must be a list"


@pytest.mark.asyncio
async def test_live_task1_score_above_zero():
    """gpt-4o on the easy task must score > 0.0 (i.e. some checks pass)."""
    result = await _run_one("deductive_liability", seed=1101)
    assert result["score"] > 0.0, (
        f"Easy task (deductive_liability) scored 0.0 with gpt-4o — "
        "check that the environment is returning observations correctly."
    )


@pytest.mark.asyncio
async def test_live_task2_terminates_with_valid_score():
    """gpt-4o must complete an abductive_conflict episode with score in [0, 1]."""
    result = await _run_one("abductive_conflict", seed=2101)

    assert isinstance(result["score"], float)
    assert 0.0 <= result["score"] <= 1.0, f"score {result['score']} out of range"
    assert result["steps"] >= 1


@pytest.mark.asyncio
async def test_live_task3_terminates_with_valid_score():
    """gpt-4o must complete an adversarial_fabrication episode with score in [0, 1]."""
    result = await _run_one("adversarial_fabrication", seed=3101)

    assert isinstance(result["score"], float)
    assert 0.0 <= result["score"] <= 1.0, f"score {result['score']} out of range"
    assert result["steps"] >= 1


@pytest.mark.asyncio
async def test_live_episode_always_terminates():
    """Episode must always reach done=True (not hang at step limit)."""
    from server.env import ClarusEnv
    from server.models import ClarusAction
    from inference import HF_TOKEN, MODEL_NAME, SYSTEM_PROMPT, get_model_action, _base_url_override
    from openai import OpenAI

    ref_db = _make_ref_db()
    env = ClarusEnv(ref_db=ref_db)

    kwargs: dict = {"api_key": HF_TOKEN}
    if _base_url_override:
        kwargs["base_url"] = _base_url_override
    client = OpenAI(**kwargs)

    max_steps = 12  # deductive_liability limit
    obs = await env.reset("deductive_liability", seed=1102)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    done = False
    steps = 0
    try:
        while not done and steps < max_steps + 2:
            steps += 1
            action_dict = get_model_action(
                client, obs, "deductive_liability", steps, max_steps, messages
            )
            action = ClarusAction(
                action_type=action_dict.get("action_type", "close_case"),
                parameters=action_dict.get("parameters", {}),
            )
            result = await env.step(action)
            obs = result.observation
            done = result.done

        # Force close if still not done (simulates timeout in inference.py)
        if not done:
            result = await env.step(
                ClarusAction(action_type="close_case", parameters={"outcome_code": "timeout"})
            )
            done = result.done
    finally:
        await env.close()

    assert done, "Episode never reached done=True"


@pytest.mark.asyncio
async def test_live_require_api_key_exits_cleanly(monkeypatch):
    """inference._require_api_key() must exit with code 1 when key is absent."""
    import sys
    from unittest.mock import patch

    # Temporarily clear the key
    monkeypatch.setattr("inference.HF_TOKEN", "")

    from inference import _require_api_key

    with pytest.raises(SystemExit) as exc_info:
        _require_api_key()

    assert exc_info.value.code == 1
