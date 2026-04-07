"""Self-test: runs at docker build time.

A lightweight smoke test — verifies that:
1. All three tasks can be reset and produce a valid observation.
2. A minimal trajectory on each task produces a non-zero grader score.
3. The reference DB has the required data.

This is NOT the full trajectory test (test_episodes.py).
Failure here causes the Docker build to fail.
"""

from __future__ import annotations

import asyncio
import sqlite3
import sys


def make_ref_db() -> sqlite3.Connection:
    """Build an in-memory reference DB from bundle data."""
    from server.schema import create_tables
    from data.setup import load_all

    db = sqlite3.connect(":memory:", check_same_thread=False)
    db.row_factory = sqlite3.Row
    create_tables(db)
    counts = load_all(db)

    assert counts["cpt_codes"] > 0, f"No CPT codes loaded: {counts}"
    assert counts["ncci_edits"] > 0, f"No NCCI edits loaded: {counts}"
    assert counts["plan_templates"] == 8, f"Expected 8 plans: {counts}"
    assert counts["nsa_qpa_rates"] > 0, f"No NSA QPA rates: {counts}"
    return db


async def _smoke_test_task(ref_db, task_name: str, seed: int) -> None:
    """Run a minimal episode for the task — reset + auth + close."""
    from server.env import ClarusEnv
    from server.models import ClarusAction

    env = ClarusEnv(ref_db=ref_db)
    obs = await env.reset(task_name, seed=seed)
    assert obs.case_id, f"case_id missing for {task_name}"
    assert obs.patient_name, f"patient_name missing for {task_name}"
    assert obs.patient_complaint, f"patient_complaint missing for {task_name}"
    assert obs.patient_emotional_state in ("calm", "frustrated", "distressed")
    assert obs.step_number == 0

    # Authenticate
    r = await env.step(ClarusAction(action_type="authenticate_patient"))
    assert not r.done
    assert r.reward > 0

    # Close immediately (low score but must not crash)
    r = await env.step(ClarusAction(action_type="close_case", parameters={"outcome_code": "resolved"}))
    assert r.done
    assert r.info["episode_score"] is not None
    score = r.info["episode_score"]
    assert 0.0 < score < 1.0, f"Score not strictly in (0, 1) for {task_name}: {score}"
    print(f"  {task_name} seed={seed}: score={score:.3f} — smoke test PASSED")


async def main() -> None:
    """Run all smoke tests."""
    print("=== Clarus self-test ===", flush=True)

    ref_db = make_ref_db()
    print("  Reference DB loaded OK.", flush=True)

    tests = [
        ("deductive_liability", 1001),
        ("abductive_conflict", 2001),
        ("adversarial_fabrication", 3001),
    ]
    for task_name, seed in tests:
        await _smoke_test_task(ref_db, task_name, seed)

    print("=== All self-tests PASSED ===", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
