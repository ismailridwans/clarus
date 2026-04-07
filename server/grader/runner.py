"""Grader runner for Clarus.

run_grader() executes all SQL checks for a completed episode and returns
the episode score and per-check results.

Formula: episode_score = (passing_checks + 0.5) / (total_checks + 1.0)
Laplace-smoothed so the score is always strictly in (0, 1) — never 0.0
or 1.0 — which satisfies the OpenEnv hackathon Phase 2 validator.

For a perfect agent:   (N + 0.5) / (N + 1)  →  0.972 – 0.983
For a zero agent:      0.5       / (N + 1)  →  0.028 – 0.017

Each check query is expected to return exactly one row with one integer
column (0 or 1).  COALESCE in every query ensures no NULL is returned.
All ? placeholders in a query are bound to episode_id.
"""

from __future__ import annotations

import sqlite3
from typing import List, Tuple

from server.grader.checks import GraderCheck, count_placeholders, get_checks
from server.models import CheckResult


def run_grader(
    episode_id: str,
    db: sqlite3.Connection,
    task_name: str,
) -> Tuple[float, List[CheckResult]]:
    """Run all grader checks for the episode and return score + details.

    Args:
        episode_id: The episode to grade.
        db: Runtime SQLite connection (contains episode_artifacts, etc.).
        task_name: Task name to determine which checks to run.

    Returns:
        Tuple of (episode_score, check_results) where:
        - episode_score is (passing + 0.5) / (total + 1) in (0.0, 1.0) strict
        - check_results is a list of CheckResult with per-check pass/fail
    """
    checks: List[GraderCheck] = get_checks(task_name)
    results: List[CheckResult] = []
    passed = 0

    for check in checks:
        n_params = count_placeholders(check.query)
        params = (episode_id,) * n_params

        try:
            row = db.execute(check.query, params).fetchone()
            actual = row[0] if row is not None else 0
            # Ensure we got an integer 0 or 1
            actual_int = int(actual) if actual is not None else 0
            check_passed = actual_int == 1
        except Exception as exc:
            actual_int = 0
            check_passed = False
            print(
                f"[GRADER ERROR] {check.description}: {exc}",
                flush=True,
            )

        if check_passed:
            passed += 1

        results.append(
            CheckResult(
                description=check.description,
                passed=check_passed,
                actual=actual_int,
                expected=1,
            )
        )

    total = len(checks)
    # Laplace smoothing: always strictly in (0, 1) — satisfies OpenEnv validator
    episode_score = (passed + 0.5) / (total + 1.0) if total > 0 else 0.5
    return episode_score, results
