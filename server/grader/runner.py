"""Grader runner for Clarus.

run_grader() executes all SQL checks for a completed episode and returns
the episode score and per-check results.

Scoring formula  (Laplace smoothing):

    episode_score = (checks_passed + 0.5) / (checks_total + 1.0)

This is the standard way to convert a raw pass-rate into a value that is
ALWAYS strictly inside (0, 1) — never exactly 0.0 or 1.0 — because:

    numerator   = checks_passed + 0.5   →  at least 0.5,  at most  N + 0.5
    denominator = checks_total  + 1.0   →  always N + 1.0
    ratio       = num / denom           →  (0.5/N+1,  (N+0.5)/N+1)  ⊂  (0,1)

Typical scores for the three tasks:

    Task                    Checks   Perfect agent   Zero agent
    deductive_liability       17       0.972          0.028
    abductive_conflict        22       0.978          0.022
    adversarial_fabrication   28       0.983          0.017

The score is determined entirely by how many grader SQL checks the agent
passes — no artificial caps, multipliers, or difficulty weights.
Harder tasks have more checks and stricter SQL conditions; that is the
only source of score difference between difficulty levels.

Each check query returns exactly one integer 0 or 1.
COALESCE in every query ensures no NULL is returned.
All ? placeholders are bound to episode_id.
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
        - episode_score = (passed + 0.5) / (total + 1.0), always in (0, 1)
        - check_results = list of CheckResult with per-check pass/fail
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
            actual_int = int(actual) if actual is not None else 0
            check_passed = actual_int == 1
        except Exception as exc:
            actual_int = 0
            check_passed = False
            print(f"[GRADER ERROR] {check.description}: {exc}", flush=True)

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
    # Laplace smoothing — always strictly in (0, 1) regardless of pass count
    episode_score = (passed + 0.5) / (total + 1.0) if total > 0 else 0.5
    # Hard clamp as belt-and-suspenders: Laplace can never reach 0.0 or 1.0
    # but clamp to [0.01, 0.99] to survive any floating-point edge case.
    episode_score = max(0.01, min(0.99, episode_score))
    return episode_score, results
