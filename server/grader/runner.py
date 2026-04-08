"""Grader runner for Clarus.

run_grader() executes all SQL checks for a completed episode and returns
the episode score and per-check results.

Formula: episode_score = base + ceiling * (passing_checks / total_checks)
  Each task has its own ceiling reflecting its difficulty:

  Task                   base  ceiling  perfect   zero
  deductive_liability    0.05   0.68    0.730     0.050   (easy)
  abductive_conflict     0.05   0.58    0.630     0.050   (medium)
  adversarial_fabrication 0.05  0.48   0.530     0.050   (hard)

  Always strictly in (0.05, 0.73) — well inside (0, 1) on both sides.
  Satisfies the OpenEnv Phase 2 validator strict (0, 1) requirement.

Each check query is expected to return exactly one row with one integer
column (0 or 1).  COALESCE in every query ensures no NULL is returned.
All ? placeholders in a query are bound to episode_id.
"""

from __future__ import annotations

import sqlite3
from typing import List, Tuple

from server.grader.checks import GraderCheck, count_placeholders, get_checks
from server.models import CheckResult


# Per-task score parameters: (base, ceiling)
# episode_score = base + ceiling * (passed / total)
# Harder tasks have a lower ceiling so perfect performance still yields a
# meaningfully lower score than the easy task — reflects genuine difficulty.
_TASK_SCORE_PARAMS = {
    "deductive_liability":     (0.05, 0.68),   # perfect → 0.730
    "abductive_conflict":      (0.05, 0.58),   # perfect → 0.630
    "adversarial_fabrication": (0.05, 0.48),   # perfect → 0.530
}


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
        - episode_score is base + ceiling*(passing/total), always in (0, 1)
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
    if total > 0:
        base, ceiling = _TASK_SCORE_PARAMS.get(task_name, (0.05, 0.68))
        proportion = passed / total
        episode_score = base + ceiling * proportion
    else:
        episode_score = 0.5
    return episode_score, results
