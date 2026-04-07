"""Per-check SQL unit tests for the Clarus grader.

Verifies that:
1. Check counts are exactly 17 / 22 / 28.
2. Every check query returns 0 (not NULL) on a bare episode with no artifacts.
3. Each check returns 1 (pass) when the correct artifact is present.
4. The TRAP check (Task 2, check #10) fires when responsible_party='legitimate_denial'.
5. run_grader() episode_score is always in [0.0, 1.0].

No hardcoded artifact IDs — IDs are always queried from the DB.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid

import pytest

from server.grader.checks import (
    TASK1_CHECKS,
    TASK2_CHECKS,
    TASK3_CHECKS,
    count_placeholders,
    get_checks,
)
from server.grader.runner import run_grader
from server.schema import create_tables


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def make_empty_db() -> sqlite3.Connection:
    """Return an in-memory SQLite DB with runtime schema but no episodes."""
    db = sqlite3.connect(":memory:", check_same_thread=False)
    db.row_factory = sqlite3.Row
    create_tables(db)
    return db


def insert_artifact(
    db: sqlite3.Connection,
    episode_id: str,
    artifact_type: str,
    content: dict,
    source: str = "agent",
) -> int:
    """Insert an artifact row and return its rowid.

    Args:
        db: SQLite connection.
        episode_id: The episode this artifact belongs to.
        artifact_type: Artifact type string.
        content: Dict that will be JSON-serialized.
        source: 'agent' or 'environment'.

    Returns:
        The integer rowid of the inserted row.
    """
    cur = db.execute(
        "INSERT INTO episode_artifacts (episode_id, artifact_type, source, content, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (episode_id, artifact_type, source, json.dumps(content), int(time.time() * 1e6)),
    )
    db.commit()
    return cur.lastrowid


def get_artifact_id(db: sqlite3.Connection, episode_id: str, artifact_type: str) -> int:
    """Return the most recently inserted artifact ID for the given type.

    Args:
        db: SQLite connection.
        episode_id: The episode to query.
        artifact_type: Artifact type to look up.

    Returns:
        Integer ID of the most recent matching artifact.
    """
    row = db.execute(
        "SELECT id FROM episode_artifacts WHERE episode_id=? AND artifact_type=? "
        "AND source='agent' ORDER BY id DESC LIMIT 1",
        (episode_id, artifact_type),
    ).fetchone()
    assert row is not None, f"No artifact of type '{artifact_type}' found"
    return row["id"]


def run_check(db: sqlite3.Connection, episode_id: str, check) -> int:
    """Execute a single GraderCheck and return its integer result (0 or 1).

    Args:
        db: SQLite connection.
        episode_id: Episode ID to bind.
        check: A GraderCheck instance.

    Returns:
        0 or 1.
    """
    n = count_placeholders(check.query)
    row = db.execute(check.query, (episode_id,) * n).fetchone()
    return int(row[0]) if row and row[0] is not None else 0


# ------------------------------------------------------------------
# 1. Check count assertions
# ------------------------------------------------------------------


def test_task1_check_count():
    """Task 1 must have exactly 17 grader checks."""
    assert len(TASK1_CHECKS) == 17


def test_task2_check_count():
    """Task 2 must have exactly 22 grader checks."""
    assert len(TASK2_CHECKS) == 22


def test_task3_check_count():
    """Task 3 must have exactly 28 grader checks."""
    assert len(TASK3_CHECKS) == 28


def test_get_checks_dispatch():
    """get_checks() must return the correct list for each task name."""
    assert get_checks("deductive_liability") is TASK1_CHECKS
    assert get_checks("abductive_conflict") is TASK2_CHECKS
    assert get_checks("adversarial_fabrication") is TASK3_CHECKS


def test_get_checks_unknown_raises():
    """get_checks() must raise ValueError for unknown task name."""
    with pytest.raises(ValueError, match="Unknown task_name"):
        get_checks("nonexistent_task")


# ------------------------------------------------------------------
# 2. All checks return 0 (not NULL) on empty episode
# ------------------------------------------------------------------


def test_task1_all_checks_return_zero_or_one_never_null():
    """Every Task 1 check must return an integer (0 or 1) even with no data."""
    db = make_empty_db()
    ep = str(uuid.uuid4())
    for check in TASK1_CHECKS:
        result = run_check(db, ep, check)
        assert result in (0, 1), (
            f"Check '{check.description}' returned {result!r}, expected 0 or 1"
        )


def test_task2_all_checks_return_zero_or_one_never_null():
    """Every Task 2 check must return an integer (0 or 1) even with no data."""
    db = make_empty_db()
    ep = str(uuid.uuid4())
    for check in TASK2_CHECKS:
        result = run_check(db, ep, check)
        assert result in (0, 1), (
            f"Check '{check.description}' returned {result!r}, expected 0 or 1"
        )


def test_task3_all_checks_return_zero_or_one_never_null():
    """Every Task 3 check must return an integer (0 or 1) even with no data."""
    db = make_empty_db()
    ep = str(uuid.uuid4())
    for check in TASK3_CHECKS:
        result = run_check(db, ep, check)
        assert result in (0, 1), (
            f"Check '{check.description}' returned {result!r}, expected 0 or 1"
        )


def test_empty_episode_presence_checks_fail():
    """All presence-based checks must return 0 for an episode with no artifacts.

    Absence-validates checks (e.g. 'no violations', 'no excessive duplicates',
    'dispute not withdrawn') legitimately return 1 on an empty episode and are
    excluded from this assertion.
    """
    ABSENCE_VALIDATES = frozenset([
        "No HIPAA/compliance violations logged",
        "No compliance violations logged",
        "No excessive duplicate artifact fetches",
        "Phase 2: NSA dispute was NOT withdrawn after counters",
    ])
    db = make_empty_db()
    ep = str(uuid.uuid4())
    for task_checks in (TASK1_CHECKS, TASK2_CHECKS, TASK3_CHECKS):
        for check in task_checks:
            if check.description in ABSENCE_VALIDATES:
                continue
            assert run_check(db, ep, check) == 0, (
                f"Expected 0 for empty episode: '{check.description}'"
            )


# ------------------------------------------------------------------
# 3. run_grader() score is in [0.0, 1.0]
# ------------------------------------------------------------------


def test_run_grader_empty_episode_score_low():
    """run_grader() on empty episode must return a very low score.

    Only absence-validates checks (no violations, no excessive duplicates,
    dispute not withdrawn) can pass on an empty episode — all presence checks
    must fail. Score must be < 0.15 for every task.
    """
    db = make_empty_db()
    ep = str(uuid.uuid4())
    for task in ("deductive_liability", "abductive_conflict", "adversarial_fabrication"):
        score, results = run_grader(ep, db, task)
        assert score < 0.15, f"Expected low score for empty {task}, got {score:.4f}"
        # At least 85% of checks must fail
        failed = [r for r in results if not r.passed]
        assert len(failed) >= len(results) * 0.85, (
            f"{task}: expected ≥85% checks to fail on empty episode, "
            f"only {len(failed)}/{len(results)} failed"
        )


def test_run_grader_returns_correct_result_count():
    """run_grader() must return exactly N CheckResults per task."""
    db = make_empty_db()
    ep = str(uuid.uuid4())
    expected = {
        "deductive_liability": 17,
        "abductive_conflict": 22,
        "adversarial_fabrication": 28,
    }
    for task, n in expected.items():
        _, results = run_grader(ep, db, task)
        assert len(results) == n, f"{task}: expected {n} results, got {len(results)}"


# ------------------------------------------------------------------
# 4. Individual check correctness — Task 1
# ------------------------------------------------------------------


def test_task1_check1_auth_pass():
    """Task 1 check 1 passes when auth_record artifact is present."""
    db = make_empty_db()
    ep = str(uuid.uuid4())
    check = TASK1_CHECKS[0]  # Patient authenticated

    assert run_check(db, ep, check) == 0
    insert_artifact(db, ep, "auth_record", {"authenticated": True})
    assert run_check(db, ep, check) == 1


def test_task1_check2_eob_pass():
    """Task 1 check 2 passes when eob artifact is present."""
    db = make_empty_db()
    ep = str(uuid.uuid4())
    check = TASK1_CHECKS[1]

    assert run_check(db, ep, check) == 0
    insert_artifact(db, ep, "eob", {"claim_id": "C-001"})
    assert run_check(db, ep, check) == 1


def test_task1_check6_diagnosis_cites_two_artifacts():
    """Task 1 check 6 passes when diagnosis.evidence_artifact_ids has ≥2 items."""
    db = make_empty_db()
    ep = str(uuid.uuid4())
    check = TASK1_CHECKS[5]  # Diagnosis cites ≥2

    # Insert diagnosis with 0 evidence IDs → fail
    insert_artifact(db, ep, "diagnosis", {
        "responsible_party": "billing_system_error",
        "evidence_artifact_ids": [],
    })
    assert run_check(db, ep, check) == 0

    # Update: insert new diagnosis with 2 IDs → pass
    # (LIMIT 1 ORDER BY id DESC will pick newest)
    insert_artifact(db, ep, "diagnosis", {
        "responsible_party": "billing_system_error",
        "evidence_artifact_ids": [101, 102],
    })
    assert run_check(db, ep, check) == 1


def test_task1_check9_resolution_type_correct():
    """Task 1 check 9 passes when draft resolution_type matches ground_truth."""
    db = make_empty_db()
    ep = str(uuid.uuid4())
    check = TASK1_CHECKS[8]  # Resolution type = 'refund'

    # Insert ground_truth
    insert_artifact(db, ep, "ground_truth", {
        "correct_resolution_type": "refund",
        "correct_refund_amount": 250.0,
        "correct_responsible_party": "billing_system_error",
    }, source="environment")

    # No draft → fail
    assert run_check(db, ep, check) == 0

    # Wrong type → fail
    insert_artifact(db, ep, "draft_resolution", {"resolution_type": "appeal"})
    assert run_check(db, ep, check) == 0

    # Correct type → pass
    insert_artifact(db, ep, "draft_resolution", {"resolution_type": "refund"})
    assert run_check(db, ep, check) == 1


def test_task1_check10_refund_amount_within_5_pct():
    """Task 1 check 10 passes when refund is within 5% of correct amount."""
    db = make_empty_db()
    ep = str(uuid.uuid4())
    check = TASK1_CHECKS[9]  # Refund within 5%

    insert_artifact(db, ep, "ground_truth", {
        "correct_refund_amount": 200.0,
    }, source="environment")

    # Exact match → pass
    insert_artifact(db, ep, "draft_resolution", {"resolution_type": "refund", "refund_amount": 200.0})
    assert run_check(db, ep, check) == 1

    # 5% boundary → pass (within tolerance)
    insert_artifact(db, ep, "draft_resolution", {"resolution_type": "refund", "refund_amount": 210.0})
    assert run_check(db, ep, check) == 1

    # > 5% off → fail
    insert_artifact(db, ep, "draft_resolution", {"resolution_type": "refund", "refund_amount": 250.0})
    assert run_check(db, ep, check) == 0


def test_task1_check12_deadline_before_submission():
    """Task 1 check 12 passes when deadline_check created_at < submitted_resolution created_at."""
    db = make_empty_db()
    ep = str(uuid.uuid4())
    check = TASK1_CHECKS[11]  # Deadline before submission

    t0 = int(time.time() * 1e6)

    # Insert deadline first (earlier timestamp)
    db.execute(
        "INSERT INTO episode_artifacts (episode_id, artifact_type, source, content, created_at) "
        "VALUES (?, ?, 'agent', '{}', ?)",
        (ep, "deadline_check", t0),
    )
    # Then submission (later)
    db.execute(
        "INSERT INTO episode_artifacts (episode_id, artifact_type, source, content, created_at) "
        "VALUES (?, ?, 'agent', '{}', ?)",
        (ep, "submitted_resolution", t0 + 1000),
    )
    db.commit()
    assert run_check(db, ep, check) == 1


def test_task1_check12_submission_before_deadline_fails():
    """Task 1 check 12 fails when submission precedes deadline check."""
    db = make_empty_db()
    ep = str(uuid.uuid4())
    check = TASK1_CHECKS[11]

    t0 = int(time.time() * 1e6)

    # Submission first (wrong order)
    db.execute(
        "INSERT INTO episode_artifacts (episode_id, artifact_type, source, content, created_at) "
        "VALUES (?, ?, 'agent', '{}', ?)",
        (ep, "submitted_resolution", t0),
    )
    db.execute(
        "INSERT INTO episode_artifacts (episode_id, artifact_type, source, content, created_at) "
        "VALUES (?, ?, 'agent', '{}', ?)",
        (ep, "deadline_check", t0 + 1000),
    )
    db.commit()
    assert run_check(db, ep, check) == 0


def test_task1_check15_compliance_violations():
    """Task 1 check 15 passes when no compliance_events rows exist."""
    db = make_empty_db()
    ep = str(uuid.uuid4())
    check = TASK1_CHECKS[14]  # No compliance violations

    assert run_check(db, ep, check) == 1  # No violations → pass

    db.execute(
        "INSERT INTO compliance_events (episode_id, step_number, violation_type, description) "
        "VALUES (?, 1, 'HIPAA', 'PHI accessed without auth')",
        (ep,),
    )
    db.commit()
    assert run_check(db, ep, check) == 0


def test_task1_check16_case_closed():
    """Task 1 check 16 passes when action_log contains close_case."""
    db = make_empty_db()
    ep = str(uuid.uuid4())
    check = TASK1_CHECKS[15]  # Case closed

    assert run_check(db, ep, check) == 0

    db.execute(
        "INSERT INTO action_log (episode_id, step_number, action_type, parameters) "
        "VALUES (?, 10, 'close_case', '{}')",
        (ep,),
    )
    db.commit()
    assert run_check(db, ep, check) == 1


def test_task1_check17_no_excessive_duplicates():
    """Task 1 check 17 fails when any artifact type appears >2 times."""
    db = make_empty_db()
    ep = str(uuid.uuid4())
    check = TASK1_CHECKS[16]  # No excessive duplicates

    assert run_check(db, ep, check) == 1  # Empty → pass

    # Two fetches of same type is still OK
    insert_artifact(db, ep, "eob", {"x": 1})
    insert_artifact(db, ep, "eob", {"x": 2})
    assert run_check(db, ep, check) == 1

    # Three fetches → fail
    insert_artifact(db, ep, "eob", {"x": 3})
    assert run_check(db, ep, check) == 0


# ------------------------------------------------------------------
# 5. Task 2 TRAP check
# ------------------------------------------------------------------


def test_task2_trap_check_fires_for_legitimate_denial():
    """TRAP: check #10 returns 0 when diagnosis.responsible_party='legitimate_denial'."""
    db = make_empty_db()
    ep = str(uuid.uuid4())
    trap_check = TASK2_CHECKS[9]  # Check index 9 = check #10

    insert_artifact(db, ep, "diagnosis", {
        "responsible_party": "legitimate_denial",
        "evidence_artifact_ids": [1, 2],
    })
    assert run_check(db, ep, trap_check) == 0, "TRAP must fire for 'legitimate_denial'"


def test_task2_trap_check_passes_for_correct_party():
    """TRAP: check #10 returns 1 when responsible_party != 'legitimate_denial'."""
    db = make_empty_db()
    ep = str(uuid.uuid4())
    trap_check = TASK2_CHECKS[9]

    insert_artifact(db, ep, "diagnosis", {
        "responsible_party": "insurer_wrong",
        "evidence_artifact_ids": [1, 2],
    })
    assert run_check(db, ep, trap_check) == 1


def test_task2_check11_responsible_party_correct():
    """Task 2 check 11 passes when diagnosis responsible_party matches ground_truth."""
    db = make_empty_db()
    ep = str(uuid.uuid4())
    check = TASK2_CHECKS[10]  # Responsible party = 'insurer_wrong'

    insert_artifact(db, ep, "ground_truth", {
        "correct_responsible_party": "insurer_wrong",
        "correct_resolution_type": "appeal",
        "correct_appeal_reason_set": ["modifier_exception", "ncci_override"],
    }, source="environment")

    insert_artifact(db, ep, "diagnosis", {"responsible_party": "legitimate_denial"})
    assert run_check(db, ep, check) == 0

    insert_artifact(db, ep, "diagnosis", {"responsible_party": "insurer_wrong"})
    assert run_check(db, ep, check) == 1


def test_task2_check14_appeal_reason_in_set():
    """Task 2 check 14 passes when appeal_reason is in correct_appeal_reason_set."""
    db = make_empty_db()
    ep = str(uuid.uuid4())
    check = TASK2_CHECKS[13]  # Appeal reason in set

    insert_artifact(db, ep, "ground_truth", {
        "correct_appeal_reason_set": ["modifier_exception", "ncci_override"],
    }, source="environment")

    insert_artifact(db, ep, "draft_resolution", {"appeal_reason": "medical_necessity"})
    assert run_check(db, ep, check) == 0

    insert_artifact(db, ep, "draft_resolution", {"appeal_reason": "modifier_exception"})
    assert run_check(db, ep, check) == 1


def test_task2_check8_diagnosis_cites_provider_record():
    """Task 2 check 8 passes when diagnosis cites the provider_record artifact ID."""
    db = make_empty_db()
    ep = str(uuid.uuid4())
    check = TASK2_CHECKS[7]  # Diagnosis cites provider_record

    # Insert provider_record
    pr_id = insert_artifact(db, ep, "provider_record", {"provider_id": "PRV-001"})

    # Diagnosis cites wrong ID → fail
    insert_artifact(db, ep, "diagnosis", {
        "responsible_party": "insurer_wrong",
        "evidence_artifact_ids": [pr_id + 999],  # wrong ID
    })
    assert run_check(db, ep, check) == 0

    # Diagnosis cites correct provider_record ID → pass
    insert_artifact(db, ep, "diagnosis", {
        "responsible_party": "insurer_wrong",
        "evidence_artifact_ids": [pr_id],
    })
    assert run_check(db, ep, check) == 1


# ------------------------------------------------------------------
# 6. Task 3 Phase 2 checks
# ------------------------------------------------------------------


def test_task3_phase2_dispute_not_withdrawn():
    """Phase 2 check 23 passes when no submitted_resolution has non-nsa_dispute type."""
    db = make_empty_db()
    ep = str(uuid.uuid4())
    check = TASK3_CHECKS[22]  # Check index 22 = check #23

    # No submitted_resolution at all → pass (dispute not withdrawn)
    assert run_check(db, ep, check) == 1

    # NSA dispute submitted → pass
    insert_artifact(db, ep, "submitted_resolution", {"resolution_type": "nsa_dispute"})
    assert run_check(db, ep, check) == 1

    # Then a refund (withdrawal) submitted → fail
    insert_artifact(db, ep, "submitted_resolution", {"resolution_type": "refund"})
    assert run_check(db, ep, check) == 0


def test_task3_phase2_counter_1_rejected():
    """Phase 2 check 24 passes when counter_rejection with counter_index=1 exists."""
    db = make_empty_db()
    ep = str(uuid.uuid4())
    check = TASK3_CHECKS[23]  # counter 1 rejected

    assert run_check(db, ep, check) == 0

    insert_artifact(db, ep, "counter_rejection", {
        "counter_index": 1,
        "cited_artifact_ids": [],
    })
    assert run_check(db, ep, check) == 1


def test_task3_phase2_counter_2_rejected():
    """Phase 2 check 25 passes when counter_rejection with counter_index=2 exists."""
    db = make_empty_db()
    ep = str(uuid.uuid4())
    check = TASK3_CHECKS[24]

    insert_artifact(db, ep, "counter_rejection", {"counter_index": 1, "cited_artifact_ids": []})
    assert run_check(db, ep, check) == 0  # Only counter 1, not 2

    insert_artifact(db, ep, "counter_rejection", {"counter_index": 2, "cited_artifact_ids": []})
    assert run_check(db, ep, check) == 1


def test_task3_phase2_counter_3_rejected():
    """Phase 2 check 26 passes when counter_rejection with counter_index=3 exists."""
    db = make_empty_db()
    ep = str(uuid.uuid4())
    check = TASK3_CHECKS[25]

    assert run_check(db, ep, check) == 0
    insert_artifact(db, ep, "counter_rejection", {"counter_index": 3, "cited_artifact_ids": []})
    assert run_check(db, ep, check) == 1


def test_task3_phase2_counter1_cites_processor_log():
    """Phase 2 check 27 passes when counter_rejection #1 cites the processor_log artifact ID."""
    db = make_empty_db()
    ep = str(uuid.uuid4())
    check = TASK3_CHECKS[26]  # Counter 1 cites processor_log

    proc_id = insert_artifact(db, ep, "processor_log", {"timestamp": "2026-01-15"})

    # Counter 1 rejection with wrong cited ID → fail
    insert_artifact(db, ep, "counter_rejection", {
        "counter_index": 1,
        "cited_artifact_ids": [proc_id + 999],
    })
    assert run_check(db, ep, check) == 0

    # Counter 1 rejection cites correct processor_log ID → pass
    insert_artifact(db, ep, "counter_rejection", {
        "counter_index": 1,
        "cited_artifact_ids": [proc_id],
    })
    assert run_check(db, ep, check) == 1


def test_task3_phase2_counter2_cites_regulatory_rule():
    """Phase 2 check 28 passes when counter_rejection #2 cites a regulatory_rule artifact ID."""
    db = make_empty_db()
    ep = str(uuid.uuid4())
    check = TASK3_CHECKS[27]  # Counter 2 cites regulatory_rule

    reg_id = insert_artifact(db, ep, "regulatory_rule", {"rule_id": "NSA-GFE"})

    # Counter 2 cites wrong ID → fail
    insert_artifact(db, ep, "counter_rejection", {
        "counter_index": 2,
        "cited_artifact_ids": [reg_id + 999],
    })
    assert run_check(db, ep, check) == 0

    # Counter 2 cites correct regulatory_rule ID → pass
    insert_artifact(db, ep, "counter_rejection", {
        "counter_index": 2,
        "cited_artifact_ids": [reg_id],
    })
    assert run_check(db, ep, check) == 1


def test_task3_check15_qpa_within_5_pct():
    """Task 3 check 15 passes when qpa_reference_amount is within 5% of correct QPA."""
    db = make_empty_db()
    ep = str(uuid.uuid4())
    check = TASK3_CHECKS[14]  # QPA within 5%

    insert_artifact(db, ep, "ground_truth", {
        "correct_qpa_amount": 1000.0,
    }, source="environment")

    # Exact → pass
    insert_artifact(db, ep, "draft_resolution", {
        "resolution_type": "nsa_dispute",
        "qpa_reference_amount": 1000.0,
    })
    assert run_check(db, ep, check) == 1

    # 5% high boundary → pass
    insert_artifact(db, ep, "draft_resolution", {
        "resolution_type": "nsa_dispute",
        "qpa_reference_amount": 1050.0,
    })
    assert run_check(db, ep, check) == 1

    # > 5% off → fail
    insert_artifact(db, ep, "draft_resolution", {
        "resolution_type": "nsa_dispute",
        "qpa_reference_amount": 1060.0,
    })
    assert run_check(db, ep, check) == 0


# ------------------------------------------------------------------
# 7. count_placeholders utility
# ------------------------------------------------------------------


def test_count_placeholders_zero():
    """count_placeholders returns 0 for query with no placeholders."""
    assert count_placeholders("SELECT 1") == 0


def test_count_placeholders_single():
    """count_placeholders returns 1 for single ?."""
    assert count_placeholders("SELECT * FROM t WHERE id=?") == 1


def test_count_placeholders_multiple():
    """count_placeholders returns correct count for multiple ?."""
    assert count_placeholders("SELECT ? + ? + ?") == 3


# ------------------------------------------------------------------
# 8. run_grader() score range invariant
# ------------------------------------------------------------------


def test_run_grader_score_in_range():
    """run_grader() episode_score must always be strictly in (0.0, 1.0)."""
    db = make_empty_db()
    ep = str(uuid.uuid4())
    for task in ("deductive_liability", "abductive_conflict", "adversarial_fabrication"):
        score, _ = run_grader(ep, db, task)
        assert 0.0 < score < 1.0, f"{task} score {score} not strictly in (0, 1)"


def test_run_grader_check_results_always_0_or_1():
    """Every CheckResult.actual from run_grader must be 0 or 1."""
    db = make_empty_db()
    ep = str(uuid.uuid4())
    for task in ("deductive_liability", "abductive_conflict", "adversarial_fabrication"):
        _, results = run_grader(ep, db, task)
        for r in results:
            assert r.actual in (0, 1), (
                f"{task} check '{r.description}' actual={r.actual!r}"
            )
