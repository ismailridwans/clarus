"""Compliance event logging for Clarus.

Records HIPAA and regulatory violations when the agent takes actions
out of proper sequence (e.g., accessing PHI before authentication,
filing an NSA dispute without checking regulatory rules).

check_and_log_compliance() is called in step() AFTER validation but
BEFORE execution, so violations are logged regardless of whether the
action succeeds.
"""

from __future__ import annotations

import sqlite3
from typing import Optional


VIOLATION_DESCRIPTIONS: dict = {
    "pii_before_auth": (
        "Agent attempted to access patient health information (PHI) before "
        "authenticating the patient. HIPAA requires identity verification "
        "before any PHI disclosure."
    ),
    "dispute_no_regulatory": (
        "NSA dispute resolution submitted without first checking the applicable "
        "regulatory rule. 45 CFR § 149.510 compliance requires regulatory "
        "verification before filing."
    ),
    "appeal_no_draft": (
        "Resolution submitted without a prior draft_resolution artifact. "
        "Workflow requires drafting before submission."
    ),
    "refund_no_ledger": (
        "Refund resolution submitted without fetching the payment ledger. "
        "Cannot verify patient payment without ledger review."
    ),
}

# Actions that require prior authentication (PHI access)
_PHI_ACTIONS = frozenset(
    {
        "fetch_claim_record",
        "fetch_eob",
        "fetch_provider_record",
        "fetch_payment_ledger",
        "fetch_plan_document",
        "fetch_facility_record",
        "fetch_payment_processor_log",
        "write_diagnosis",
        "draft_resolution",
        "submit_resolution",
        "send_patient_communication",
        "notify_provider",
        "reject_counter_argument",
        "write_audit_entry",
    }
)


def _is_authenticated(episode_id: str, db: sqlite3.Connection) -> bool:
    """Return True if authenticate_patient has been called for this episode."""
    row = db.execute(
        "SELECT 1 FROM episode_artifacts "
        "WHERE episode_id=? AND artifact_type='auth_record' LIMIT 1",
        (episode_id,),
    ).fetchone()
    return row is not None


def _has_artifact(
    episode_id: str, artifact_type: str, db: sqlite3.Connection
) -> bool:
    """Return True if at least one artifact of the given type exists."""
    row = db.execute(
        "SELECT 1 FROM episode_artifacts "
        "WHERE episode_id=? AND artifact_type=? LIMIT 1",
        (episode_id, artifact_type),
    ).fetchone()
    return row is not None


def check_and_log_compliance(
    action_type: str,
    episode_id: str,
    db: sqlite3.Connection,
    step_number: int,
    resolution_type: Optional[str] = None,
) -> None:
    """Check for compliance violations and insert events if any are found.

    Called in step() before action execution.  Does not block the action —
    violations are logged for grader analysis only.

    Args:
        action_type: The action being taken.
        episode_id: Current episode identifier.
        db: SQLite connection (runtime tables).
        step_number: Current step number.
        resolution_type: Resolved resolution type from draft (for submit actions).
    """
    violations = []

    # PHI access before authentication
    if action_type in _PHI_ACTIONS and not _is_authenticated(episode_id, db):
        violations.append("pii_before_auth")

    # NSA dispute without regulatory check
    if action_type == "submit_resolution" and resolution_type == "nsa_dispute":
        if not _has_artifact(episode_id, "regulatory_rule", db):
            violations.append("dispute_no_regulatory")

    # Submission without prior draft
    if action_type == "submit_resolution":
        if not _has_artifact(episode_id, "draft_resolution", db):
            violations.append("appeal_no_draft")

    # Refund without ledger check
    if action_type == "submit_resolution" and resolution_type == "refund":
        if not _has_artifact(episode_id, "payment_ledger", db):
            violations.append("refund_no_ledger")

    for violation_type in violations:
        db.execute(
            "INSERT INTO compliance_events "
            "(episode_id, step_number, violation_type, description) "
            "VALUES (?, ?, ?, ?)",
            (
                episode_id,
                step_number,
                violation_type,
                VIOLATION_DESCRIPTIONS.get(violation_type, violation_type),
            ),
        )
    if violations:
        db.commit()
