"""Write action execution for Clarus.

execute_write_action() handles all state-mutating actions:
authenticate_patient, write_diagnosis, draft_resolution, submit_resolution,
send_patient_communication, notify_provider, reject_counter_argument,
write_audit_entry, close_case.

State transitions (e.g., patient emotional state) happen INSIDE this
function, BEFORE the reward is computed by the caller.  The caller reads
the post-transition state from DB to determine the reward.

submit_resolution does NOT read resolution_type from action_parameters —
it reads it from the draft artifact content to prevent spoofing.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Dict, Optional, Tuple

from server.scenario.params import EpisodeParams


# Patient state transitions per message_type
STATE_TRANSITIONS: Dict[str, Dict[str, str]] = {
    "de_escalation": {
        "distressed": "frustrated",
        "frustrated": "calm",
        "calm": "calm",
    },
    "outcome": {
        "distressed": "frustrated",
        "frustrated": "calm",
        "calm": "calm",
    },
    "explanation": {
        "distressed": "distressed",
        "frustrated": "frustrated",
        "calm": "calm",
    },
}

STATE_ORDINAL: Dict[str, int] = {"calm": 0, "frustrated": 1, "distressed": 2}

# Adversarial counter messages for Task 3 Phase 2
# Injected into episode_artifacts after submit_resolution
ADVERSARIAL_COUNTERS = [
    {
        "counter_index": 1,
        "source_label": "provider_response",
        "content": (
            "We have EHR notes from Jan 10 signed by our coordinator "
            "confirming the Good Faith Estimate was prepared that day. "
            "Our clinical documentation supports the date on the GFE."
        ),
        "correct_response": "reject",
        "rejection_reason": (
            "Provider's own records are not independent evidence. "
            "The payment processor log is an independent system timestamp "
            "that cannot be altered retroactively."
        ),
    },
    {
        "counter_index": 2,
        "source_label": "provider_response",
        "content": (
            "The No Surprises Act contains exceptions for emergency services. "
            "NSA § 149.110 provides an emergency exception that applies here. "
            "Our legal team has reviewed this and the NSA does not apply."
        ),
        "correct_response": "reject",
        "rejection_reason": (
            "The NSA emergency exception (§ 149.110) applies only to emergency "
            "medical conditions requiring immediate treatment. This was an "
            "elective, scheduled procedure. The emergency exception does not apply."
        ),
    },
    {
        "counter_index": 3,
        "source_label": "provider_response",
        "content": (
            "We have engaged legal counsel in this matter. Our attorneys have "
            "advised that pursuing this dispute further could result in "
            "collections activity. We urge you to reconsider before proceeding."
        ),
        "correct_response": "reject",
        "rejection_reason": (
            "Legal threat without new factual evidence. The processor log "
            "timestamp is an independent, unalterable record. The dispute "
            "stands on its merits regardless of legal pressure."
        ),
    },
]


def _get_current_patient_state(
    episode_id: str, db: sqlite3.Connection, initial_state: str
) -> str:
    """Return the current patient emotional state for the episode.

    Looks for the most recent patient_state artifact; falls back to initial.

    Args:
        episode_id: Current episode ID.
        db: Runtime SQLite connection.
        initial_state: The initial_patient_state from EpisodeParams.

    Returns:
        One of "calm", "frustrated", "distressed".
    """
    row = db.execute(
        "SELECT content FROM episode_artifacts "
        "WHERE episode_id=? AND artifact_type='patient_state' "
        "ORDER BY id DESC LIMIT 1",
        (episode_id,),
    ).fetchone()
    if row:
        return json.loads(row[0]).get("state", initial_state)
    return initial_state


def _set_patient_state(
    episode_id: str, step_number: int, new_state: str, db: sqlite3.Connection
) -> None:
    """Write the updated patient state as an artifact."""
    db.execute(
        "INSERT INTO episode_artifacts "
        "(episode_id, artifact_type, source, content, created_at) "
        "VALUES (?, 'patient_state', 'environment', ?, ?)",
        (episode_id, json.dumps({"state": new_state}), step_number),
    )


def _insert_write_artifact(
    episode_id: str,
    artifact_type: str,
    content: Dict[str, Any],
    step_number: int,
    db: sqlite3.Connection,
) -> int:
    """Insert a write artifact and return its assigned artifact_id."""
    cursor = db.execute(
        "INSERT INTO episode_artifacts "
        "(episode_id, artifact_type, source, content, created_at) "
        "VALUES (?, ?, 'agent', ?, ?)",
        (episode_id, artifact_type, json.dumps(content), step_number),
    )
    return cursor.lastrowid


def validate_and_enrich_submit_resolution(
    parameters: Dict[str, Any],
    episode_id: str,
    db: sqlite3.Connection,
) -> Tuple[Optional[str], Optional[str]]:
    """Validate submit_resolution and extract resolution_type from the draft.

    Reads resolution_type from the draft artifact content, NOT from
    action parameters.  This prevents the agent from spoofing the type.

    Args:
        parameters: Raw action parameters (must contain draft_artifact_id).
        episode_id: Current episode ID.
        db: Runtime SQLite connection.

    Returns:
        Tuple of (error_message, resolution_type).
        error_message is None on success; resolution_type is None on failure.
    """
    draft_id = parameters.get("draft_artifact_id")
    if draft_id is None:
        return "Missing required field: draft_artifact_id", None

    row = db.execute(
        "SELECT content, source FROM episode_artifacts "
        "WHERE id=? AND episode_id=? AND artifact_type='draft_resolution'",
        (draft_id, episode_id),
    ).fetchone()
    if row is None:
        return (
            f"draft_artifact_id={draft_id} not found or does not belong "
            "to this episode",
            None,
        )
    if row[1] != "agent":
        return (
            f"draft_artifact_id={draft_id} was not created by the agent",
            None,
        )

    draft_content = json.loads(row[0])
    resolution_type = draft_content.get("resolution_type")
    if not resolution_type:
        return "Draft artifact missing resolution_type field", None

    return None, resolution_type


def _inject_adversarial_counters(
    episode_id: str,
    step_number: int,
    params: EpisodeParams,
    db: sqlite3.Connection,
) -> None:
    """Inject Task 3 Phase 2 counter-argument artifacts after submit_resolution.

    Each counter is stored as a 'provider_response' artifact with source
    'provider'.  The agent must call reject_counter_argument for each.

    Args:
        episode_id: Current episode ID.
        step_number: Step number at which submit_resolution was called.
        params: EpisodeParams (checked for task_name == adversarial_fabrication).
        db: Runtime SQLite connection.
    """
    if params.task_name != "adversarial_fabrication":
        return

    for counter in ADVERSARIAL_COUNTERS:
        db.execute(
            "INSERT INTO episode_artifacts "
            "(episode_id, artifact_type, source, content, created_at) "
            "VALUES (?, 'provider_response', 'provider', ?, ?)",
            (episode_id, json.dumps(counter), step_number),
        )


async def execute_write_action(
    action_type: str,
    action_parameters: Dict[str, Any],
    episode_id: str,
    step_number: int,
    params: EpisodeParams,
    db: sqlite3.Connection,
    resolved_resolution_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute a write action and return the result payload.

    State transitions happen inside this function BEFORE returning.
    The reward computation in the caller reads the DB state AFTER this
    function completes, ensuring reward reflects the new state.

    Args:
        action_type: Write action name.
        action_parameters: Parameters from the agent.
        episode_id: Current episode ID.
        step_number: Current step number.
        params: EpisodeParams for this episode.
        db: Runtime SQLite connection.
        resolved_resolution_type: For submit_resolution — type read from draft.

    Returns:
        Dict describing what was written (artifact_id, artifact_type, etc.).
    """
    ap = action_parameters

    # ------------------------------------------------------------------
    # authenticate_patient
    # ------------------------------------------------------------------
    if action_type == "authenticate_patient":
        content = {
            "patient_id": params.patient_id,
            "patient_name": params.patient_name,
            "authenticated": True,
            "method": ap.get("method", "identity_verification"),
        }
        artifact_id = _insert_write_artifact(
            episode_id, "auth_record", content, step_number, db
        )
        db.commit()
        return {
            "artifact_id": artifact_id,
            "artifact_type": "auth_record",
            "authenticated": True,
            "patient_id": params.patient_id,
            "patient_name": params.patient_name,
        }

    # ------------------------------------------------------------------
    # write_diagnosis
    # ------------------------------------------------------------------
    if action_type == "write_diagnosis":
        responsible_party = ap.get("responsible_party", "")
        evidence_ids = ap.get("evidence_artifact_ids", [])
        content = {
            "responsible_party": responsible_party,
            "evidence_artifact_ids": evidence_ids,
            "diagnosis_text": ap.get("diagnosis_text", ""),
            "written_at_step": step_number,
        }
        artifact_id = _insert_write_artifact(
            episode_id, "diagnosis", content, step_number, db
        )
        db.commit()
        return {
            "artifact_id": artifact_id,
            "artifact_type": "diagnosis",
            "responsible_party": responsible_party,
            "evidence_artifact_ids": evidence_ids,
        }

    # ------------------------------------------------------------------
    # draft_resolution
    # ------------------------------------------------------------------
    if action_type == "draft_resolution":
        resolution_type = ap.get("resolution_type", "")
        content = {
            "resolution_type": resolution_type,
            "refund_amount": ap.get("refund_amount"),
            "appeal_reason": ap.get("appeal_reason"),
            "nsa_violation_basis": ap.get("nsa_violation_basis"),
            "qpa_reference_amount": ap.get("qpa_reference_amount"),
            "summary": ap.get("summary", ""),
            "drafted_at_step": step_number,
        }
        artifact_id = _insert_write_artifact(
            episode_id, "draft_resolution", content, step_number, db
        )
        db.commit()
        return {
            "artifact_id": artifact_id,
            "artifact_type": "draft_resolution",
            "resolution_type": resolution_type,
        }

    # ------------------------------------------------------------------
    # submit_resolution
    # ------------------------------------------------------------------
    if action_type == "submit_resolution":
        # resolution_type was already read from draft by the caller
        content = {
            "draft_artifact_id": ap.get("draft_artifact_id"),
            "resolution_type": resolved_resolution_type,
            "submitted_at_step": step_number,
            "status": "submitted",
        }
        artifact_id = _insert_write_artifact(
            episode_id, "submitted_resolution", content, step_number, db
        )
        # Inject Phase 2 counters if Task 3
        _inject_adversarial_counters(episode_id, step_number, params, db)
        db.commit()
        return {
            "artifact_id": artifact_id,
            "artifact_type": "submitted_resolution",
            "resolution_type": resolved_resolution_type,
            "status": "submitted",
        }

    # ------------------------------------------------------------------
    # send_patient_communication
    # State transition happens HERE — before reward computation
    # ------------------------------------------------------------------
    if action_type == "send_patient_communication":
        message_type = ap.get("message_type", "explanation")
        old_state = _get_current_patient_state(
            episode_id, db, params.initial_patient_state
        )
        transitions = STATE_TRANSITIONS.get(
            message_type, STATE_TRANSITIONS["explanation"]
        )
        new_state = transitions.get(old_state, old_state)

        # Write the state update BEFORE returning (reward reads it after)
        _set_patient_state(episode_id, step_number, new_state, db)
        content = {
            "message_type": message_type,
            "message_text": ap.get("message_text", ""),
            "old_patient_state": old_state,
            "new_patient_state": new_state,
            "sent_at_step": step_number,
        }
        artifact_id = _insert_write_artifact(
            episode_id, "patient_communication", content, step_number, db
        )
        db.commit()
        return {
            "artifact_id": artifact_id,
            "artifact_type": "patient_communication",
            "message_type": message_type,
            "old_patient_state": old_state,
            "new_patient_state": new_state,
        }

    # ------------------------------------------------------------------
    # notify_provider
    # ------------------------------------------------------------------
    if action_type == "notify_provider":
        content = {
            "provider_id": params.provider_id,
            "notification_type": ap.get("notification_type", "dispute_filed"),
            "message": ap.get("message", ""),
            "sent_at_step": step_number,
        }
        artifact_id = _insert_write_artifact(
            episode_id, "provider_notice", content, step_number, db
        )
        db.commit()
        return {
            "artifact_id": artifact_id,
            "artifact_type": "provider_notice",
            "provider_id": params.provider_id,
        }

    # ------------------------------------------------------------------
    # reject_counter_argument (Task 3 Phase 2)
    # ------------------------------------------------------------------
    if action_type == "reject_counter_argument":
        counter_index = ap.get("counter_index", 0)
        rejection_reasoning = ap.get("rejection_reasoning", "")
        cited_artifact_ids = ap.get("cited_artifact_ids", [])
        content = {
            "counter_index": counter_index,
            "rejection_reasoning": rejection_reasoning,
            "cited_artifact_ids": cited_artifact_ids,
            "dispute_maintained": True,
            "rejected_at_step": step_number,
        }
        artifact_id = _insert_write_artifact(
            episode_id, "counter_rejection", content, step_number, db
        )
        db.commit()
        return {
            "artifact_id": artifact_id,
            "artifact_type": "counter_rejection",
            "counter_index": counter_index,
            "dispute_maintained": True,
        }

    # ------------------------------------------------------------------
    # write_audit_entry
    # ------------------------------------------------------------------
    if action_type == "write_audit_entry":
        content = {
            "summary": ap.get("summary", ""),
            "outcome_code": ap.get("outcome_code"),
            "written_at_step": step_number,
        }
        artifact_id = _insert_write_artifact(
            episode_id, "audit_entry", content, step_number, db
        )
        db.commit()
        return {
            "artifact_id": artifact_id,
            "artifact_type": "audit_entry",
            "summary": ap.get("summary", ""),
        }

    # ------------------------------------------------------------------
    # close_case — triggers grader in step(), no artifact needed
    # ------------------------------------------------------------------
    if action_type == "close_case":
        content = {
            "outcome_code": ap.get("outcome_code", "resolved"),
            "closed_at_step": step_number,
        }
        db.commit()
        return {
            "artifact_type": "close_case",
            "outcome_code": ap.get("outcome_code", "resolved"),
            "closed": True,
        }

    raise ValueError(f"Unknown write action_type: {action_type!r}")
