"""Read action execution for Clarus.

execute_read_action() is called by ClarusEnv.step() for all read-type
actions.  It:
1. Checks the action type is in AGENT_READABLE_ARTIFACT_TYPES.
2. Checks if this episode already has an artifact of this type
   (to determine first_fetch vs duplicate_fetch for reward computation).
3. Calls get_seeded_payload() for the genuine payload, or
   get_distractor_payload() if the episode designates this type as a distractor.
4. Inserts the artifact row and returns the payload dict with artifact_id.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Dict, Optional

from server.scenario.params import EpisodeParams
from server.tools.distractors import get_distractor_payload, is_distractor
from server.tools.payloads import ACTION_TO_ARTIFACT_TYPE, get_seeded_payload


# Frozenset of artifact types the agent is allowed to read.
# "ground_truth" is explicitly absent — it must NEVER appear here.
AGENT_READABLE_ARTIFACT_TYPES: frozenset = frozenset(
    ACTION_TO_ARTIFACT_TYPE.values()
)


def artifact_already_fetched(
    episode_id: str,
    artifact_type: str,
    db: sqlite3.Connection,
) -> bool:
    """Return True if an agent-fetched artifact of this type already exists.

    Only counts artifacts created by the agent (source='agent'), not
    ground_truth or environment-seeded artifacts.

    Args:
        episode_id: Current episode ID.
        artifact_type: Artifact type to check.
        db: SQLite runtime connection.

    Returns:
        True if a prior fetch of this type exists in this episode.
    """
    row = db.execute(
        "SELECT 1 FROM episode_artifacts "
        "WHERE episode_id=? AND artifact_type=? AND source='agent' LIMIT 1",
        (episode_id, artifact_type),
    ).fetchone()
    return row is not None


def execute_read_action(
    action_type: str,
    action_parameters: Dict[str, Any],
    episode_id: str,
    step_number: int,
    params: EpisodeParams,
    db: sqlite3.Connection,
    ref_db: sqlite3.Connection,
) -> Dict[str, Any]:
    """Execute a read action, insert an artifact, and return the payload.

    The returned dict includes artifact_id so the agent can cite it later.
    All content is stored as JSON in the episode_artifacts table.

    Args:
        action_type: One of the read action names.
        action_parameters: Dict of action-specific parameters from the agent.
        episode_id: Current episode identifier.
        step_number: Current step number (stored as created_at).
        params: EpisodeParams for this episode.
        db: Runtime SQLite connection.
        ref_db: Reference SQLite connection (CPT/NCCI lookups).

    Returns:
        Dict with artifact_id, artifact_type, and all payload fields.

    Raises:
        ValueError: If action_type is not in AGENT_READABLE_ARTIFACT_TYPES.
    """
    artifact_type = ACTION_TO_ARTIFACT_TYPE.get(action_type)
    if artifact_type is None or artifact_type not in AGENT_READABLE_ARTIFACT_TYPES:
        raise ValueError(
            f"Action {action_type!r} is not a permitted read action. "
            f"Permitted types: {sorted(AGENT_READABLE_ARTIFACT_TYPES)}"
        )

    # Build payload — distractor or genuine
    if is_distractor(artifact_type, params.distractor_artifact_type):
        payload = get_distractor_payload(artifact_type)
        source = "agent_distractor"
    else:
        payload = get_seeded_payload(
            action_type, params, ref_db, action_parameters
        )
        source = "agent"

    # Insert artifact row
    cursor = db.execute(
        "INSERT INTO episode_artifacts "
        "(episode_id, artifact_type, source, content, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (episode_id, artifact_type, source, json.dumps(payload), step_number),
    )
    artifact_id = cursor.lastrowid
    db.commit()

    # Return payload with artifact_id prepended
    result = {"artifact_id": artifact_id, "artifact_type": artifact_type}
    result.update(payload)
    return result
