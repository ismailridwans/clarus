"""ClarusEnv — the core RL environment.

Implements reset() / step() / state() following the OpenEnv interface.
The environment is stateful: one episode at a time per instance.

Canonical step() execution order (never reorder):
1. Rate limit check
2. Validate action (submit_resolution reads resolution_type FROM draft)
3. check_and_log_compliance()
4. Execute action (state transition inside write actions)
5. compute_structural_reward() — reads NEW state from DB
6. log_action() + increment_api_calls()
7. run_grader() at close_case
8. Return StepResult
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

from server.grader.runner import run_grader
from server.models import (
    CheckResult,
    ClarusAction,
    ClarusObservation,
    StateResponse,
)
from server.scenario.generator import TODAY, generate
from server.scenario.params import EpisodeParams
from server.tools.compliance import check_and_log_compliance
from server.tools.distractors import is_distractor
from server.tools.rate_limits import (
    RateState,
    get_cooldown_status,
    get_rate_limited_tools,
    init_rate_state,
    is_rate_limited,
    get_cooldown_remaining,
)
from server.tools.reads import artifact_already_fetched, execute_read_action
from server.tools.writes import (
    execute_write_action,
    validate_and_enrich_submit_resolution,
)


# ------------------------------------------------------------------
# Structural reward schedule
# ------------------------------------------------------------------

STRUCTURAL_REWARDS: Dict[str, float] = {
    "authenticate_patient":         +0.05,
    "first_fetch_of_type":          +0.03,
    "duplicate_fetch":              -0.02,
    "write_diagnosis_2plus_ids":    +0.05,
    "write_diagnosis_1_id":         +0.02,
    "write_diagnosis_no_ids":       -0.03,
    "patient_state_improved":       +0.05,
    "patient_state_unchanged":      +0.00,
    "check_deadline_before_submit": +0.03,
    "check_deadline_after_submit":  -0.01,
    "draft_resolution":             +0.02,
    "notify_provider":              +0.02,
    "write_audit_entry":            +0.03,
    "action_error":                 -0.02,
    "rate_limited":                 -0.02,
    "distractor_fetch":             -0.01,
}

# Read actions that benefit from first-fetch / duplicate tracking
READ_ACTIONS = frozenset(
    {
        "fetch_claim_record",
        "fetch_eob",
        "fetch_provider_record",
        "fetch_payment_ledger",
        "fetch_plan_document",
        "lookup_procedure_code",
        "fetch_facility_record",
        "fetch_payment_processor_log",
        "check_regulatory_rule",
        "check_deadline",
    }
)

WRITE_ACTIONS = frozenset(
    {
        "authenticate_patient",
        "write_diagnosis",
        "draft_resolution",
        "submit_resolution",
        "send_patient_communication",
        "notify_provider",
        "reject_counter_argument",
        "write_audit_entry",
        "close_case",
    }
)

ALL_VALID_ACTIONS = READ_ACTIONS | WRITE_ACTIONS


class StepResult:
    """Internal result from a single step."""

    def __init__(
        self,
        observation: ClarusObservation,
        reward: float,
        done: bool,
        info: Dict[str, Any],
    ):
        """Initialise a StepResult."""
        self.observation = observation
        self.reward = reward
        self.done = done
        self.info = info


class ClarusEnv:
    """Clarus RL environment for healthcare billing dispute resolution.

    Usage:
        env = ClarusEnv(ref_db=ref_db)
        obs = await env.reset("deductive_liability", seed=1001)
        result = await env.step(ClarusAction(action_type="authenticate_patient"))
    """

    def __init__(self, ref_db: sqlite3.Connection) -> None:
        """Initialise the environment with a loaded reference DB.

        Args:
            ref_db: Open SQLite connection with all reference tables loaded.
                    Must remain open for the lifetime of the environment.
        """
        self.ref_db = ref_db
        self.db: Optional[sqlite3.Connection] = None

        # Episode state
        self.episode_id: Optional[str] = None
        self.task_name: Optional[str] = None
        self.seed: Optional[int] = None
        self.params: Optional[EpisodeParams] = None
        self.step_number: int = 0
        self.api_calls_used: int = 0
        self.rate_state: RateState = {}
        self.done: bool = False
        self._last_observation: Optional[ClarusObservation] = None

    def _new_db(self) -> sqlite3.Connection:
        """Create a fresh in-memory SQLite DB for this episode's runtime data."""
        from server.schema import create_tables

        db = sqlite3.connect(":memory:", check_same_thread=False)
        db.row_factory = sqlite3.Row
        create_tables(db)
        return db

    def _seed_ground_truth(self) -> None:
        """Insert the ground_truth row — must be called once at reset().

        Mechanism 1 of the ground_truth non-leak guarantee:
        source='environment' and artifact_type='ground_truth' are absent
        from AGENT_READABLE_ARTIFACT_TYPES.
        """
        from server.scenario.generator import _derive_resolution_type, _derive_responsible_party

        p = self.params
        gt = {
            "correct_refund_amount": round(p.correct_refund, 2),
            "correct_qpa_amount": round(p.qpa_amount, 2),
            "correct_responsible_party": _derive_responsible_party(p.task_name),
            "correct_resolution_type": _derive_resolution_type(p.task_name),
            "correct_appeal_reason_set": p.appeal_reason_set,
            "fabricated_field": (
                "good_faith_estimate_date"
                if p.task_name == "adversarial_fabrication"
                else None
            ),
            "contradiction_source": (
                "processor_log"
                if p.task_name == "adversarial_fabrication"
                else None
            ),
            "task_name": p.task_name,
            "episode_seed": p.episode_seed,
        }
        self.db.execute(
            "INSERT INTO episode_artifacts "
            "(episode_id, artifact_type, source, content, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (self.episode_id, "ground_truth", "environment", json.dumps(gt), 0),
        )
        self.db.commit()

    def _get_patient_state(self) -> str:
        """Return the current patient emotional state."""
        row = self.db.execute(
            "SELECT content FROM episode_artifacts "
            "WHERE episode_id=? AND artifact_type='patient_state' "
            "ORDER BY id DESC LIMIT 1",
            (self.episode_id,),
        ).fetchone()
        if row:
            return json.loads(row[0]).get("state", self.params.initial_patient_state)
        return self.params.initial_patient_state

    def _build_observation(
        self,
        last_action_type: Optional[str],
        last_action_result: Optional[Dict],
        last_action_error: Optional[str],
        step_reward: float,
    ) -> ClarusObservation:
        """Build the agent-facing observation from current DB state.

        Mechanism 2 of the ground_truth non-leak guarantee:
        This method never queries ground_truth content.

        Args:
            last_action_type: Action type of the most recent action.
            last_action_result: Payload returned by the most recent action.
            last_action_error: Error message if the action failed.
            step_reward: Structural reward earned this step.

        Returns:
            ClarusObservation populated from current DB state.
        """
        # Build action_log_summary from action_log table — type + ok/error only
        rows = self.db.execute(
            "SELECT step_number, action_type, result, error "
            "FROM action_log WHERE episode_id=? ORDER BY id",
            (self.episode_id,),
        ).fetchall()
        summary = []
        for r in rows:
            step_n = r[0]
            atype = r[1]
            if r[3]:  # error
                entry = f"step{step_n}: {atype} → error"
            elif r[2]:  # result exists
                result_data = json.loads(r[2]) if r[2] else {}
                aid = result_data.get("artifact_id")
                if aid:
                    entry = f"step{step_n}: {atype} → artifact_id={aid}"
                else:
                    entry = f"step{step_n}: {atype} → ok"
            else:
                entry = f"step{step_n}: {atype} → ok"
            summary.append(entry)

        return ClarusObservation(
            step_number=self.step_number,
            api_calls_used=self.api_calls_used,
            api_call_budget=18,
            rate_limited_tools=get_rate_limited_tools(
                self.rate_state, self.step_number
            ),
            cooldown_steps=get_cooldown_status(self.rate_state, self.step_number),
            case_id=self.episode_id,
            patient_complaint=self.params.patient_complaint,
            patient_name=self.params.patient_name,
            patient_emotional_state=self._get_patient_state(),
            last_action_type=last_action_type,
            last_action_result=last_action_result,
            last_action_error=last_action_error,
            action_log_summary=summary,
            step_reward=step_reward,
            done=self.done,
        )

    def _validate_action(self, action: ClarusAction) -> Optional[str]:
        """Validate a non-submit action.  Returns error string or None."""
        if action.action_type not in ALL_VALID_ACTIONS:
            return f"Unknown action_type: {action.action_type!r}"
        return None

    def _log_action(
        self,
        action: ClarusAction,
        result: Optional[Dict],
        error: Optional[str],
    ) -> None:
        """Insert a row into action_log."""
        self.db.execute(
            "INSERT INTO action_log "
            "(episode_id, step_number, action_type, parameters, result, error) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                self.episode_id,
                self.step_number,
                action.action_type,
                json.dumps(action.parameters),
                json.dumps(result) if result else None,
                error,
            ),
        )
        self.db.commit()

    def _compute_structural_reward(
        self,
        action_type: str,
        result: Optional[Dict],
        error: Optional[str],
        was_first_fetch: bool,
        was_distractor: bool,
        old_patient_state: str,
        result_obj: Optional[Dict] = None,
    ) -> float:
        """Compute the structural reward for the current step.

        Reads NEW state from DB after state transition.  Never reads ground_truth.

        Args:
            action_type: The action type executed.
            result: The result payload (None on error).
            error: Error string (None on success).
            was_first_fetch: True if this was the first fetch of this artifact type.
            was_distractor: True if this was a distractor fetch.
            old_patient_state: Patient state BEFORE send_patient_communication.
            result_obj: The raw result dict (for diagnosis ID count etc.).

        Returns:
            Float reward value.
        """
        if error:
            return STRUCTURAL_REWARDS["action_error"]

        if action_type == "authenticate_patient":
            return STRUCTURAL_REWARDS["authenticate_patient"]

        if action_type in READ_ACTIONS:
            if was_distractor:
                return STRUCTURAL_REWARDS["distractor_fetch"]
            if was_first_fetch:
                return STRUCTURAL_REWARDS["first_fetch_of_type"]
            return STRUCTURAL_REWARDS["duplicate_fetch"]

        if action_type == "write_diagnosis":
            if result_obj:
                ids = result_obj.get("evidence_artifact_ids", [])
                n = len(ids)
                if n >= 2:
                    return STRUCTURAL_REWARDS["write_diagnosis_2plus_ids"]
                if n == 1:
                    return STRUCTURAL_REWARDS["write_diagnosis_1_id"]
            return STRUCTURAL_REWARDS["write_diagnosis_no_ids"]

        if action_type == "send_patient_communication":
            new_state = self._get_patient_state()
            from server.tools.writes import STATE_ORDINAL
            if STATE_ORDINAL.get(new_state, 1) < STATE_ORDINAL.get(old_patient_state, 1):
                return STRUCTURAL_REWARDS["patient_state_improved"]
            return STRUCTURAL_REWARDS["patient_state_unchanged"]

        if action_type == "check_deadline":
            # Reward only if checked BEFORE any submission
            submitted_exists = self.db.execute(
                "SELECT 1 FROM episode_artifacts "
                "WHERE episode_id=? AND artifact_type='submitted_resolution' "
                "  AND source='agent' LIMIT 1",
                (self.episode_id,),
            ).fetchone()
            if submitted_exists:
                return STRUCTURAL_REWARDS["check_deadline_after_submit"]
            return STRUCTURAL_REWARDS["check_deadline_before_submit"]

        if action_type == "draft_resolution":
            return STRUCTURAL_REWARDS["draft_resolution"]

        if action_type == "notify_provider":
            return STRUCTURAL_REWARDS["notify_provider"]

        if action_type == "write_audit_entry":
            return STRUCTURAL_REWARDS["write_audit_entry"]

        # submit_resolution, reject_counter_argument, close_case: no structural reward
        return 0.0

    async def reset(
        self,
        task_name: str = "deductive_liability",
        seed: Optional[int] = None,
    ) -> ClarusObservation:
        """Reset the environment for a new episode.

        Args:
            task_name: One of the three task names.
            seed: Episode seed.  If None, a default train seed is used.

        Returns:
            Initial ClarusObservation.
        """
        import random as _random

        valid_tasks = {
            "deductive_liability": range(1001, 1016),
            "abductive_conflict": range(2001, 2016),
            "adversarial_fabrication": range(3001, 3016),
        }
        if task_name not in valid_tasks:
            raise ValueError(f"Unknown task_name: {task_name!r}")

        if seed is None:
            seed = _random.choice(list(valid_tasks[task_name]))

        # Fresh runtime DB for this episode
        if self.db is not None:
            try:
                self.db.close()
            except Exception:
                pass
        self.db = self._new_db()

        self.episode_id = str(uuid.uuid4())
        self.task_name = task_name
        self.seed = seed
        self.step_number = 0
        self.api_calls_used = 0
        self.done = False

        # Generate episode parameters
        self.params = generate(task_name, seed, self.ref_db)

        # Initialise rate state
        self.rate_state = init_rate_state(
            self.params.rate_limited_at_start,
            self.params.rate_limit_cooldown_steps,
            start_step=1,
        )

        # Seed ground truth (never exposed to agent)
        self._seed_ground_truth()

        obs = self._build_observation(
            last_action_type=None,
            last_action_result=None,
            last_action_error=None,
            step_reward=0.0,
        )
        self._last_observation = obs
        return obs

    async def step(self, action: ClarusAction) -> StepResult:
        """Execute one action and return the result.

        Canonical execution order — see module docstring.

        Args:
            action: The agent's chosen action.

        Returns:
            StepResult with observation, reward, done, info.
        """
        if self.done:
            raise RuntimeError("Episode is done.  Call reset() to start a new episode.")
        if self.params is None:
            raise RuntimeError("Call reset() before step().")

        self.step_number += 1

        # Capture patient state before potential mutation
        old_patient_state = self._get_patient_state()

        # -----------------------------------------------------------
        # 1. Rate limit check
        # -----------------------------------------------------------
        if is_rate_limited(action.action_type, self.step_number, self.rate_state):
            remaining = get_cooldown_remaining(
                action.action_type, self.step_number, self.rate_state
            )
            error_msg = (
                f"RATE_LIMITED: {action.action_type} available "
                f"in {remaining} step(s)"
            )
            self._log_action(action, result=None, error=error_msg)
            self.api_calls_used += 1
            reward = STRUCTURAL_REWARDS["rate_limited"]
            obs = self._build_observation(
                last_action_type=action.action_type,
                last_action_result={"error": error_msg},
                last_action_error=error_msg,
                step_reward=reward,
            )
            self._last_observation = obs
            return StepResult(
                observation=obs,
                reward=reward,
                done=False,
                info={"episode_score": None},
            )

        # -----------------------------------------------------------
        # 2. Validate action
        # -----------------------------------------------------------
        error: Optional[str] = None
        resolved_resolution_type: Optional[str] = None

        if action.action_type == "submit_resolution":
            error, resolved_resolution_type = validate_and_enrich_submit_resolution(
                action.parameters, self.episode_id, self.db
            )
        else:
            error = self._validate_action(action)

        if error:
            self._log_action(action, result=None, error=error)
            self.api_calls_used += 1
            reward = STRUCTURAL_REWARDS["action_error"]
            obs = self._build_observation(
                last_action_type=action.action_type,
                last_action_result={"error": error},
                last_action_error=error,
                step_reward=reward,
            )
            self._last_observation = obs
            return StepResult(
                observation=obs,
                reward=reward,
                done=False,
                info={"episode_score": None},
            )

        # -----------------------------------------------------------
        # 3. Compliance check (before execution)
        # -----------------------------------------------------------
        check_and_log_compliance(
            action_type=action.action_type,
            episode_id=self.episode_id,
            db=self.db,
            step_number=self.step_number,
            resolution_type=resolved_resolution_type,
        )

        # -----------------------------------------------------------
        # 4. Execute action (state transitions happen inside)
        # -----------------------------------------------------------
        result: Optional[Dict] = None
        exec_error: Optional[str] = None
        was_first_fetch = False
        was_distractor_fetch = False

        try:
            if action.action_type in READ_ACTIONS:
                # Track first-fetch vs duplicate BEFORE insertion
                from server.tools.payloads import ACTION_TO_ARTIFACT_TYPE
                artifact_type = ACTION_TO_ARTIFACT_TYPE.get(action.action_type, "")
                already_fetched = artifact_already_fetched(
                    self.episode_id, artifact_type, self.db
                )
                was_first_fetch = not already_fetched
                was_distractor_fetch = is_distractor(
                    artifact_type, self.params.distractor_artifact_type
                )

                result = execute_read_action(
                    action_type=action.action_type,
                    action_parameters=action.parameters,
                    episode_id=self.episode_id,
                    step_number=self.step_number,
                    params=self.params,
                    db=self.db,
                    ref_db=self.ref_db,
                )
            else:
                result = await execute_write_action(
                    action_type=action.action_type,
                    action_parameters=action.parameters,
                    episode_id=self.episode_id,
                    step_number=self.step_number,
                    params=self.params,
                    db=self.db,
                    resolved_resolution_type=resolved_resolution_type,
                )
        except Exception as exc:
            exec_error = str(exc)

        if exec_error:
            self._log_action(action, result=None, error=exec_error)
            self.api_calls_used += 1
            reward = STRUCTURAL_REWARDS["action_error"]
            obs = self._build_observation(
                last_action_type=action.action_type,
                last_action_result={"error": exec_error},
                last_action_error=exec_error,
                step_reward=reward,
            )
            self._last_observation = obs
            return StepResult(
                observation=obs,
                reward=reward,
                done=False,
                info={"action_error": exec_error, "rate_limited": False,
                      "episode_score": None, "check_results": None},
            )

        # -----------------------------------------------------------
        # 5. Structural reward (reads NEW state — transitions already done)
        # -----------------------------------------------------------
        reward = self._compute_structural_reward(
            action_type=action.action_type,
            result=result,
            error=None,
            was_first_fetch=was_first_fetch,
            was_distractor=was_distractor_fetch,
            old_patient_state=old_patient_state,
            result_obj=result,
        )

        # -----------------------------------------------------------
        # 6. Log + increment
        # -----------------------------------------------------------
        self._log_action(action, result=result, error=None)
        self.api_calls_used += 1

        # -----------------------------------------------------------
        # 7. Grader fires at close_case
        # -----------------------------------------------------------
        episode_score: Optional[float] = None
        check_results: Optional[List[CheckResult]] = None
        done = action.action_type == "close_case"
        if done:
            self.done = True
            episode_score, check_results = run_grader(
                self.episode_id, self.db, self.task_name
            )

        # -----------------------------------------------------------
        # 8. Build and return observation
        # -----------------------------------------------------------
        obs = self._build_observation(
            last_action_type=action.action_type,
            last_action_result=result,
            last_action_error=None,
            step_reward=reward,
        )
        self._last_observation = obs

        return StepResult(
            observation=obs,
            reward=reward,
            done=done,
            info={
                "episode_score": episode_score,
                "check_results": check_results,
            },
        )

    async def close(self) -> None:
        """Close the episode DB connection. Called by inference scripts for cleanup."""
        if self.db is not None:
            try:
                self.db.close()
            except Exception:
                pass
            self.db = None

    def state(self) -> StateResponse:
        """Return current environment state for GET /state."""
        return StateResponse(
            episode_id=self.episode_id,
            task_name=self.task_name,
            seed=self.seed,
            step_number=self.step_number,
            done=self.done,
            observation=self._last_observation,
        )
