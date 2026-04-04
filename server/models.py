"""Pydantic models for Clarus OpenEnv API.

All API request/response types are defined here.
ClarusObservation is the agent-facing observation returned from /reset and /step.
ClarusAction is the agent-facing action accepted by /step.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ------------------------------------------------------------------
# Observation
# ------------------------------------------------------------------


class ClarusObservation(BaseModel):
    """Agent-facing observation returned after every step and reset.

    The agent sees ONLY what is in this model — no ground_truth content,
    no boolean progress flags, no prior-step artifact content.
    """

    # Session metadata
    step_number: int
    api_calls_used: int
    api_call_budget: int = 18
    rate_limited_tools: List[str] = Field(default_factory=list)
    cooldown_steps: Dict[str, int] = Field(default_factory=dict)

    # Case context — set at reset(), never changes
    case_id: str
    patient_complaint: str
    patient_name: str
    patient_emotional_state: Literal["calm", "frustrated", "distressed"]

    # Last action result — only the most recent step's payload
    last_action_type: Optional[str] = None
    last_action_result: Optional[Dict[str, Any]] = None
    last_action_error: Optional[str] = None

    # Action history summary — action_type + ok/error only, no content
    # Format: "step1: authenticate_patient → ok"
    # "step2: fetch_eob → artifact_id=3"
    action_log_summary: List[str] = Field(default_factory=list)

    # Per-step structural reward
    step_reward: float = 0.0

    # Episode termination flag
    done: bool = False


# ------------------------------------------------------------------
# Action
# ------------------------------------------------------------------


class ClarusAction(BaseModel):
    """Agent action submitted to /step."""

    action_type: str = Field(
        ...,
        description=(
            "One of the 19 permitted action types: "
            "authenticate_patient, fetch_claim_record, fetch_eob, "
            "fetch_provider_record, fetch_payment_ledger, fetch_plan_document, "
            "lookup_procedure_code, fetch_facility_record, "
            "fetch_payment_processor_log, check_regulatory_rule, check_deadline, "
            "write_diagnosis, draft_resolution, submit_resolution, "
            "send_patient_communication, notify_provider, "
            "reject_counter_argument, write_audit_entry, close_case"
        ),
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Action-specific parameters (varies by action_type).",
    )


# ------------------------------------------------------------------
# Reset request / response
# ------------------------------------------------------------------


class ResetRequest(BaseModel):
    """Request body for POST /reset."""

    task_name: str = Field(
        default="deductive_liability",
        description=(
            "Task to initialise: deductive_liability, abductive_conflict, "
            "or adversarial_fabrication."
        ),
    )
    seed: Optional[int] = Field(
        default=None,
        description=(
            "Episode seed.  If None, a seed is drawn from the train split."
        ),
    )


class ResetResponse(BaseModel):
    """Response from POST /reset."""

    episode_id: str
    task_name: str
    seed: int
    observation: ClarusObservation


# ------------------------------------------------------------------
# Step request / response
# ------------------------------------------------------------------


class StepRequest(BaseModel):
    """Request body for POST /step."""

    action: ClarusAction


class CheckResult(BaseModel):
    """Result of a single grader check."""

    description: str
    passed: bool
    actual: Optional[Any] = None
    expected: Optional[Any] = None


class StepResponse(BaseModel):
    """Response from POST /step."""

    observation: ClarusObservation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


# ------------------------------------------------------------------
# State
# ------------------------------------------------------------------


class StateResponse(BaseModel):
    """Response from GET /state."""

    episode_id: Optional[str]
    task_name: Optional[str]
    seed: Optional[int]
    step_number: int
    done: bool
    observation: Optional[ClarusObservation] = None
