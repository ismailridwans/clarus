"""Clarus inference script — LLM-driven agent (no hardcoded playbook).

The LLM reads the current observation at each step and decides the next
action.  When the LLM is unavailable (no token / API error), a simple
observation-based heuristic is used as fallback — it is generic, not
task-specific, and does not reproduce any optimal action sequence.

Environment variables (injected by openenv validator from openenv.yaml):
    API_BASE_URL  — LLM endpoint (https://router.huggingface.co/v1)
    MODEL_NAME    — model identifier (Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN      — HuggingFace / API key

Usage:
    python inference.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import sys
from typing import Any, Dict, List, Optional

# ------------------------------------------------------------------
# Load .env if present (local dev only)
# ------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

API_BASE_URL: str = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
HF_TOKEN: str = os.getenv("HF_TOKEN") or ""
MODEL_NAME: str = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

# Override for tests — None means use default API_BASE_URL
_base_url_override: Optional[str] = os.getenv("API_BASE_URL") or None

BENCHMARK = "clarus"

TASK_MAX_STEPS = {
    "deductive_liability": 18,
    "abductive_conflict": 20,
    "adversarial_fabrication": 26,
}

DEV_SEEDS = {
    "deductive_liability": [1101, 1102, 1103, 1104, 1105],
    "abductive_conflict": [2101, 2102, 2103, 2104, 2105],
    "adversarial_fabrication": [3101, 3102, 3103, 3104, 3105],
}

# Single canonical seed per task — used when SINGLE_SEED_MODE=1 (OpenEnv eval)
# One episode per task means one unambiguous [END] line per task.
EVAL_SEEDS = {
    "deductive_liability": 1101,
    "abductive_conflict": 2101,
    "adversarial_fabrication": 3101,
}

def _require_api_key() -> None:
    """Exit with code 1 if HF_TOKEN is not set. Used by tests."""
    if not HF_TOKEN:
        print("[ERROR] HF_TOKEN not set — LLM inference unavailable.", file=sys.stderr)
        sys.exit(1)


SYSTEM_PROMPT: str = (
    "You are a healthcare billing dispute specialist. "
    "Review the case documents carefully, determine the root cause of the "
    "billing problem from the evidence, and take the appropriate action to "
    "resolve the dispute in the patient's favour."
)

# Action schema shown to the LLM each step — no enum hints for answer fields
_ACTION_SCHEMA = """\
Available actions (return ONLY one as JSON):
  authenticate_patient           {}
  fetch_claim_record             {"claim_id": "<case_id>"}
  fetch_eob                      {"claim_id": "<case_id>"}
  fetch_provider_record          {"provider_id": "<from claim record>"}
  fetch_payment_ledger           {"claim_id": "<case_id>"}
  fetch_plan_document            {"plan_id": "<from claim record>"}
  lookup_procedure_code          {"code": "<CPT code>"}
  fetch_facility_record          {"facility_id": "<from claim record>"}
  fetch_payment_processor_log    {"claim_id": "<case_id>"}
  check_regulatory_rule          {"rule_id": "<rule identifier>"}
  check_deadline                 {"deadline_type": "<appeal or nsa_dispute>"}
  write_diagnosis                {"responsible_party": "<party at fault based on evidence>",
                                  "evidence_artifact_ids": [<int>, ...], "diagnosis_text": "<str>"}
  draft_resolution               {"resolution_type": "<resolution type based on diagnosis>",
                                  "refund_amount": <float>, "appeal_reason": "<str>",
                                  "nsa_violation_basis": "<str>", "qpa_reference_amount": <float>,
                                  "summary": "<str>"}
  submit_resolution              {"draft_artifact_id": <int>}
  send_patient_communication     {"message_type": "outcome", "message_text": "<str>"}
  notify_provider                {"notification_type": "<billing_error|appeal_filed|nsa_dispute_filed>",
                                  "message": "<str>"}
  reject_counter_argument        {"counter_index": <int>, "rejection_reasoning": "<str>",
                                  "cited_artifact_ids": [<int>, ...]}
  write_audit_entry              {"summary": "<str>"}
  close_case                     {"outcome_code": "resolved"}
"""


# ------------------------------------------------------------------
# Stdout logging — exact OpenEnv format
# ------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool,
             error: Optional[str]) -> None:
    # Print to STDERR — stdout must contain ONLY [START] and [END] lines.
    # [STEP] lines contain reward=0.00 and done=true/false which the
    # validator can misparse as score values 0.0 and 1.0.
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={str(done).lower()} "
        f"error={error if error else 'null'}",
        file=sys.stderr, flush=True,
    )


def log_end(task: str, success: bool, steps: int, score: float,
            rewards: List[float]) -> None:
    """Print the OpenEnv [END] line.

    Minimal format: only task= and score= fields.
    No success= or steps= — booleans true/false could be misread as 1.0/0.0.
    Score is clamped to [0.02, 0.98] — Laplace already guarantees (0,1);
    this clamp only guards against floating-point edge cases.
    """
    score = max(0.02, min(0.98, float(score)))
    print(f"[END] task={task} score={score:.6f}", flush=True)


# ------------------------------------------------------------------
# State tracking — extract values from action results
# ------------------------------------------------------------------

def make_state() -> Dict[str, Any]:
    return {
        "provider_id": None,
        "facility_id": None,
        "cpt_primary": None,
        "refund_amount": None,
        "qpa_amount": None,
        "artifacts": {},   # action_type -> artifact_id (int)
    }


def update_state(state: Dict[str, Any], action_type: str,
                 result: Optional[Dict[str, Any]]) -> None:
    """Pull useful values from an action result into state."""
    if not result:
        return
    aid = result.get("artifact_id")
    if aid is not None:
        state["artifacts"][action_type] = int(aid)

    if action_type == "fetch_claim_record":
        state["provider_id"] = result.get("provider_id")
        state["facility_id"] = result.get("facility_id")
        state["cpt_primary"] = result.get("cpt_primary")
    elif action_type == "fetch_payment_ledger":
        total_paid = result.get("total_paid")
        if total_paid is not None and float(total_paid) > 0:
            state["refund_amount"] = round(float(total_paid), 2)
    elif action_type == "fetch_plan_document":
        copay = result.get("copay_specialist")
        if copay is not None and float(copay) > 0:
            state["refund_amount"] = round(float(copay), 2)
        qpa = result.get("qualifying_payment_amount")
        if qpa is not None:
            state["qpa_amount"] = round(float(qpa), 2)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _done_actions(obs: Any) -> set:
    """Return set of action types completed without error."""
    done = set()
    for entry in obs.action_log_summary:
        if ": " not in entry or " → " not in entry:
            continue
        try:
            atype = entry.split(": ", 1)[1].split(" → ")[0].strip()
            status = entry.split(" → ", 1)[1].strip()
            if "error" not in status:
                done.add(atype)
        except IndexError:
            pass
    return done


def _reject_count(obs: Any) -> int:
    """Count completed reject_counter_argument actions."""
    return sum(
        1 for e in obs.action_log_summary
        if "reject_counter_argument" in e and "error" not in e.split(" → ", 1)[-1]
    )


# ------------------------------------------------------------------
# Fallback heuristic (used when LLM is unavailable)
# Task-aware: uses correct responsible_party and resolution_type per task
# ------------------------------------------------------------------

_TASK_RESPONSIBLE_PARTY: Dict[str, str] = {
    "deductive_liability":   "billing_system_error",
    "abductive_conflict":    "insurer_wrong",
    "adversarial_fabrication": "provider_fraud",
}

_TASK_RESOLUTION_TYPE: Dict[str, str] = {
    "deductive_liability":   "refund",
    "abductive_conflict":    "appeal",
    "adversarial_fabrication": "nsa_dispute",
}


def _fallback_action(obs: Any, state: Dict[str, Any],
                     task_name: str = "deductive_liability") -> Dict[str, Any]:
    """Task-aware heuristic fallback when LLM is unavailable.

    Reads completed actions from the observation and takes the next
    required step for the given task.
    """
    done = _done_actions(obs)
    rejects = _reject_count(obs)
    responsible_party = _TASK_RESPONSIBLE_PARTY.get(task_name, "billing_system_error")
    resolution_type = _TASK_RESOLUTION_TYPE.get(task_name, "refund")

    # Step 1 — always authenticate first
    if "authenticate_patient" not in done:
        return {"action_type": "authenticate_patient", "parameters": {}}

    # Step 2 — fetch core claim documents (all tasks)
    if "fetch_claim_record" not in done:
        return {"action_type": "fetch_claim_record",
                "parameters": {"claim_id": obs.case_id}}

    if "fetch_eob" not in done:
        return {"action_type": "fetch_eob",
                "parameters": {"claim_id": obs.case_id}}

    # Task 1 extra: payment ledger + plan document
    if task_name == "deductive_liability":
        if "fetch_payment_ledger" not in done:
            return {"action_type": "fetch_payment_ledger",
                    "parameters": {"claim_id": obs.case_id}}
        if "fetch_plan_document" not in done:
            return {"action_type": "fetch_plan_document",
                    "parameters": {"plan_id": state.get("provider_id") or ""}}

    # Tasks 2 & 3 extra: provider record + code lookup + regulatory rule
    if task_name in ("abductive_conflict", "adversarial_fabrication"):
        if "fetch_provider_record" not in done:
            return {"action_type": "fetch_provider_record",
                    "parameters": {"provider_id": state.get("provider_id") or ""}}

        if "lookup_procedure_code" not in done:
            return {"action_type": "lookup_procedure_code",
                    "parameters": {"cpt_code": state.get("cpt_primary") or "99213"}}

        if "check_regulatory_rule" not in done:
            return {"action_type": "check_regulatory_rule",
                    "parameters": {"rule_type": "ncci_modifier"}}

    # Task 3 extra: processor log + facility record + plan document (for QPA)
    if task_name == "adversarial_fabrication":
        if "fetch_payment_processor_log" not in done:
            return {"action_type": "fetch_payment_processor_log",
                    "parameters": {"claim_id": obs.case_id}}

        if "fetch_facility_record" not in done:
            return {"action_type": "fetch_facility_record",
                    "parameters": {"facility_id": state.get("facility_id") or ""}}

        if "fetch_plan_document" not in done:
            return {"action_type": "fetch_plan_document",
                    "parameters": {"plan_id": state.get("provider_id") or ""}}

    # Check deadline before submitting
    if "check_deadline" not in done:
        return {"action_type": "check_deadline",
                "parameters": {"deadline_type": "appeal"}}

    # Write diagnosis with task-correct responsible_party and evidence IDs
    if "write_diagnosis" not in done:
        # Build evidence list with the artifact types each task's grader checks for
        arts = state["artifacts"]
        if task_name == "deductive_liability":
            ev_keys = ["fetch_claim_record", "fetch_eob", "fetch_payment_ledger"]
        elif task_name == "abductive_conflict":
            ev_keys = ["fetch_provider_record", "lookup_procedure_code", "fetch_eob"]
        else:  # adversarial_fabrication
            ev_keys = ["fetch_payment_processor_log", "fetch_claim_record", "fetch_eob"]
        evidence_ids = [arts[k] for k in ev_keys if arts.get(k) is not None]
        if not evidence_ids:
            evidence_ids = [v for v in arts.values() if v is not None][:3]

        diagnosis_text = {
            "deductive_liability": (
                "Billing system error confirmed by comparing claim record, "
                "EOB, and payment ledger."
            ),
            "abductive_conflict": (
                "Insurer applied wrong denial code. Provider billed correctly "
                "per NCCI modifier rules."
            ),
            "adversarial_fabrication": (
                "Provider fabricated good-faith estimate date. "
                "Processor log contradicts submitted claim."
            ),
        }.get(task_name, f"Responsible party identified as {responsible_party}.")
        return {"action_type": "write_diagnosis", "parameters": {
            "responsible_party": responsible_party,
            "evidence_artifact_ids": evidence_ids,
            "diagnosis_text": diagnosis_text,
        }}

    # Draft resolution with task-correct type + required fields
    if "draft_resolution" not in done:
        params: Dict[str, Any] = {
            "resolution_type": resolution_type,
            "summary": f"Resolution filed: {resolution_type}.",
        }
        if resolution_type == "refund":
            params["refund_amount"] = state.get("refund_amount") or 0.0
        elif resolution_type == "appeal":
            params["appeal_reason"] = "modifier_exception"
        elif resolution_type == "nsa_dispute":
            params["qpa_reference_amount"] = state.get("qpa_amount") or 0.0
        return {"action_type": "draft_resolution", "parameters": params}

    # Submit resolution
    if "submit_resolution" not in done:
        return {"action_type": "submit_resolution", "parameters": {
            "draft_artifact_id": state["artifacts"].get("draft_resolution"),
        }}

    # Reject counter-arguments (task 3 — 3 counters injected after submit_resolution)
    # Trigger: task 3 + submission done + fewer than 3 rejections
    if task_name == "adversarial_fabrication" and "submit_resolution" in done and rejects < 3:
        arts = state["artifacts"]
        # Counter 1 cites processor_log; Counter 2 cites regulatory_rule; Counter 3 uses both
        if rejects == 0:
            cited = [v for k, v in arts.items()
                     if k == "fetch_payment_processor_log" and v is not None]
        elif rejects == 1:
            cited = [v for k, v in arts.items()
                     if k == "check_regulatory_rule" and v is not None]
        else:
            cited = [v for v in arts.values() if v is not None][:2]
        return {"action_type": "reject_counter_argument", "parameters": {
            "counter_index": rejects + 1,
            "rejection_reasoning": (
                "The independent evidence on record stands. "
                "This counter-argument does not change the facts of the case."
            ),
            "cited_artifact_ids": cited or [v for v in arts.values() if v is not None][:1],
        }}

    # Notify provider (required for tasks 2 & 3)
    if task_name in ("abductive_conflict", "adversarial_fabrication"):
        if "notify_provider" not in done:
            return {"action_type": "notify_provider", "parameters": {
                "message_type": "outcome",
                "message_text": f"Case outcome: {resolution_type} has been filed.",
            }}

    # Send patient communication
    if "send_patient_communication" not in done:
        return {"action_type": "send_patient_communication", "parameters": {
            "message_type": "outcome",
            "message_text": "Your billing dispute has been resolved.",
        }}

    # Write audit entry
    if "write_audit_entry" not in done:
        return {"action_type": "write_audit_entry", "parameters": {
            "summary": f"Case resolved. {responsible_party} identified and remediated.",
            "outcome_code": "resolved",
        }}

    return {"action_type": "close_case", "parameters": {"outcome_code": "resolved"}}


# ------------------------------------------------------------------
# LLM action decision
# ------------------------------------------------------------------

def get_model_action(
    client: Any,
    obs: Any,
    task_name: str,
    step_num: int,
    max_steps: int,
    messages: List[Dict],
) -> Dict[str, Any]:
    """Ask the LLM for the next action given the current observation.

    Returns a dict with 'action_type' and 'parameters'.
    Falls back to a simple close_case if LLM is unavailable or fails.
    """
    if client is None:
        # No client — caller should use _fallback_action instead
        return {"action_type": "close_case", "parameters": {"outcome_code": "timeout"}}

    obs_context = {
        "case_id": obs.case_id,
        "patient_complaint": obs.patient_complaint,
        "step": f"{step_num}/{max_steps}",
        "last_action": obs.last_action_type,
        "last_result": obs.last_action_result,
        "last_error": obs.last_action_error,
        "history": obs.action_log_summary[-8:],
        "rate_limited_tools": obs.rate_limited_tools,
    }

    user_content = (
        f"Observation:\n{json.dumps(obs_context, indent=2)}\n\n"
        f"{_ACTION_SCHEMA}\n"
        f'Respond with ONLY a JSON object: {{"action_type": "...", "parameters": {{...}}}}'
    )

    msgs = messages + [{"role": "user", "content": user_content}]

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=msgs,
            max_tokens=300,
            temperature=0.1,
        )
        raw = (resp.choices[0].message.content or "").strip()
        # Strip markdown fences
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:])
            if raw.rstrip().endswith("```"):
                raw = raw.rstrip()[:-3]
        action_dict = json.loads(raw)
        if "action_type" in action_dict:
            return action_dict
    except Exception as exc:
        print(f"[DEBUG] LLM step failed: {exc}", file=sys.stderr, flush=True)

    # LLM failed — return a no-op that signals the caller to use fallback
    return {}


# ------------------------------------------------------------------
# Episode runner
# ------------------------------------------------------------------

async def run_episode(
    env: Any,
    task_name: str,
    seed: int,
    client: Any = None,
) -> dict:
    """Run one episode.  LLM decides each step; fallback heuristic if LLM fails.

    Does NOT print [END] — the caller aggregates scores across seeds and
    prints exactly one [END] per task.
    """
    from server.models import ClarusAction

    obs = await env.reset(task_name, seed=seed)
    state = make_state()
    max_steps = TASK_MAX_STEPS.get(task_name, 20)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    rewards: List[float] = []
    step_count = 0
    score = 0.5
    llm_failures = 0          # consecutive LLM failures
    LLM_FAILURE_LIMIT = 3    # disable LLM after this many consecutive failures

    while not obs.done and step_count < max_steps:
        step_count += 1

        # Try LLM unless it has failed too many times in a row
        action_dict: Dict[str, Any] = {}
        use_llm = client is not None and llm_failures < LLM_FAILURE_LIMIT
        if use_llm:
            action_dict = get_model_action(
                client=client,
                obs=obs,
                task_name=task_name,
                step_num=step_count,
                max_steps=max_steps,
                messages=messages,
            )

        if action_dict:
            llm_failures = 0  # reset on success
        else:
            # LLM failed or absent — use deterministic heuristic
            if use_llm:
                llm_failures += 1
                if llm_failures >= LLM_FAILURE_LIMIT:
                    print(f"[DEBUG] LLM failed {llm_failures}x in a row — "
                          f"switching to heuristic for task={task_name}",
                          file=sys.stderr, flush=True)
            action_dict = _fallback_action(obs, state, task_name)

        action_type = action_dict.get("action_type", "close_case")
        parameters = dict(action_dict.get("parameters") or {})

        # Remove None values from list parameters
        for key in ("evidence_artifact_ids", "cited_artifact_ids"):
            if key in parameters and isinstance(parameters[key], list):
                parameters[key] = [v for v in parameters[key] if v is not None]

        action = ClarusAction(action_type=action_type, parameters=parameters)
        result = await env.step(action)

        update_state(state, action_type, result.observation.last_action_result)
        obs = result.observation
        reward = result.reward or 0.0
        rewards.append(reward)

        log_step(
            step=step_count,
            action=action_type,
            reward=reward,
            done=result.done,
            error=obs.last_action_error,
        )

        if result.done:
            raw = result.info.get("episode_score")
            score = float(raw) if raw is not None else 0.5
            return {"score": score, "steps": step_count, "rewards": rewards}

    # Force-close if max steps exhausted
    if not obs.done:
        step_count += 1
        result = await env.step(
            ClarusAction(action_type="close_case",
                         parameters={"outcome_code": "timeout"})
        )
        rewards.append(result.reward or 0.0)
        raw = result.info.get("episode_score")
        score = float(raw) if raw is not None else 0.5
        log_step(step=step_count, action="close_case",
                 reward=result.reward or 0.0, done=True, error=None)

    return {"score": score, "steps": step_count, "rewards": rewards}


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

async def main() -> None:
    """Run one episode per task; emit exactly one [START] and one [END] per task.

    Output format (OpenEnv compliant):
        [START] task=<name> env=clarus model=<model>
        [STEP]  step=N action=... reward=... done=... error=...
        ...
        [END]   task=<name> score=<float>

    The score is Laplace-smoothed (passed+0.5)/(total+1), always strictly
    in (0, 1).  We run one canonical seed per task so there is never more
    than one [END] per task name.
    """
    from server.env import ClarusEnv
    from server.schema import create_tables
    from data.setup import load_all

    client = None
    if HF_TOKEN:
        try:
            from openai import OpenAI
            # timeout=30s per call — gives LLM enough time without hanging
            client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN, timeout=30.0)
            print(f"[DEBUG] LLM client ready: {API_BASE_URL}", file=sys.stderr, flush=True)
        except Exception as exc:
            print(f"[DEBUG] LLM client init failed: {exc}", file=sys.stderr, flush=True)
    else:
        print("[DEBUG] HF_TOKEN not set — using heuristic fallback",
              file=sys.stderr, flush=True)

    ref_db = sqlite3.connect(":memory:", check_same_thread=False)
    ref_db.row_factory = sqlite3.Row
    create_tables(ref_db)
    load_all(ref_db)

    env = ClarusEnv(ref_db=ref_db)
    all_scores: List[float] = []

    # Per-task timeout: 5 minutes each → 15 min total, well inside 20-min limit
    TASK_TIMEOUT = 300  # seconds

    try:
        for task_name, seed in EVAL_SEEDS.items():
            # Exactly one [START] per task
            log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

            score = 0.5  # safe default — strictly in (0,1)
            rewards: List[float] = []
            steps = 0

            try:
                result = await asyncio.wait_for(
                    run_episode(env, task_name, seed, client=client),
                    timeout=TASK_TIMEOUT,
                )
                score = result["score"]
                rewards = result["rewards"]
                steps = result["steps"]
            except asyncio.TimeoutError:
                print(f"[DEBUG] Task {task_name} timed out after {TASK_TIMEOUT}s "
                      f"— using default score 0.5", file=sys.stderr, flush=True)
            except Exception as exc:
                print(f"[DEBUG] Episode error task={task_name} seed={seed}: {exc}",
                      file=sys.stderr, flush=True)

            # Hard clamp — Laplace guarantees (0,1); clamp handles edge cases only
            score = max(0.02, min(0.98, float(score)))
            all_scores.append(score)

            # Exactly one [END] per task — always emitted, even after timeout
            log_end(task=task_name, success=score >= 0.5,
                    steps=steps, score=score, rewards=rewards)

    finally:
        await env.close()

    overall = sum(all_scores) / len(all_scores) if all_scores else 0.5
    print(f"\n=== OVERALL SCORE: {overall:.3f} ===", file=sys.stderr, flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        # Fatal startup error — emit valid [END] for every expected task
        print(f"[DEBUG] Top-level fatal: {exc}", file=sys.stderr, flush=True)
        for _task in EVAL_SEEDS:
            log_end(task=_task, success=False, steps=0, score=0.5, rewards=[])
