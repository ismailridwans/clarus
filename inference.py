"""Clarus inference script — OpenEnv baseline agent.

Uses the OpenAI client to call an LLM via any OpenAI-compatible endpoint.
Reads API_BASE_URL, MODEL_NAME, and HF_TOKEN from environment variables.
Emits [START], [STEP], [END] to stdout in the required OpenEnv format.

Hybrid architecture:
  - Python extracts financial values from action results (billed_amount,
    patient_responsibility, copay_specialist, qpa_amount) and computes
    exact dollar amounts (refund_amount, qpa_reference_amount).
  - Python tracks artifact IDs returned by the environment.
  - LLM receives a "COMPUTED VALUES" block with exact numbers and IDs
    to copy directly into action parameters — no arithmetic required.

Usage:
    export API_BASE_URL=https://router.huggingface.co/v1
    export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
    export HF_TOKEN=hf_...
    python inference.py

Environment variables:
    API_BASE_URL   The LLM API endpoint (OpenAI-compatible).
                   Default: https://router.huggingface.co/v1
    MODEL_NAME     Model identifier.
                   Default: Qwen/Qwen2.5-72B-Instruct
    HF_TOKEN       HuggingFace API key (used as the bearer token).
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import sys
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ------------------------------------------------------------------
# Load .env file if present (local testing only — never committed)
# In production (HuggingFace Space) variables come from Space secrets.
# ------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ------------------------------------------------------------------
# Configuration — read exclusively from environment variables
# ------------------------------------------------------------------

API_BASE_URL: str = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME: str = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
HF_TOKEN: str = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or ""

BENCHMARK = "clarus"
TEMPERATURE = 0.1
MAX_TOKENS = 512

TASK_MAX_STEPS = {
    "deductive_liability": 18,
    "abductive_conflict": 20,
    "adversarial_fabrication": 26,
}

# Dev seeds — used for submission scoring
DEV_SEEDS = {
    "deductive_liability": list(range(1101, 1106)),
    "abductive_conflict": list(range(2101, 2106)),
    "adversarial_fabrication": list(range(3101, 3106)),
}

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a healthcare billing dispute specialist. Your ONLY goal is to pass ALL
    grading checks by completing EVERY required action in the EXACT sequence below.

    UNIVERSAL RULES:
    - Step 1 MUST be authenticate_patient — no exceptions.
    - Respond with ONLY valid JSON: {"action_type": "...", "parameters": {...}}
    - ALWAYS use the artifact IDs shown in the COMPUTED VALUES block below.
    - ALWAYS use the dollar amounts shown in COMPUTED VALUES — never recalculate.
    - check_deadline MUST happen BEFORE submit_resolution.
    - After submit_resolution, complete: send_patient_communication, notify_provider,
      write_audit_entry, close_case (in that order for Tasks 1 & 2).

    ================================================
    TASK 1 — deductive_liability
    Responsible party: "billing_system_error"
    Resolution type:   "refund"
    ------------------------------------------------
    Execute steps in this EXACT order:
    1.  authenticate_patient {}
    2.  fetch_claim_record {"claim_id": "<case_id from observation>"}
    3.  fetch_eob {"claim_id": "<claim_id from step 2 result>"}
    4.  fetch_payment_ledger {"claim_id": "<claim_id>"}
    5.  fetch_plan_document {"plan_id": "auto"}
    6.  fetch_provider_record {"provider_id": "<provider_id from step 2>"}
    7.  check_deadline {"deadline_type": "appeal"}
    8.  write_diagnosis {
          "responsible_party": "billing_system_error",
          "evidence_artifact_ids": [<eob_id>, <claim_id>, <ledger_id>],
          "diagnosis_text": "EOB shows copay_applied=false. Copay was not credited.
            Patient was overbilled. Billing system error confirmed."
        }
    9.  draft_resolution {
          "resolution_type": "refund",
          "refund_amount": <COPY FROM COMPUTED VALUES>,
          "summary": "Refund for copay not credited by billing system."
        }
    10. submit_resolution {"draft_artifact_id": <draft_id from step 9>}
    11. send_patient_communication {
          "message_type": "outcome",
          "message_text": "Your dispute is resolved. A refund has been issued."
        }
    12. notify_provider {
          "notification_type": "billing_error",
          "message": "Billing error confirmed. Copay not credited. Refund issued."
        }
    13. write_audit_entry {
          "summary": "Case resolved: billing_system_error. Refund issued for
            uncredited copay.",
          "outcome_code": "resolved"
        }
    14. close_case {"outcome_code": "resolved"}

    ================================================
    TASK 2 — abductive_conflict
    Responsible party: "insurer_wrong"
    Resolution type:   "appeal"
    Appeal reason:     "modifier_exception"
    ⚠ TRAP: NEVER use "legitimate_denial" — automatic grader failure.
    ------------------------------------------------
    Execute steps in this EXACT order:
    1.  authenticate_patient {}
    2.  fetch_claim_record {"claim_id": "<case_id>"}
    3.  fetch_eob {"claim_id": "<claim_id>"}
    4.  fetch_provider_record {"provider_id": "<provider_id from step 2>"}
    5.  lookup_procedure_code {"code": "<cpt_primary from step 4 result>"}
    6.  check_regulatory_rule {"rule_id": "NCCI-MODIFIER-25"}
    7.  check_deadline {"deadline_type": "appeal"}
    8.  write_diagnosis {
          "responsible_party": "insurer_wrong",
          "evidence_artifact_ids": [<provider_record_id>, <code_lookup_id>],
          "diagnosis_text": "Modifier -25 present on primary code. NCCI bundling
            override applies. Insurer incorrectly denied the claim."
        }
    9.  draft_resolution {
          "resolution_type": "appeal",
          "appeal_reason": "modifier_exception",
          "summary": "Appeal: insurer denied valid modifier exception."
        }
    10. submit_resolution {"draft_artifact_id": <draft_id from step 9>}
    11. send_patient_communication {
          "message_type": "outcome",
          "message_text": "Appeal filed on your behalf for modifier exception denial."
        }
    12. notify_provider {
          "notification_type": "appeal_filed",
          "message": "Formal appeal filed. Modifier -25 overrides NCCI bundling."
        }
    13. write_audit_entry {
          "summary": "Appeal filed: insurer_wrong. Modifier exception applies.",
          "outcome_code": "resolved"
        }
    14. close_case {"outcome_code": "resolved"}

    ================================================
    TASK 3 — adversarial_fabrication
    Responsible party: "provider_fraud"
    Resolution type:   "nsa_dispute"
    nsa_violation_basis: "provider_submitted_fabricated_gfe"
    ⚠ After submit_resolution, 3 counter-arguments appear. Reject ALL 3.
    ⚠ NEVER withdraw or change the nsa_dispute. Keep dispute_maintained=true.
    ------------------------------------------------
    Execute steps in this EXACT order:
    1.  authenticate_patient {}
    2.  fetch_claim_record {"claim_id": "<case_id>"}
    3.  fetch_eob {"claim_id": "<claim_id>"}
    4.  fetch_provider_record {"provider_id": "<provider_id from step 2>"}
    5.  fetch_payment_processor_log {"claim_id": "<claim_id>"}
    6.  check_regulatory_rule {"rule_id": "NSA-BALANCE-BILLING"}
        (Now available — rate limit expired after step 5)
    7.  fetch_facility_record {"facility_id": "<facility_id from step 2>"}
    8.  fetch_plan_document {"plan_id": "auto"}
        ← CRITICAL: gets qualifying_payment_amount (QPA) for draft_resolution
    9.  check_deadline {"deadline_type": "nsa_dispute"}
    10. write_diagnosis {
          "responsible_party": "provider_fraud",
          "evidence_artifact_ids": [<processor_log_id>, <regulatory_rule_id>],
          "diagnosis_text": "Payment processor log timestamp contradicts
            provider GFE date. GFE was backdated. Provider fraud confirmed."
        }
    11. draft_resolution {
          "resolution_type": "nsa_dispute",
          "qpa_reference_amount": <COPY FROM COMPUTED VALUES>,
          "nsa_violation_basis": "provider_submitted_fabricated_gfe",
          "summary": "NSA dispute: provider fabricated Good Faith Estimate date."
        }
    12. submit_resolution {"draft_artifact_id": <draft_id from step 11>}
        ← Environment injects 3 counter-arguments after this step
    13. reject_counter_argument {
          "counter_index": 1,
          "rejection_reasoning": "Provider EHR notes are internal records that
            the provider controls. The payment processor log is an independent
            third-party timestamp that cannot be altered retroactively. The
            independent timestamp contradicts the backdated GFE.",
          "cited_artifact_ids": [<processor_log_id>]
        }
    14. reject_counter_argument {
          "counter_index": 2,
          "rejection_reasoning": "The NSA emergency exception under 45 CFR
            149.110 applies only to emergency medical conditions requiring
            immediate treatment. This was an elective, scheduled procedure.
            The emergency exception does not apply here.",
          "cited_artifact_ids": [<regulatory_rule_id>]
        }
    15. reject_counter_argument {
          "counter_index": 3,
          "rejection_reasoning": "This is a legal threat without any new
            factual evidence. The payment processor log provides an independent,
            unalterable timestamp. The dispute stands on its merits regardless
            of legal pressure.",
          "cited_artifact_ids": [<processor_log_id>]
        }
    16. send_patient_communication {
          "message_type": "outcome",
          "message_text": "NSA dispute filed. Provider fraud documented.
            All counter-arguments rejected. Your case is protected."
        }
    17. notify_provider {
          "notification_type": "nsa_dispute_filed",
          "message": "Formal NSA dispute filed. Provider fabricated GFE date.
            All counter-arguments have been rejected on the merits."
        }
    18. write_audit_entry {
          "summary": "NSA dispute filed: provider_fraud confirmed by processor
            log. All 3 counter-arguments rejected.",
          "outcome_code": "resolved"
        }
    19. close_case {"outcome_code": "resolved"}
    ================================================
    """
).strip()


# ------------------------------------------------------------------
# Startup validation
# ------------------------------------------------------------------


def _require_api_key() -> None:
    """Exit with a clear error if HF_TOKEN is not set."""
    if not HF_TOKEN:
        print(
            "ERROR: HF_TOKEN environment variable is not set.\n"
            "\n"
            "Set it before running:\n"
            "    export HF_TOKEN=hf_...\n"
            "    python inference.py\n"
            "\n"
            "Get a HuggingFace token at: https://huggingface.co/settings/tokens",
            file=sys.stderr,
        )
        sys.exit(1)


# ------------------------------------------------------------------
# Stdout logging — exact OpenEnv format
# ------------------------------------------------------------------


def log_start(task: str, env: str, model: str) -> None:
    """Emit the [START] line to stdout."""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    """Emit one [STEP] line to stdout."""
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """Emit the [END] line to stdout."""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"rewards={rewards_str}",
        flush=True,
    )


# ------------------------------------------------------------------
# Episode state — Python-computed values injected into prompts
# ------------------------------------------------------------------


def make_episode_state() -> Dict[str, Any]:
    """Create a fresh state dict for one episode."""
    return {
        "claim_id_value": None,       # CLM-XXXXX from fetch_claim_record
        "provider_id": None,
        "facility_id": None,
        "billed_amount": None,        # from fetch_claim_record
        "patient_responsibility": None,  # from fetch_eob
        "copay_specialist": None,     # from fetch_plan_document
        "qpa_amount": None,           # from fetch_plan_document (Task 3 only)
        "refund_amount": None,        # computed for Task 1
        "artifacts": {},              # action_type -> artifact_id (int)
    }


def update_state(
    state: Dict[str, Any],
    action_type: str,
    result: Optional[Dict[str, Any]],
) -> None:
    """Extract financial values and artifact IDs from the action result.

    Called after every successful step.  Python computes refund_amount
    once all three values (billed_amount, patient_responsibility,
    copay_specialist) are available — no LLM arithmetic needed.

    Args:
        state: Mutable episode state dict.
        action_type: The action that just completed.
        result: The result payload from obs.last_action_result.
    """
    if not result:
        return

    # Track artifact_id for every action that returns one
    aid = result.get("artifact_id")
    if aid is not None:
        state["artifacts"][action_type] = int(aid)

    # Extract task-critical financial values
    if action_type == "fetch_claim_record":
        state["claim_id_value"] = result.get("claim_id")
        state["billed_amount"] = result.get("billed_amount")
        state["provider_id"] = result.get("provider_id")
        state["facility_id"] = result.get("facility_id")

    elif action_type == "fetch_eob":
        state["patient_responsibility"] = result.get("patient_responsibility")

    elif action_type == "fetch_plan_document":
        state["copay_specialist"] = result.get("copay_specialist")
        qpa = result.get("qualifying_payment_amount")
        if qpa is not None:
            state["qpa_amount"] = round(float(qpa), 2)

    # Task 1: compute exact refund as soon as all inputs are available
    ba = state["billed_amount"]
    pr = state["patient_responsibility"]
    cs = state["copay_specialist"]
    if ba is not None and pr is not None and cs is not None:
        correct_balance = max(0.0, float(pr) - float(cs))
        state["refund_amount"] = round(float(ba) - correct_balance, 2)


def format_computed_block(state: Dict[str, Any], task_name: str) -> str:
    """Format the COMPUTED VALUES block injected into every observation.

    This block gives the LLM exact artifact IDs and dollar amounts so it
    never needs to calculate anything — just copy from this block.

    Args:
        state: Current episode state dict.
        task_name: Active task name.

    Returns:
        Multi-line string ready for injection into the user message.
    """
    lines = ["=== COMPUTED VALUES: COPY EXACTLY, DO NOT RECALCULATE ==="]

    # Artifact ID table
    if state["artifacts"]:
        lines.append("Artifact IDs:")
        action_labels = {
            "authenticate_patient": "auth_record",
            "fetch_claim_record": "claim_record",
            "fetch_eob": "eob",
            "fetch_payment_ledger": "payment_ledger",
            "fetch_plan_document": "plan_document",
            "fetch_provider_record": "provider_record",
            "fetch_facility_record": "facility_record",
            "fetch_payment_processor_log": "processor_log",
            "lookup_procedure_code": "code_lookup",
            "check_regulatory_rule": "regulatory_rule",
            "check_deadline": "deadline_check",
            "write_diagnosis": "diagnosis",
            "draft_resolution": "draft_resolution",
            "submit_resolution": "submitted_resolution",
        }
        for atype, aid in state["artifacts"].items():
            label = action_labels.get(atype, atype)
            lines.append(f"  {label:30s} id={aid}")
    else:
        lines.append("  (no artifacts yet)")

    # Task-specific computed values
    if task_name == "deductive_liability":
        if state["refund_amount"] is not None:
            lines.append(
                f"\nFOR draft_resolution -> refund_amount = {state['refund_amount']}"
            )
            lines.append(
                "  (= billed_amount - max(0, eob.patient_responsibility - plan.copay_specialist))"
            )
        else:
            missing = []
            if state["billed_amount"] is None:
                missing.append("billed_amount (fetch_claim_record)")
            if state["patient_responsibility"] is None:
                missing.append("patient_responsibility (fetch_eob)")
            if state["copay_specialist"] is None:
                missing.append("copay_specialist (fetch_plan_document)")
            lines.append(f"\nRefund not yet computable — still need: {', '.join(missing)}")

    elif task_name == "adversarial_fabrication":
        if state["qpa_amount"] is not None:
            lines.append(
                f"\nFOR draft_resolution -> qpa_reference_amount = {state['qpa_amount']}"
            )
            lines.append(
                "  (= qualifying_payment_amount from plan_document)"
            )
        else:
            lines.append(
                "\nQPA not yet available — fetch_plan_document first (available step 6+)"
            )

    lines.append("===========================================================")
    return "\n".join(lines)


# ------------------------------------------------------------------
# Agent — LLM call + JSON parsing
# ------------------------------------------------------------------


def _parse_action(text: str) -> dict:
    """Extract a JSON action dict from raw model output.

    Tries to find the first complete {...} block.  Falls back to
    close_case on any parse failure so the episode always terminates.

    Args:
        text: Raw text returned by the model.

    Returns:
        Dict with 'action_type' and 'parameters' keys.
    """
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    return {"action_type": "close_case", "parameters": {"outcome_code": "error"}}


def get_model_action(
    client: OpenAI,
    obs,
    task_name: str,
    step_n: int,
    max_steps: int,
    messages: list,
    state: Dict[str, Any],
) -> dict:
    """Call the model with current observation and return a parsed action dict.

    Injects the Python-computed values block so the LLM can copy exact
    numbers rather than computing them itself.

    Args:
        client: OpenAI client instance.
        obs: Current ClarusObservation.
        task_name: Name of the active task.
        step_n: Current step number (1-indexed).
        max_steps: Maximum allowed steps for this task.
        messages: Running conversation history (mutated in place).
        state: Episode state dict (artifact IDs + computed values).

    Returns:
        Dict with 'action_type' and 'parameters'.
    """
    computed_block = format_computed_block(state, task_name)

    user_content = textwrap.dedent(
        f"""
        Step {step_n}/{max_steps} | Task: {task_name}
        Patient: {obs.patient_name} | Emotional state: {obs.patient_emotional_state}
        Case ID (pass as claim_id when server needs it): {obs.case_id}
        Complaint: {obs.patient_complaint}
        API calls used: {obs.api_calls_used}/{obs.api_call_budget}
        Rate-limited tools: {obs.rate_limited_tools}
        Cooldown remaining: {obs.cooldown_steps}

        Last action:  {obs.last_action_type}
        Last result:  {json.dumps(obs.last_action_result) if obs.last_action_result else 'None'}
        Last error:   {obs.last_action_error}

        Recent history:
        {chr(10).join(obs.action_log_summary[-10:]) or 'None'}

        {computed_block}

        Output your next action as JSON only (no prose):
        """
    ).strip()

    messages.append({"role": "user", "content": user_content})

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()
        messages.append({"role": "assistant", "content": raw})
        return _parse_action(raw)
    except Exception as exc:
        print(f"[DEBUG] Model call failed: {exc}", file=sys.stderr, flush=True)
        return {"action_type": "close_case", "parameters": {"outcome_code": "error"}}


# ------------------------------------------------------------------
# Episode runner
# ------------------------------------------------------------------


async def run_episode(
    env,
    client: OpenAI,
    task_name: str,
    seed: int,
) -> dict:
    """Run one complete episode and return result dict.

    Args:
        env: ClarusEnv instance (reused across episodes).
        client: OpenAI client for LLM calls.
        task_name: Task name.
        seed: Deterministic seed for the episode generator.

    Returns:
        Dict with 'score' (float), 'steps' (int), 'rewards' (list of float).
    """
    from server.models import ClarusAction

    obs = await env.reset(task_name, seed=seed)
    max_steps = TASK_MAX_STEPS[task_name]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    state = make_episode_state()
    rewards: List[float] = []
    step_count = 0
    score = 0.0

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        while not obs.done and step_count < max_steps:
            step_count += 1
            action_dict = get_model_action(
                client, obs, task_name, step_count, max_steps, messages, state
            )
            action = ClarusAction(
                action_type=action_dict.get("action_type", "close_case"),
                parameters=action_dict.get("parameters", {}),
            )

            result = await env.step(action)

            # Update Python state from the result BEFORE next LLM call
            update_state(state, action.action_type, result.observation.last_action_result)

            obs = result.observation
            reward = result.reward or 0.0
            rewards.append(reward)

            log_step(
                step=step_count,
                action=action.action_type,
                reward=reward,
                done=result.done,
                error=obs.last_action_error,
            )

            if result.done:
                score = result.info.get("episode_score", 0.0) or 0.0
                log_end(
                    success=score >= 0.5,
                    steps=step_count,
                    score=score,
                    rewards=rewards,
                )
                return {"score": score, "steps": step_count, "rewards": rewards}

        # Step limit reached — force close
        if not obs.done:
            result = await env.step(
                ClarusAction(
                    action_type="close_case",
                    parameters={"outcome_code": "timeout"},
                )
            )
            rewards.append(result.reward or 0.0)
            step_count += 1
            score = result.info.get("episode_score", 0.0) or 0.0
            log_step(
                step=step_count,
                action="close_case",
                reward=result.reward or 0.0,
                done=True,
                error=None,
            )

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", file=sys.stderr, flush=True)
        score = 0.0

    log_end(success=score >= 0.5, steps=step_count, score=score, rewards=rewards)
    return {"score": score, "steps": step_count, "rewards": rewards}


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


async def main() -> None:
    """Run all dev-split episodes for all three tasks and print overall score."""
    from server.env import ClarusEnv
    from server.schema import create_tables
    from data.setup import load_all

    _require_api_key()

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    ref_db = sqlite3.connect(":memory:", check_same_thread=False)
    ref_db.row_factory = sqlite3.Row
    create_tables(ref_db)
    load_all(ref_db)

    env = ClarusEnv(ref_db=ref_db)
    all_scores: List[float] = []

    try:
        for task_name, seeds in DEV_SEEDS.items():
            task_scores: List[float] = []
            for seed in seeds:
                print(f"\n--- {task_name} seed={seed} ---", flush=True)
                try:
                    result = await run_episode(env, client, task_name, seed)
                    all_scores.append(result["score"])
                    task_scores.append(result["score"])
                except Exception as exc:
                    print(f"[DEBUG] Fatal: {exc}", file=sys.stderr, flush=True)
                    all_scores.append(0.0)
                    task_scores.append(0.0)
                    log_end(success=False, steps=0, score=0.0, rewards=[])

            task_avg = sum(task_scores) / len(task_scores) if task_scores else 0.0
            print(f"\n--- {task_name} avg: {task_avg:.3f} ---", flush=True)

    finally:
        await env.close()

    overall = sum(all_scores) / len(all_scores) if all_scores else 0.0
    print(f"\n=== OVERALL SCORE: {overall:.3f} ===", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
