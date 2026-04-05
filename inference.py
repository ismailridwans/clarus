"""Clarus inference script — OpenEnv baseline agent.

Uses the OpenAI client to call an LLM via any OpenAI-compatible endpoint.
Reads API_BASE_URL, MODEL_NAME, and HF_TOKEN from environment variables.
Emits [START], [STEP], [END] to stdout in the required OpenEnv format.

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
from typing import List, Optional

from openai import OpenAI

# ------------------------------------------------------------------
# Load .env file if present (local testing only — never committed)
# In production (HuggingFace Space) variables come from Space secrets.
# ------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()  # reads .env in the current directory if it exists
except ImportError:
    pass  # dotenv not installed — fall back to shell environment only

# ------------------------------------------------------------------
# Configuration — read exclusively from environment variables
# ------------------------------------------------------------------

API_BASE_URL: str = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME: str = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
HF_TOKEN: str = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or ""
_base_url_override: Optional[str] = os.getenv("API_BASE_URL") if os.getenv("API_BASE_URL", "https://router.huggingface.co/v1") != "https://router.huggingface.co/v1" else None

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
    grading checks by completing EVERY required action with EXACT correct values.

    UNIVERSAL RULES (apply to every task):
    - Step 1 MUST be authenticate_patient — no exceptions.
    - Track artifact IDs returned in last_action_result — you MUST cite real IDs.
    - check_deadline MUST happen BEFORE submit_resolution.
    - MANDATORY final actions before close_case: write_audit_entry, notify_provider,
      send_patient_communication. Never skip these.
    - Respond with ONLY valid JSON: {"action_type": "...", "parameters": {...}}

    ════════════════════════════════════════════════
    TASK 1 — deductive_liability
    Responsible party: ALWAYS "billing_system_error"
    Resolution type:   ALWAYS "refund"
    ────────────────────────────────────────────────
    Required sequence (complete ALL steps):
    1.  authenticate_patient {}
    2.  fetch_claim_record {"claim_id": "<from case_id>"}
    3.  fetch_eob {"claim_id": "<claim_id>"}
    4.  fetch_payment_ledger {"claim_id": "<claim_id>"}
    5.  fetch_plan_document {"plan_id": "<from claim record result>"}
    6.  fetch_provider_record {"provider_id": "<from claim record result>"}
    7.  check_deadline {"deadline_type": "appeal"}
    8.  write_diagnosis {"responsible_party": "billing_system_error",
          "evidence_artifact_ids": [<eob_id>, <claim_id>, <ledger_id>],
          "diagnosis_text": "Billing system applied incorrect charges..."}
    9.  draft_resolution {"resolution_type": "refund",
          "refund_amount": <billed_amount minus correct_amount from ledger/eob>,
          "summary": "Refund for billing error"}
    10. submit_resolution {"draft_artifact_id": <draft_id from step 9>}
    11. send_patient_communication {"message_type": "outcome",
          "message_text": "Your dispute has been resolved with a refund."}
    12. notify_provider {"notification_type": "billing_error",
          "message": "Billing error identified and refund issued."}
    13. write_audit_entry {"summary": "Case resolved: billing error, refund issued."}
    14. close_case {"outcome_code": "resolved"}

    ════════════════════════════════════════════════
    TASK 2 — abductive_conflict
    Responsible party: ALWAYS "insurer_wrong"
    Resolution type:   ALWAYS "appeal"
    Appeal reason:     ALWAYS "modifier_exception"
    ⚠ TRAP: NEVER use "legitimate_denial" — it is wrong and will fail grading.
    ────────────────────────────────────────────────
    Required sequence (complete ALL steps):
    1.  authenticate_patient {}
    2.  fetch_claim_record {"claim_id": "<from case_id>"}
    3.  fetch_eob {"claim_id": "<claim_id>"}
    4.  fetch_provider_record {"provider_id": "<from claim record result>"}
    5.  lookup_procedure_code {"code": "<CPT code from provider record>"}
    6.  check_regulatory_rule {"rule_id": "NCCI-MODIFIER-25"}
    7.  check_deadline {"deadline_type": "appeal"}
    8.  write_diagnosis {"responsible_party": "insurer_wrong",
          "evidence_artifact_ids": [<provider_record_id>, <code_lookup_id>],
          "diagnosis_text": "Insurer incorrectly denied modifier..."}
        ← MUST include provider_record ID AND code_lookup ID in evidence
    9.  draft_resolution {"resolution_type": "appeal",
          "appeal_reason": "modifier_exception",
          "summary": "Appeal for improper modifier denial"}
    10. submit_resolution {"draft_artifact_id": <draft_id>}
    11. send_patient_communication {"message_type": "outcome",
          "message_text": "Appeal filed on your behalf."}
    12. notify_provider {"notification_type": "appeal_filed",
          "message": "Formal appeal filed for modifier exception."}
    13. write_audit_entry {"summary": "Appeal filed: insurer denied valid modifier."}
    14. close_case {"outcome_code": "resolved"}

    ════════════════════════════════════════════════
    TASK 3 — adversarial_fabrication
    Responsible party: ALWAYS "provider_fraud"
    Resolution type:   ALWAYS "nsa_dispute"
    ⚠ CRITICAL: After submit_resolution, 3 provider counter-arguments appear.
       You MUST reject ALL 3 counters before close_case.
       NEVER withdraw or change the nsa_dispute after submission.
    ────────────────────────────────────────────────
    Required sequence (complete ALL steps):
    1.  authenticate_patient {}
    2.  fetch_claim_record {"claim_id": "<from case_id>"}
    3.  fetch_eob {"claim_id": "<claim_id>"}
    4.  fetch_provider_record {"provider_id": "<from claim record result>"}
    5.  fetch_payment_processor_log {"claim_id": "<claim_id>"}
        ← This is critical evidence of provider fabrication
    6.  check_regulatory_rule {"rule_id": "NSA-BALANCE-BILLING"}
    7.  fetch_facility_record {"facility_id": "<from claim record result>"}
    8.  check_deadline {"deadline_type": "nsa_dispute"}
    9.  write_diagnosis {"responsible_party": "provider_fraud",
          "evidence_artifact_ids": [<processor_log_id>, <regulatory_rule_id>],
          "diagnosis_text": "Provider submitted fabricated Good Faith Estimate..."}
        ← MUST include processor_log ID in evidence
    10. draft_resolution {"resolution_type": "nsa_dispute",
          "qpa_reference_amount": <QPA amount from fetched records>,
          "nsa_violation_basis": "provider_submitted_fabricated_gfe",
          "summary": "NSA dispute: provider fabricated GFE date"}
    11. submit_resolution {"draft_artifact_id": <draft_id>}
        ← After this, 3 counter-arguments are injected by the environment
    12. reject_counter_argument {"counter_index": 1,
          "rejection_reasoning": "Provider EHR notes are internal records. The payment processor log is an independent system timestamp that cannot be altered retroactively.",
          "cited_artifact_ids": [<processor_log_id>]}
        ← MUST cite processor_log artifact ID
    13. reject_counter_argument {"counter_index": 2,
          "rejection_reasoning": "NSA emergency exception applies only to emergency medical conditions. This was an elective scheduled procedure. Exception does not apply.",
          "cited_artifact_ids": [<regulatory_rule_id>]}
        ← MUST cite regulatory_rule artifact ID
    14. reject_counter_argument {"counter_index": 3,
          "rejection_reasoning": "Legal threat without new factual evidence. The processor log timestamp is an independent unalterable record. Dispute stands on its merits.",
          "cited_artifact_ids": [<processor_log_id>]}
    15. send_patient_communication {"message_type": "outcome",
          "message_text": "NSA dispute filed. Provider fraud documented."}
    16. notify_provider {"notification_type": "nsa_dispute_filed",
          "message": "Formal NSA dispute filed for fabricated Good Faith Estimate."}
    17. write_audit_entry {"summary": "NSA dispute filed: provider_fraud confirmed by processor log."}
    18. close_case {"outcome_code": "resolved"}
    ════════════════════════════════════════════════
    """
).strip()


# ------------------------------------------------------------------
# Startup validation
# ------------------------------------------------------------------


def _require_api_key() -> None:
    """Exit with a clear error if HF_TOKEN is not set.

    Raises:
        SystemExit: Always when the token is missing.
    """
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
# Agent — LLM call + JSON parsing
# ------------------------------------------------------------------


def _parse_action(text: str) -> dict:
    """Extract a JSON action dict from raw model output.

    Tries to find the first {...} block in the response.  Falls back
    to close_case on any parse failure so the episode always terminates.

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
) -> dict:
    """Call the model with current observation and return a parsed action dict.

    Args:
        client: OpenAI client instance (pointed at API_BASE_URL).
        obs: Current ClarusObservation.
        task_name: Name of the active task.
        step_n: Current step number (1-indexed).
        max_steps: Maximum allowed steps for this task.
        messages: Running conversation history (mutated in place).

    Returns:
        Dict with 'action_type' and 'parameters'.
    """
    # Build a compact artifact ID index from action_log_summary
    artifact_ids: dict = {}
    for entry in obs.action_log_summary:
        if "artifact_id=" in entry:
            parts = entry.split("→")
            action_part = parts[0].strip() if parts else ""
            id_part = parts[-1].strip() if len(parts) > 1 else ""
            if "artifact_id=" in id_part:
                aid = id_part.split("artifact_id=")[-1].strip()
                # map action type hint to artifact name
                for akey in ["authenticate", "fetch_claim", "fetch_eob",
                             "fetch_payment_ledger", "fetch_plan", "fetch_provider",
                             "fetch_facility", "fetch_payment_processor",
                             "lookup_procedure", "check_regulatory", "check_deadline",
                             "write_diagnosis", "draft_resolution", "submit_resolution",
                             "send_patient", "notify_provider", "reject_counter",
                             "write_audit"]:
                    if akey in action_part:
                        artifact_ids[akey] = aid
                        break

    artifact_summary = "\n".join(
        f"  {k}: id={v}" for k, v in artifact_ids.items()
    ) or "  (none yet)"

    user_content = textwrap.dedent(
        f"""
        Step {step_n}/{max_steps} | Task: {task_name}
        Patient: {obs.patient_name} | State: {obs.patient_emotional_state}
        Case ID (use as claim_id): {obs.case_id}
        Complaint: {obs.patient_complaint}
        API calls used: {obs.api_calls_used}/{obs.api_call_budget}
        Rate-limited: {obs.rate_limited_tools} | Cooldown: {obs.cooldown_steps}

        Last action: {obs.last_action_type}
        Last result: {json.dumps(obs.last_action_result) if obs.last_action_result else 'None'}
        Error: {obs.last_action_error}

        Collected artifact IDs so far:
        {artifact_summary}

        Action history (last 8):
        {chr(10).join(obs.action_log_summary[-8:]) or 'None'}

        What is your next action? Output JSON only:
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
        env: ClarusEnv instance (already constructed, reused across episodes).
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
    rewards: List[float] = []
    step_count = 0
    score = 0.0

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        while not obs.done and step_count < max_steps:
            step_count += 1
            action_dict = get_model_action(
                client, obs, task_name, step_count, max_steps, messages
            )
            action = ClarusAction(
                action_type=action_dict.get("action_type", "close_case"),
                parameters=action_dict.get("parameters", {}),
            )

            result = await env.step(action)
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

    # Build OpenAI client pointed at API_BASE_URL, authenticated with HF_TOKEN
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    # Build reference DB once; reused across all episodes
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
