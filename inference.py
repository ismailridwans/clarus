"""Clarus inference script — OpenEnv baseline agent.

Reads OPENAI_API_KEY from the environment and runs gpt-4o against the
Clarus environment.  Emits [START], [STEP], [END] to stdout in the
required OpenEnv format.

Usage:
    export OPENAI_API_KEY=sk-...
    python inference.py

    # Override model or target specific tasks:
    MODEL_NAME=gpt-4o-mini python inference.py

Environment variables:
    OPENAI_API_KEY   Required. Your OpenAI API key (sk-...).
    MODEL_NAME       Optional. Defaults to gpt-4o.
    API_BASE_URL     Optional. Override API endpoint (e.g. Azure OpenAI).
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
# Configuration — read from environment variables only, no hardcoding
# ------------------------------------------------------------------

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4o")

# API_BASE_URL is optional; when unset the standard OpenAI endpoint is used.
# Set it to point at Azure OpenAI, a local proxy, or an OpenAI-compatible server.
_base_url_override: Optional[str] = os.getenv("API_BASE_URL")

BENCHMARK = "clarus"
TEMPERATURE = 0.1
MAX_TOKENS = 400

TASK_MAX_STEPS = {
    "deductive_liability": 12,
    "abductive_conflict": 15,
    "adversarial_fabrication": 22,
}

# Dev seeds — used for submission scoring
DEV_SEEDS = {
    "deductive_liability": list(range(1101, 1106)),
    "abductive_conflict": list(range(2101, 2106)),
    "adversarial_fabrication": list(range(3101, 3106)),
}

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a healthcare billing dispute specialist and patient advocate.
    Your goal is to investigate the patient's billing complaint, gather evidence
    from the available record sources, identify the responsible party,
    and file the correct resolution. You fight for the patient.

    Available actions:
    - authenticate_patient: Must be first action before any PHI access.
    - fetch_claim_record(claim_id): Get claim details and billed amount.
    - fetch_eob(claim_id): Get Explanation of Benefits — look for errors.
    - fetch_provider_record(provider_id): Get provider's submitted codes and modifiers.
    - fetch_payment_ledger(claim_id): Get what patient has already paid.
    - fetch_plan_document(plan_id): Get deductible, coinsurance, copay details.
    - lookup_procedure_code(code): Check CPT code and NCCI bundling rules.
    - fetch_facility_record(facility_id): Check if facility is in-network.
    - fetch_payment_processor_log(claim_id): Independent timestamp record.
    - check_regulatory_rule(rule_id): Look up NSA/NCCI regulations.
      (rule_ids: NSA-BALANCE-BILLING, NSA-GFE, NSA-IDR, NSA-EMERGENCY-EXCEPTION,
       NCCI-BUNDLING, NCCI-MODIFIER-25, ACA-APPEAL)
    - check_deadline(deadline_type): Check appeal/nsa_dispute deadline. Do BEFORE submitting.
    - write_diagnosis(responsible_party, evidence_artifact_ids, diagnosis_text):
      Document your finding. Cite at least 2 artifact IDs as evidence.
      responsible_party choices: billing_system_error, insurer_wrong, provider_fraud
    - draft_resolution(resolution_type, ...): Draft before submitting.
      resolution_type: refund, appeal, nsa_dispute
    - submit_resolution(draft_artifact_id): Submit the draft.
    - send_patient_communication(message_type, message_text):
      message_type: de_escalation, outcome, explanation
    - notify_provider(notification_type, message): Notify provider.
    - reject_counter_argument(counter_index, rejection_reasoning, cited_artifact_ids):
      Task 3 only — reject each provider counter-argument (counter_index 1, 2, 3).
    - write_audit_entry(summary): Required for compliance.
    - close_case(outcome_code): Ends the episode and triggers grading.

    Strategy:
    1. authenticate_patient FIRST (required before any PHI access).
    2. Fetch all relevant records. Store artifact IDs from last_action_result.
    3. check_deadline BEFORE submitting.
    4. write_diagnosis citing at least 2 evidence artifact IDs.
    5. draft_resolution → submit_resolution → send_patient_communication.
    6. notify_provider, write_audit_entry, close_case.

    Respond with ONLY valid JSON:
    {"action_type": "action_name", "parameters": {"key": "value"}}
    """
).strip()


# ------------------------------------------------------------------
# Startup validation
# ------------------------------------------------------------------


def _require_api_key() -> None:
    """Exit with a clear error if OPENAI_API_KEY is not set.

    Raises:
        SystemExit: Always when the key is missing.
    """
    if not OPENAI_API_KEY:
        print(
            "ERROR: OPENAI_API_KEY environment variable is not set.\n"
            "\n"
            "Set it before running:\n"
            "    export OPENAI_API_KEY=sk-...\n"
            "    python inference.py\n"
            "\n"
            "Get a key at: https://platform.openai.com/api-keys",
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
        f"score={score:.3f} rewards={rewards_str}",
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
        client: OpenAI client instance.
        obs: Current ClarusObservation.
        task_name: Name of the active task.
        step_n: Current step number (1-indexed).
        max_steps: Maximum allowed steps for this task.
        messages: Running conversation history (mutated in place).

    Returns:
        Dict with 'action_type' and 'parameters'.
    """
    user_content = textwrap.dedent(
        f"""
        Step {step_n}/{max_steps} | Task: {task_name}
        Patient: {obs.patient_name} | State: {obs.patient_emotional_state}
        Complaint: {obs.patient_complaint}
        API calls used: {obs.api_calls_used}/{obs.api_call_budget}
        Rate-limited tools: {obs.rate_limited_tools}
        Cooldown steps remaining: {obs.cooldown_steps}

        Last action: {obs.last_action_type}
        Last result:
        {json.dumps(obs.last_action_result, indent=2) if obs.last_action_result else 'None'}
        Error: {obs.last_action_error}

        Action history (last 6):
        {chr(10).join(obs.action_log_summary[-6:]) or 'None'}

        What is your next action? (JSON only):
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
        task_name: Task name (deductive_liability, abductive_conflict, adversarial_fabrication).
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

    # Build OpenAI client — use standard OpenAI endpoint unless overridden
    client_kwargs: dict = {"api_key": OPENAI_API_KEY}
    if _base_url_override:
        client_kwargs["base_url"] = _base_url_override
    client = OpenAI(**client_kwargs)

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
