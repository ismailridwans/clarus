"""Full trajectory tests — all three tasks must score 1.0.

Tests use the canonical optimal trajectory for each task.
No hardcoded artifact IDs or dollar amounts — all values are queried
from the DB or read from the generator.

Run: pytest tests/test_episodes.py -v
"""

from __future__ import annotations

import asyncio
import sqlite3
from typing import List

import pytest

from server.env import ClarusEnv
from server.models import ClarusAction
from server.scenario.generator import generate


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


def make_ref_db() -> sqlite3.Connection:
    """Build an in-memory reference DB with all bundle data loaded."""
    from server.schema import create_tables
    from data.setup import load_all

    db = sqlite3.connect(":memory:", check_same_thread=False)
    db.row_factory = sqlite3.Row
    create_tables(db)
    load_all(db)
    return db


def get_artifact_id(
    db: sqlite3.Connection, episode_id: str, artifact_type: str
) -> int:
    """Query the real artifact ID from DB — never hardcode.

    Args:
        db: Runtime SQLite connection.
        episode_id: Current episode ID.
        artifact_type: The artifact type to look up.

    Returns:
        Integer artifact ID of the most recently created artifact of this type.

    Raises:
        AssertionError: If no such artifact exists.
    """
    row = db.execute(
        "SELECT id FROM episode_artifacts "
        "WHERE episode_id=? AND artifact_type=? "
        "  AND source NOT IN ('environment') "
        "ORDER BY id DESC LIMIT 1",
        (episode_id, artifact_type),
    ).fetchone()
    assert row is not None, (
        f"Artifact '{artifact_type}' not found in episode {episode_id}"
    )
    return row[0]


# ------------------------------------------------------------------
# Task 1 — Deductive Liability
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_task1_score_1():
    """Optimal Task 1 trajectory must score 1.0."""
    ref_db = make_ref_db()
    env = ClarusEnv(ref_db=ref_db)
    await env.reset("deductive_liability", seed=1001)
    ep = env.episode_id
    p = generate("deductive_liability", 1001, ref_db)

    async def step(action_type: str, **params):
        return await env.step(ClarusAction(action_type=action_type, parameters=params))

    await step("authenticate_patient")
    await step("fetch_claim_record", claim_id=p.claim_id)
    await step("fetch_eob", claim_id=p.claim_id)
    await step("fetch_payment_ledger", claim_id=p.claim_id)
    await step("fetch_plan_document", plan_id=p.plan_id)
    await step("check_deadline", deadline_type="appeal")

    eob_id = get_artifact_id(env.db, ep, "eob")
    ledger_id = get_artifact_id(env.db, ep, "payment_ledger")
    plan_id_ = get_artifact_id(env.db, ep, "plan_document")

    await step(
        "write_diagnosis",
        responsible_party="billing_system_error",
        evidence_artifact_ids=[eob_id, ledger_id, plan_id_],
        diagnosis_text=(
            "EOB shows copay_applied=False. Payment ledger confirms "
            "copay was collected. Correct refund per plan formula."
        ),
    )
    await step(
        "draft_resolution",
        resolution_type="refund",
        refund_amount=p.correct_refund,
        summary="Copay not credited in EOB. Refund due.",
    )
    draft_id = get_artifact_id(env.db, ep, "draft_resolution")
    await step("submit_resolution", draft_artifact_id=draft_id)
    await step("send_patient_communication", message_type="outcome",
               message_text="Your refund has been filed.")
    await step("notify_provider", notification_type="refund_filed",
               message="Patient refund filed due to copay miscalculation.")
    await step("write_audit_entry",
               summary="Copay not credited. Refund filed per plan formula.")
    result = await step("close_case", outcome_code="resolved")

    assert result.done, "Episode should be done after close_case"
    score = result.info["episode_score"]
    failed = [r for r in result.info["check_results"] if not r.passed]
    assert score == 1.0, (
        f"Task 1 score={score:.4f}. "
        f"Failing checks: {[(r.description, r.actual) for r in failed]}"
    )


# ------------------------------------------------------------------
# Task 2 — Abductive Conflict
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_task2_score_1():
    """Optimal Task 2 trajectory must score 1.0."""
    ref_db = make_ref_db()
    env = ClarusEnv(ref_db=ref_db)
    await env.reset("abductive_conflict", seed=2001)
    ep = env.episode_id
    p = generate("abductive_conflict", 2001, ref_db)

    async def step(action_type: str, **params):
        return await env.step(ClarusAction(action_type=action_type, parameters=params))

    await step("authenticate_patient")
    await step("fetch_claim_record", claim_id=p.claim_id)
    await step("fetch_eob", claim_id=p.claim_id)
    await step("fetch_provider_record", provider_id=p.provider_id)
    await step("lookup_procedure_code", code=p.cpt_primary)
    await step("lookup_procedure_code", code=p.cpt_secondary)
    await step("check_regulatory_rule", rule_id="NCCI-MODIFIER-25")
    await step("check_deadline", deadline_type="appeal")

    eob_id = get_artifact_id(env.db, ep, "eob")
    provider_id_ = get_artifact_id(env.db, ep, "provider_record")
    # Get last code_lookup (secondary code)
    code_id = get_artifact_id(env.db, ep, "code_lookup")
    reg_id = get_artifact_id(env.db, ep, "regulatory_rule")

    await step(
        "write_diagnosis",
        responsible_party="insurer_wrong",
        evidence_artifact_ids=[eob_id, provider_id_, code_id, reg_id],
        diagnosis_text=(
            f"Provider appended modifier -25 to {p.cpt_primary}. "
            f"NCCI modifier_indicator=1 for this pair allows override. "
            "Insurer's CO-97 denial is incorrect."
        ),
    )
    await step(
        "draft_resolution",
        resolution_type="appeal",
        appeal_reason=p.appeal_reason_correct,
        summary="NCCI modifier -25 overrides bundling. Appeal filed.",
    )
    draft_id = get_artifact_id(env.db, ep, "draft_resolution")
    await step("submit_resolution", draft_artifact_id=draft_id)
    await step("send_patient_communication", message_type="outcome",
               message_text="Appeal filed. Modifier -25 overrides bundling denial.")
    await step("notify_provider", notification_type="appeal_filed",
               message="Appeal filed on patient's behalf re: CO-97 denial.")
    await step("write_audit_entry",
               summary="NCCI modifier -25 overrides CO-97 bundling. Appeal submitted.")
    result = await step("close_case", outcome_code="resolved")

    assert result.done
    score = result.info["episode_score"]
    failed = [r for r in result.info["check_results"] if not r.passed]
    assert score == 1.0, (
        f"Task 2 score={score:.4f}. "
        f"Failing: {[(r.description, r.actual) for r in failed]}"
    )


# ------------------------------------------------------------------
# Task 3 — Adversarial Fabrication
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_task3_score_1():
    """Optimal Task 3 trajectory (both phases) must score 1.0."""
    ref_db = make_ref_db()
    env = ClarusEnv(ref_db=ref_db)
    await env.reset("adversarial_fabrication", seed=3001)
    ep = env.episode_id
    p = generate("adversarial_fabrication", 3001, ref_db)

    async def step(action_type: str, **params):
        return await env.step(ClarusAction(action_type=action_type, parameters=params))

    # Phase 1 — investigation (rate limits on fetch_plan_document and check_regulatory_rule)
    await step("authenticate_patient")
    await step("fetch_claim_record", claim_id=p.claim_id)
    await step("fetch_eob", claim_id=p.claim_id)
    await step("fetch_provider_record", provider_id=p.provider_id)
    await step("fetch_payment_processor_log", claim_id=p.claim_id)
    await step("fetch_facility_record", facility_id=p.facility_id)
    # Steps 1-5 rate-limited; check_regulatory_rule available step 6+
    await step("check_regulatory_rule", rule_id="NSA-BALANCE-BILLING")
    await step("check_regulatory_rule", rule_id="NSA-GFE")
    await step("check_deadline", deadline_type="nsa_dispute")

    proc_id = get_artifact_id(env.db, ep, "processor_log")
    eob_id = get_artifact_id(env.db, ep, "eob")
    provider_rec_id = get_artifact_id(env.db, ep, "provider_record")
    reg_id = get_artifact_id(env.db, ep, "regulatory_rule")

    await step(
        "write_diagnosis",
        responsible_party="provider_fraud",
        evidence_artifact_ids=[proc_id, eob_id, provider_rec_id, reg_id],
        diagnosis_text=(
            f"provider_record.good_faith_estimate_date={p.gfe_fabricated_date} "
            f"contradicts processor_log.timestamp={p.processor_timestamp}. "
            "GFE was backdated. NSA violation confirmed."
        ),
    )
    await step(
        "draft_resolution",
        resolution_type="nsa_dispute",
        nsa_violation_basis="backdated_gfe",
        qpa_reference_amount=p.qpa_amount,
        summary="Provider backdated GFE. NSA dispute filed.",
    )
    draft_id = get_artifact_id(env.db, ep, "draft_resolution")
    r_submit = await step("submit_resolution", draft_artifact_id=draft_id)
    # After submit_resolution, Phase 2 counter-arguments are injected

    # Phase 2 — reject all three counters
    await step(
        "reject_counter_argument",
        counter_index=1,
        rejection_reasoning=(
            "Provider's own EHR records are not independent. "
            "The payment processor log is an independent system timestamp "
            "recorded at time of service and cannot be altered."
        ),
        cited_artifact_ids=[proc_id],
    )
    await step(
        "reject_counter_argument",
        counter_index=2,
        rejection_reasoning=(
            "The NSA emergency exception (§ 149.110) applies only to emergency "
            "services. This was an elective, scheduled procedure. "
            "The exception does not apply."
        ),
        cited_artifact_ids=[reg_id],
    )
    await step(
        "reject_counter_argument",
        counter_index=3,
        rejection_reasoning=(
            "Legal threat without new factual evidence. "
            "The processor log timestamp is the definitive independent record. "
            "The NSA dispute stands."
        ),
        cited_artifact_ids=[proc_id],
    )

    await step("send_patient_communication", message_type="outcome",
               message_text="NSA dispute filed. All counter-arguments rejected.")
    await step("notify_provider", notification_type="nsa_dispute_filed",
               message="NSA dispute filed. Backdated GFE contradicts processor log.")
    await step("write_audit_entry",
               summary="NSA dispute: backdated GFE detected via processor log. "
                       "All three Phase 2 counters rejected.")
    result = await step("close_case", outcome_code="resolved")

    assert result.done
    score = result.info["episode_score"]
    failed = [r for r in result.info["check_results"] if not r.passed]
    assert score == 1.0, (
        f"Task 3 score={score:.4f}. "
        f"Failing: {[(r.description, r.actual) for r in failed]}"
    )


# ------------------------------------------------------------------
# Task 2 TRAP check — premature closure scores < 0.5
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_task2_premature_closure_score_low():
    """Closing with 'legitimate_denial' after only insurer sources must score low."""
    ref_db = make_ref_db()
    env = ClarusEnv(ref_db=ref_db)
    await env.reset("abductive_conflict", seed=2001)
    ep = env.episode_id
    p = generate("abductive_conflict", 2001, ref_db)

    async def step(action_type: str, **params):
        return await env.step(ClarusAction(action_type=action_type, parameters=params))

    await step("authenticate_patient")
    await step("fetch_eob", claim_id=p.claim_id)
    await step("fetch_plan_document", plan_id=p.plan_id)
    # Premature closure: only read insurer sources, concluded legitimate_denial

    await step(
        "write_diagnosis",
        responsible_party="legitimate_denial",  # WRONG — trap fires
        evidence_artifact_ids=[],
        diagnosis_text="Insurer correctly denied per NCCI bundling.",
    )
    await step(
        "draft_resolution",
        resolution_type="appeal",
        appeal_reason="medical_necessity",  # wrong reason too
        summary="Accept denial.",
    )
    draft_id = get_artifact_id(env.db, ep, "draft_resolution")
    await step("submit_resolution", draft_artifact_id=draft_id)
    result = await step("close_case", outcome_code="resolved")

    assert result.done
    score = result.info["episode_score"]
    assert score < 0.5, (
        f"Premature closure should score < 0.5, got {score:.4f}"
    )
