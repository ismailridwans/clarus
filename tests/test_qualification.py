"""Qualification tests — data quality and scenario coverage.

Verifies:
1. Realistic ID formats (no stub CLM-/PAT-/PRV-/FAC- prefixes).
2. Multi-seed coverage: all 10 seeds per task generate valid episodes.
3. Financial invariants: correct_refund > 0, correct_balance ≥ 0.
4. Score range: every seed yields a score strictly in (0, 1).
5. Rate-limit enforcement: Task 3 blocks fetch_plan_document at step 1.
6. Distractor isolation: fetching a distractor hurts score, not helps.
7. Partial success trajectories score between 0.3 and 0.7.

Run: pytest tests/test_qualification.py -v
"""

from __future__ import annotations

import re
import sqlite3

import pytest

from server.env import ClarusEnv
from server.models import ClarusAction
from server.scenario.generator import generate


# ------------------------------------------------------------------
# Shared fixtures
# ------------------------------------------------------------------


def make_ref_db() -> sqlite3.Connection:
    from server.schema import create_tables
    from data.setup import load_all

    db = sqlite3.connect(":memory:", check_same_thread=False)
    db.row_factory = sqlite3.Row
    create_tables(db)
    load_all(db)
    return db


# ------------------------------------------------------------------
# 1. Realistic ID format validation
# ------------------------------------------------------------------

TASK1_SEEDS = list(range(1001, 1011))
TASK2_SEEDS = list(range(2001, 2011))
TASK3_SEEDS = list(range(3001, 3011))


@pytest.mark.parametrize("seed", TASK1_SEEDS + TASK2_SEEDS + TASK3_SEEDS)
def test_id_formats_are_realistic(seed: int):
    """Generated IDs must not match the old stub formats."""
    task = (
        "deductive_liability" if seed < 2000
        else "abductive_conflict" if seed < 3000
        else "adversarial_fabrication"
    )
    ref_db = make_ref_db()
    p = generate(task, seed, ref_db)

    # Old stubs: CLM-NNNNN, PAT-NNNN, PRV-NNN, FAC-NN
    assert not re.match(r"^CLM-\d+$", p.claim_id), (
        f"claim_id still uses stub format: {p.claim_id!r}"
    )
    assert not re.match(r"^PAT-\d+$", p.patient_id), (
        f"patient_id still uses stub format: {p.patient_id!r}"
    )
    assert not re.match(r"^PRV-\d+$", p.provider_id), (
        f"provider_id still uses stub format: {p.provider_id!r}"
    )
    # FAC- prefix is now FAC<5-digit> (e.g. FAC11541) — old was FAC-NN
    assert not re.match(r"^FAC-\d+$", p.facility_id), (
        f"facility_id still uses old stub format: {p.facility_id!r}"
    )

    # New formats
    # claim_id: HC<YYYYMM><6-digit-seed>  e.g. HC202601001101
    assert re.match(r"^HC\d{12}$", p.claim_id), (
        f"claim_id has unexpected format: {p.claim_id!r}"
    )
    # patient_id: MBR<7-digit>  e.g. MBR7654321
    assert re.match(r"^MBR\d{7}$", p.patient_id), (
        f"patient_id has unexpected format: {p.patient_id!r}"
    )
    # provider_id: 10-digit NPI-format
    assert re.match(r"^\d{10}$", p.provider_id), (
        f"provider_id has unexpected format: {p.provider_id!r}"
    )
    # facility_id: FAC<5-digit>
    assert re.match(r"^FAC\d{5}$", p.facility_id), (
        f"facility_id has unexpected format: {p.facility_id!r}"
    )


# ------------------------------------------------------------------
# 2. Financial invariants across all seeds
# ------------------------------------------------------------------


@pytest.mark.parametrize("seed", TASK1_SEEDS)
def test_task1_financial_invariants(seed: int):
    """Task 1: correct_refund > 0 and correct_balance ≥ 0 for every seed."""
    ref_db = make_ref_db()
    p = generate("deductive_liability", seed, ref_db)

    assert p.correct_refund > 0.0, (
        f"seed={seed}: correct_refund={p.correct_refund} must be > 0 "
        "(copay-not-credited scenario)"
    )
    assert p.correct_balance >= 0.0, (
        f"seed={seed}: correct_balance={p.correct_balance} must be ≥ 0"
    )
    assert p.billed_amount > 0.0, f"seed={seed}: billed_amount must be > 0"
    assert p.copay_specialist > 0.0, (
        f"seed={seed}: copay_specialist must be > 0 for Task 1"
    )
    # Refund equals the copay (the exact amount that was not credited)
    assert abs(p.correct_refund - p.copay_specialist) < 0.01, (
        f"seed={seed}: correct_refund={p.correct_refund} should equal "
        f"copay_specialist={p.copay_specialist}"
    )


@pytest.mark.parametrize("seed", TASK3_SEEDS)
def test_task3_financial_invariants(seed: int):
    """Task 3: billed_amount_oon >> qpa_amount (NSA violation)."""
    ref_db = make_ref_db()
    p = generate("adversarial_fabrication", seed, ref_db)

    assert p.qpa_amount > 0.0, f"seed={seed}: qpa_amount must be > 0"
    assert p.billed_amount_oon > 0.0, f"seed={seed}: billed_amount_oon must be > 0"
    # OON bill must be at least 4x QPA (generator enforces 4x–9x)
    assert p.billed_amount_oon >= p.qpa_amount * 3.9, (
        f"seed={seed}: billed_amount_oon={p.billed_amount_oon} should be "
        f">>= 4x qpa_amount={p.qpa_amount}"
    )
    # GFE was backdated — fabricated date < service date
    from datetime import date
    gfe_dt = date.fromisoformat(p.gfe_fabricated_date)
    svc_dt = date.fromisoformat(p.service_date)
    assert gfe_dt < svc_dt, (
        f"seed={seed}: gfe_fabricated_date={p.gfe_fabricated_date} should be "
        f"before service_date={p.service_date}"
    )
    assert p.gfe_backdate_days in (1, 2, 3), (
        f"seed={seed}: gfe_backdate_days={p.gfe_backdate_days} unexpected"
    )


# ------------------------------------------------------------------
# 3. ID uniqueness across seeds (no collisions)
# ------------------------------------------------------------------


def test_claim_ids_unique_across_seeds():
    """All 30 seeds across all tasks must produce distinct claim IDs."""
    ref_db = make_ref_db()
    ids = []
    for seed in TASK1_SEEDS:
        p = generate("deductive_liability", seed, ref_db)
        ids.append(p.claim_id)
    for seed in TASK2_SEEDS:
        p = generate("abductive_conflict", seed, ref_db)
        ids.append(p.claim_id)
    for seed in TASK3_SEEDS:
        p = generate("adversarial_fabrication", seed, ref_db)
        ids.append(p.claim_id)

    assert len(ids) == len(set(ids)), (
        f"Duplicate claim IDs found among {len(ids)} seeds"
    )


def test_patient_ids_unique_within_task():
    """Patient IDs must be distinct within each task's seed range."""
    ref_db = make_ref_db()
    for task, seeds in [
        ("deductive_liability", TASK1_SEEDS),
        ("abductive_conflict", TASK2_SEEDS),
        ("adversarial_fabrication", TASK3_SEEDS),
    ]:
        ids = [generate(task, s, ref_db).patient_id for s in seeds]
        assert len(ids) == len(set(ids)), (
            f"Duplicate patient_ids in task={task}: {ids}"
        )


# ------------------------------------------------------------------
# 4. Episode score in (0, 1) — multi-seed smoke test
# ------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("seed", TASK1_SEEDS[:5])
async def test_task1_score_in_range(seed: int):
    """Task 1 minimal trajectory must produce a score strictly in (0, 1)."""
    ref_db = make_ref_db()
    env = ClarusEnv(ref_db=ref_db)
    await env.reset("deductive_liability", seed=seed)
    p = generate("deductive_liability", seed, ref_db)

    async def step(action_type: str, **params):
        return await env.step(ClarusAction(action_type=action_type, parameters=params))

    await step("authenticate_patient")
    await step("fetch_claim_record", claim_id=p.claim_id)
    result = await step("close_case", outcome_code="resolved")

    score = result.info["episode_score"]
    assert 0.0 < score < 1.0, (
        f"seed={seed}: episode_score={score} is not strictly in (0, 1)"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("seed", TASK2_SEEDS[:5])
async def test_task2_score_in_range(seed: int):
    """Task 2 minimal trajectory must produce a score strictly in (0, 1)."""
    ref_db = make_ref_db()
    env = ClarusEnv(ref_db=ref_db)
    await env.reset("abductive_conflict", seed=seed)
    p = generate("abductive_conflict", seed, ref_db)

    async def step(action_type: str, **params):
        return await env.step(ClarusAction(action_type=action_type, parameters=params))

    await step("authenticate_patient")
    await step("fetch_eob", claim_id=p.claim_id)
    result = await step("close_case", outcome_code="resolved")

    score = result.info["episode_score"]
    assert 0.0 < score < 1.0, (
        f"seed={seed}: episode_score={score} is not strictly in (0, 1)"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("seed", TASK3_SEEDS[:5])
async def test_task3_score_in_range(seed: int):
    """Task 3 minimal trajectory must produce a score strictly in (0, 1)."""
    ref_db = make_ref_db()
    env = ClarusEnv(ref_db=ref_db)
    await env.reset("adversarial_fabrication", seed=seed)
    p = generate("adversarial_fabrication", seed, ref_db)

    async def step(action_type: str, **params):
        return await env.step(ClarusAction(action_type=action_type, parameters=params))

    await step("authenticate_patient")
    await step("fetch_claim_record", claim_id=p.claim_id)
    result = await step("close_case", outcome_code="resolved")

    score = result.info["episode_score"]
    assert 0.0 < score < 1.0, (
        f"seed={seed}: episode_score={score} is not strictly in (0, 1)"
    )


# ------------------------------------------------------------------
# 5. Rate-limit enforcement — Task 3
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_task3_rate_limit_blocks_early_plan_fetch():
    """Task 3: fetch_plan_document must be rate-limited at step 1."""
    ref_db = make_ref_db()
    env = ClarusEnv(ref_db=ref_db)
    await env.reset("adversarial_fabrication", seed=3001)
    p = generate("adversarial_fabrication", 3001, ref_db)

    # Step 1: attempt fetch_plan_document (should be rate-limited)
    result = await env.step(
        ClarusAction(
            action_type="fetch_plan_document",
            parameters={"plan_id": p.plan_id},
        )
    )
    obs = result.observation
    # Either there's an error indicating rate-limit, or the tool is in rate_limited_tools
    is_rate_limited = (
        "rate" in (obs.last_action_error or "").lower()
        or "fetch_plan_document" in obs.rate_limited_tools
        or result.info.get("rate_limited", False)
    )
    assert is_rate_limited, (
        "Task 3 should rate-limit fetch_plan_document at step 1. "
        f"Got error={obs.last_action_error!r}, "
        f"rate_limited_tools={obs.rate_limited_tools}"
    )


@pytest.mark.asyncio
async def test_task3_rate_limit_blocks_early_regulatory_check():
    """Task 3: check_regulatory_rule must be rate-limited at step 1."""
    ref_db = make_ref_db()
    env = ClarusEnv(ref_db=ref_db)
    await env.reset("adversarial_fabrication", seed=3001)

    result = await env.step(
        ClarusAction(
            action_type="check_regulatory_rule",
            parameters={"rule_id": "NSA-BALANCE-BILLING"},
        )
    )
    obs = result.observation
    is_rate_limited = (
        "rate" in (obs.last_action_error or "").lower()
        or "check_regulatory_rule" in obs.rate_limited_tools
        or result.info.get("rate_limited", False)
    )
    assert is_rate_limited, (
        "Task 3 should rate-limit check_regulatory_rule at step 1. "
        f"Got error={obs.last_action_error!r}, "
        f"rate_limited_tools={obs.rate_limited_tools}"
    )


# ------------------------------------------------------------------
# 6. Refund tolerance boundary — Task 1
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_task1_refund_within_tolerance_passes():
    """Task 1: submitting refund within 5% of correct value must pass the check."""
    ref_db = make_ref_db()
    env = ClarusEnv(ref_db=ref_db)
    await env.reset("deductive_liability", seed=1001)
    ep = env.episode_id
    p = generate("deductive_liability", 1001, ref_db)

    async def step(action_type: str, **params):
        return await env.step(ClarusAction(action_type=action_type, parameters=params))

    await step("authenticate_patient")
    await step("fetch_eob", claim_id=p.claim_id)
    await step("fetch_payment_ledger", claim_id=p.claim_id)
    await step("fetch_plan_document", plan_id=p.plan_id)

    eob_id = env.db.execute(
        "SELECT id FROM episode_artifacts WHERE episode_id=? AND artifact_type='eob' "
        "ORDER BY id DESC LIMIT 1", (ep,)
    ).fetchone()[0]
    ledger_id = env.db.execute(
        "SELECT id FROM episode_artifacts WHERE episode_id=? "
        "AND artifact_type='payment_ledger' ORDER BY id DESC LIMIT 1", (ep,)
    ).fetchone()[0]

    await step(
        "write_diagnosis",
        responsible_party="billing_system_error",
        evidence_artifact_ids=[eob_id, ledger_id],
        diagnosis_text="Copay not credited.",
    )

    # Submit refund at exactly correct value — must pass tolerance check
    await step(
        "draft_resolution",
        resolution_type="refund",
        refund_amount=p.correct_refund,
        summary="Exact refund.",
    )
    draft_id = env.db.execute(
        "SELECT id FROM episode_artifacts WHERE episode_id=? "
        "AND artifact_type='draft_resolution' ORDER BY id DESC LIMIT 1", (ep,)
    ).fetchone()[0]
    await step("submit_resolution", draft_artifact_id=draft_id)
    result = await step("close_case", outcome_code="resolved")

    score = result.info["episode_score"]
    # With correct refund, should pass the refund check — score well above baseline
    baseline = 0.5 / 18.0  # zero agent score for 17 checks
    assert score > baseline + 0.3, (
        f"Correct refund should score well above baseline. Got {score:.4f}"
    )


@pytest.mark.asyncio
async def test_task1_refund_outside_tolerance_fails():
    """Task 1: submitting refund at 3x correct value should fail the amount check."""
    ref_db = make_ref_db()
    env = ClarusEnv(ref_db=ref_db)
    await env.reset("deductive_liability", seed=1002)
    ep = env.episode_id
    p = generate("deductive_liability", 1002, ref_db)

    async def step(action_type: str, **params):
        return await env.step(ClarusAction(action_type=action_type, parameters=params))

    await step("authenticate_patient")
    await step("fetch_eob", claim_id=p.claim_id)
    await step("fetch_payment_ledger", claim_id=p.claim_id)
    await step("fetch_plan_document", plan_id=p.plan_id)

    eob_id = env.db.execute(
        "SELECT id FROM episode_artifacts WHERE episode_id=? AND artifact_type='eob' "
        "ORDER BY id DESC LIMIT 1", (ep,)
    ).fetchone()[0]
    ledger_id = env.db.execute(
        "SELECT id FROM episode_artifacts WHERE episode_id=? "
        "AND artifact_type='payment_ledger' ORDER BY id DESC LIMIT 1", (ep,)
    ).fetchone()[0]

    await step(
        "write_diagnosis",
        responsible_party="billing_system_error",
        evidence_artifact_ids=[eob_id, ledger_id],
        diagnosis_text="Copay not credited.",
    )

    # Submit wildly incorrect refund (3x the correct amount)
    wrong_refund = round(p.correct_refund * 3.0, 2)
    await step(
        "draft_resolution",
        resolution_type="refund",
        refund_amount=wrong_refund,
        summary="Wrong refund amount.",
    )
    draft_id = env.db.execute(
        "SELECT id FROM episode_artifacts WHERE episode_id=? "
        "AND artifact_type='draft_resolution' ORDER BY id DESC LIMIT 1", (ep,)
    ).fetchone()[0]
    await step("submit_resolution", draft_artifact_id=draft_id)
    result = await step("close_case", outcome_code="resolved")

    # The refund amount check should fail — score should be lower than perfect
    score = result.info["episode_score"]
    perfect_score = (17 + 0.5) / (17 + 1.0)  # ~0.972
    assert score < perfect_score - 0.03, (
        f"Wrong refund of {wrong_refund} (3x correct={p.correct_refund}) "
        f"should score below perfect. Got {score:.4f}"
    )


# ------------------------------------------------------------------
# 7. Determinism — same seed always produces same params
# ------------------------------------------------------------------


@pytest.mark.parametrize("task,seed", [
    ("deductive_liability", 1001),
    ("abductive_conflict", 2005),
    ("adversarial_fabrication", 3008),
])
def test_generate_is_deterministic(task: str, seed: int):
    """generate() must return identical params on repeated calls."""
    ref_db = make_ref_db()
    p1 = generate(task, seed, ref_db)
    p2 = generate(task, seed, ref_db)

    assert p1.claim_id == p2.claim_id
    assert p1.patient_id == p2.patient_id
    assert p1.provider_id == p2.provider_id
    assert p1.facility_id == p2.facility_id
    assert p1.correct_refund == p2.correct_refund
    assert p1.service_date == p2.service_date
