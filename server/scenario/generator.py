"""Deterministic episode generator for all three Clarus tasks.

The generator is the single source of truth for all episode parameters.
Given a task name and integer seed, it produces an EpisodeParams with
all financial values, dates, codes, and ground-truth answers pre-computed.

Key invariants:
- random.Random(seed) is the only source of randomness — fully deterministic.
- TODAY is FIXED at the hackathon date; never use datetime.date.today().
- All dates are computed as explicit timedelta arithmetic, never ad-hoc strings.
- All dollar amounts are rounded to 2 decimal places at the point of creation.
- Grader reads expected values from EpisodeParams via ground_truth row;
  tests read them from generate() directly — no hardcoding anywhere.
"""

from __future__ import annotations

import random
import sqlite3
from datetime import date, timedelta
from typing import Optional

from server.scenario.params import EpisodeParams

# Fixed reference date — hackathon deadline, never changes
TODAY: date = date(2026, 4, 8)

# Possible service dates (seeded selection prevents temporal clustering)
SERVICE_DATES = [
    "2026-01-11",
    "2026-01-22",
    "2026-01-28",
    "2026-02-05",
    "2026-02-14",
    "2026-02-20",
    "2026-03-02",
    "2026-03-10",
    "2026-03-19",
    "2026-03-25",
]

# Modifier that overrides NCCI bundling for Task 2
NCCI_OVERRIDE_MODIFIER = "-25"

# Distractor types per task (injected in ~20% of seeds)
_DISTRACTOR_TYPES = [
    "regulatory_rule",
    "code_lookup",
    "payment_ledger",
    "plan_document",
    "facility_record",
]

_PATIENT_NAMES = [
    "Jordan Martinez", "Alex Chen", "Morgan Williams", "Taylor Johnson",
    "Casey Brown", "Riley Davis", "Jamie Wilson", "Avery Thompson",
    "Drew Anderson", "Parker Lewis", "Quinn Robinson", "Reese Walker",
    "Skyler Hall", "Cameron Young", "Blake Harris", "Finley Clark",
    "Peyton Allen", "Rowan Wright", "Sage King", "River Scott",
]


def _derive_responsible_party(task_name: str) -> str:
    """Return the canonical responsible_party string for a task."""
    if task_name == "deductive_liability":
        return "billing_system_error"
    if task_name == "abductive_conflict":
        return "insurer_wrong"
    if task_name == "adversarial_fabrication":
        return "provider_fraud"
    raise ValueError(f"Unknown task: {task_name}")


def _derive_resolution_type(task_name: str) -> str:
    """Return the canonical resolution_type string for a task."""
    if task_name == "deductive_liability":
        return "refund"
    if task_name == "abductive_conflict":
        return "appeal"
    if task_name == "adversarial_fabrication":
        return "nsa_dispute"
    raise ValueError(f"Unknown task: {task_name}")


def _make_dates(service_date_str: str) -> dict:
    """Compute all deadline dates from a service date string.

    Returns a dict with keys matching EpisodeParams date fields.
    """
    svc = date.fromisoformat(service_date_str)
    denial = svc + timedelta(days=14)
    eob_receipt = denial + timedelta(days=3)
    appeal_dl = eob_receipt + timedelta(days=30)
    nsa_dl = eob_receipt + timedelta(days=30)
    return {
        "service_date": service_date_str,
        "denial_date": denial.isoformat(),
        "eob_receipt_date": eob_receipt.isoformat(),
        "appeal_deadline": appeal_dl.isoformat(),
        "nsa_deadline": nsa_dl.isoformat(),
        "days_until_appeal": (appeal_dl - TODAY).days,
        "days_until_nsa": (nsa_dl - TODAY).days,
    }


def _pick_plan(
    rng: random.Random,
    ref_db: sqlite3.Connection,
    require_copay: bool = False,
) -> dict:
    """Pick a plan from plan_templates and return its fields as a dict.

    Args:
        rng: Seeded random instance.
        ref_db: Reference DB connection.
        require_copay: If True, only pick plans with copay_specialist > 0.
            Required for Task 1 (copay-not-credited scenario).
    """
    rows = ref_db.execute(
        "SELECT plan_id, plan_type, deductible_individual, coinsurance_rate, "
        "       copay_specialist, oop_max, nsa_compliant "
        "FROM plan_templates ORDER BY plan_id"
    ).fetchall()
    if require_copay:
        rows = [r for r in rows if r[4] > 0]
    row = rng.choice(rows)
    return {
        "plan_id": row[0],
        "plan_type": row[1],
        "deductible": row[2],
        "coinsurance_rate": row[3],
        "copay_specialist": row[4],
        "oop_max": row[5],
        "nsa_compliant": bool(row[6]),
    }


def _pick_active_cpt(rng: random.Random, ref_db: sqlite3.Connection) -> str:
    """Pick a random active CPT code from the reference table."""
    rows = ref_db.execute(
        "SELECT code FROM cpt_codes WHERE status='A' ORDER BY code"
    ).fetchall()
    return rng.choice(rows)[0]


def _generate_task1(
    rng: random.Random,
    seed: int,
    dates: dict,
    plan: dict,
    ref_db: sqlite3.Connection,
) -> dict:
    """Generate Task 1 (deductive_liability) specific parameters.

    Invariants enforced:
    - billed_amount > remaining_deductible (ensures after_deductible > 0)
    - correct_refund > 0 (copay was not credited — refund is always positive)
    """
    cpt_primary = _pick_active_cpt(rng, ref_db)

    # Pick billed amount first so we can constrain deductible_met
    # Use a range that leaves room for a meaningful computation
    billed_amount = round(rng.uniform(300.0, 900.0), 2)

    # Ensure remaining_deductible < billed_amount so after_deductible > 0.
    # remaining_deductible = deductible - deductible_met < billed_amount
    # → deductible_met > deductible - billed_amount
    min_deductible_met = max(0.0, plan["deductible"] - billed_amount + 1.0)
    # Cap at 99% of deductible so there is always some remaining balance
    max_deductible_met = plan["deductible"] * 0.99
    # Guard: if min > max (billed_amount > deductible), any value works
    if min_deductible_met > max_deductible_met:
        min_deductible_met = 0.0
    deductible_met = round(rng.uniform(min_deductible_met, max_deductible_met), 2)

    # Apply the canonical formula
    remaining_deductible = max(0.0, plan["deductible"] - deductible_met)
    after_deductible = max(0.0, billed_amount - remaining_deductible)
    patient_coinsurance = round(after_deductible * plan["coinsurance_rate"], 2)
    subtotal = round(remaining_deductible + patient_coinsurance, 2)
    correct_balance = round(max(0.0, subtotal - plan["copay_specialist"]), 2)
    # correct_refund = copay not credited in EOB (the exact billing error amount)
    correct_refund = round(plan["copay_specialist"], 2)

    # Copay ambiguity: ~14% of seeds (seed % 7 == 0)
    has_scheduling_deposit = (seed % 7 == 0) and plan["copay_specialist"] > 0
    scheduling_deposit = (
        round(rng.uniform(20.0, 80.0), 2) if has_scheduling_deposit else 0.0
    )

    # Distractor: ~20% of seeds
    distractor = None
    if rng.random() < 0.20:
        distractor = rng.choice(_DISTRACTOR_TYPES)

    patient_name = rng.choice(_PATIENT_NAMES)

    return {
        "cpt_primary": cpt_primary,
        "cpt_secondary": None,
        "modifier_used": None,
        "deductible_met": deductible_met,
        "billed_amount": billed_amount,
        "denial_code": "",
        "denial_reason": "",
        "correct_balance": correct_balance,
        "correct_refund": correct_refund,
        "has_scheduling_deposit": has_scheduling_deposit,
        "scheduling_deposit": scheduling_deposit,
        "initial_patient_state": rng.choice(["frustrated", "distressed"]),
        "distractor_artifact_type": distractor,
        "patient_name": patient_name,
        "patient_complaint": (
            f"I was billed ${billed_amount:.2f} for my visit "
            f"but my insurance EOB doesn't look right."
        ),
    }


def _generate_task2(
    rng: random.Random,
    seed: int,
    dates: dict,
    plan: dict,
    ref_db: sqlite3.Connection,
) -> dict:
    """Generate Task 2 (abductive_conflict) specific parameters."""
    # Pick an NCCI pair where modifier allows override (modifier_indicator=1)
    pairs = ref_db.execute(
        "SELECT col1_code, col2_code FROM ncci_edits "
        "WHERE modifier_indicator=1 "
        "  AND (deletion_date IS NULL OR deletion_date='') "
        "ORDER BY col1_code, col2_code"
    ).fetchall()
    pair = rng.choice(pairs)
    cpt_primary = pair[0]
    cpt_secondary = pair[1]
    modifier_used = NCCI_OVERRIDE_MODIFIER  # -25

    deductible_met = round(plan["deductible"] * rng.uniform(0.10, 0.95), 2)
    billed_amount = round(rng.uniform(200.0, 700.0), 2)

    # Appeal reasons — correct is "modifier_exception"; also include distractors
    appeal_reason_correct = "modifier_exception"
    appeal_reason_set = [
        "modifier_exception",
        "medical_necessity",
        "timely_filing",
    ]

    distractor = None
    if rng.random() < 0.20:
        distractor = rng.choice(_DISTRACTOR_TYPES)

    patient_name = rng.choice(_PATIENT_NAMES)

    return {
        "cpt_primary": cpt_primary,
        "cpt_secondary": cpt_secondary,
        "modifier_used": modifier_used,
        "deductible_met": deductible_met,
        "billed_amount": billed_amount,
        "denial_code": "CO-97",
        "denial_reason": (
            f"Procedure {cpt_secondary} is bundled with {cpt_primary} "
            "per NCCI edits and cannot be billed separately."
        ),
        "correct_balance": 0.0,
        "correct_refund": 0.0,
        "has_scheduling_deposit": False,
        "scheduling_deposit": 0.0,
        "appeal_reason_correct": appeal_reason_correct,
        "appeal_reason_set": appeal_reason_set,
        "initial_patient_state": rng.choice(["frustrated", "distressed"]),
        "distractor_artifact_type": distractor,
        "patient_name": patient_name,
        "patient_complaint": (
            f"My insurer denied procedure {cpt_secondary} claiming it was "
            f"bundled with {cpt_primary}, but my doctor used modifier -25."
        ),
    }


def _generate_task3(
    rng: random.Random,
    seed: int,
    dates: dict,
    plan: dict,
    ref_db: sqlite3.Connection,
) -> dict:
    """Generate Task 3 (adversarial_fabrication) specific parameters."""
    # Pick a code that has an NSA QPA rate (high-value surgical code)
    qpa_rows = ref_db.execute(
        "SELECT procedure_code, median_rate FROM nsa_qpa_rates "
        "WHERE year=2026 AND geographic_area='NATIONAL' "
        "ORDER BY procedure_code"
    ).fetchall()
    code_row = rng.choice(qpa_rows)
    cpt_primary = code_row[0]
    qpa_amount = round(code_row[1], 2)

    deductible_met = round(plan["deductible"] * rng.uniform(0.10, 0.95), 2)

    # Out-of-network balance bill: 4x–9x the QPA (NSA violation)
    billed_amount_oon = round(qpa_amount * rng.uniform(4.0, 9.0), 2)

    # Good faith estimate was backdated (the fabrication)
    backdate_days = rng.choice([1, 2, 3])
    svc_date = date.fromisoformat(dates["service_date"])
    gfe_fabricated_date = (svc_date - timedelta(days=backdate_days)).isoformat()

    # Payment processor logged the real service date + time
    hour = rng.randint(7, 17)
    minute = rng.randint(0, 59)
    processor_timestamp = f"{dates['service_date']}T{hour:02d}:{minute:02d}:00Z"

    patient_name = rng.choice(_PATIENT_NAMES)

    return {
        "cpt_primary": cpt_primary,
        "cpt_secondary": None,
        "modifier_used": None,
        "deductible_met": deductible_met,
        "billed_amount": billed_amount_oon,
        "denial_code": "",
        "denial_reason": "",
        "correct_balance": 0.0,
        "correct_refund": 0.0,
        "has_scheduling_deposit": False,
        "scheduling_deposit": 0.0,
        "billed_amount_oon": billed_amount_oon,
        "qpa_amount": qpa_amount,
        "gfe_fabricated_date": gfe_fabricated_date,
        "processor_timestamp": processor_timestamp,
        "gfe_backdate_days": backdate_days,
        # Task 3: rate-limit fetch_plan_document and check_regulatory_rule at start
        "rate_limited_at_start": ["fetch_plan_document", "check_regulatory_rule"],
        "initial_patient_state": "distressed",
        "distractor_artifact_type": None,
        "patient_name": patient_name,
        "patient_complaint": (
            f"I had elective surgery at an in-network facility and was billed "
            f"${billed_amount_oon:,.2f} out-of-network — far above my plan's rates."
        ),
    }


def generate(
    task_name: str, seed: int, ref_db: sqlite3.Connection
) -> EpisodeParams:
    """Generate a fully deterministic EpisodeParams for (task_name, seed).

    This is the single source of truth for all expected values used by
    the grader and tests.  Never call datetime.date.today() here.

    Args:
        task_name: One of "deductive_liability", "abductive_conflict",
                   "adversarial_fabrication".
        seed: Integer seed — fully determines all random choices.
        ref_db: Open SQLite connection with reference tables loaded.

    Returns:
        EpisodeParams with all fields populated.
    """
    rng = random.Random(seed)

    # Shared: dates, IDs, plan
    service_date_str = rng.choice(SERVICE_DATES)
    dates = _make_dates(service_date_str)

    # Realistic healthcare identifiers — deterministic but non-stub
    _svc_ym = dates["service_date"].replace("-", "")[:6]  # e.g. "202601"
    claim_id = f"HC{_svc_ym}{seed:06d}"                   # e.g. "HC202601001101"
    patient_id = f"MBR{(seed * 6271 % 9000000) + 1000000:07d}"  # e.g. "MBR7654321"
    # 10-digit NPI-format provider ID (Luhn-like, but deterministic)
    _npi_base = 1000000000 + (seed * 7919) % 900000000
    provider_id = f"{_npi_base:010d}"
    facility_id = f"FAC{10000 + (seed * 3541) % 89000:05d}"     # e.g. "FAC11541"

    # Task 1 requires plans with copay > 0 (copay-not-credited scenario)
    plan = _pick_plan(
        rng, ref_db, require_copay=(task_name == "deductive_liability")
    )

    # Task-specific generation
    if task_name == "deductive_liability":
        specific = _generate_task1(rng, seed, dates, plan, ref_db)
    elif task_name == "abductive_conflict":
        specific = _generate_task2(rng, seed, dates, plan, ref_db)
    elif task_name == "adversarial_fabrication":
        specific = _generate_task3(rng, seed, dates, plan, ref_db)
    else:
        raise ValueError(f"Unknown task_name: {task_name!r}")

    return EpisodeParams(
        task_name=task_name,
        episode_seed=seed,
        claim_id=claim_id,
        patient_id=patient_id,
        provider_id=provider_id,
        facility_id=facility_id,
        plan_id=plan["plan_id"],
        service_date=dates["service_date"],
        denial_date=dates["denial_date"],
        eob_receipt_date=dates["eob_receipt_date"],
        appeal_deadline=dates["appeal_deadline"],
        nsa_deadline=dates["nsa_deadline"],
        days_until_appeal=dates["days_until_appeal"],
        days_until_nsa=dates["days_until_nsa"],
        deductible=plan["deductible"],
        deductible_met=specific["deductible_met"],
        coinsurance_rate=plan["coinsurance_rate"],
        copay_specialist=plan["copay_specialist"],
        oop_max=plan["oop_max"],
        cpt_primary=specific["cpt_primary"],
        cpt_secondary=specific.get("cpt_secondary"),
        modifier_used=specific.get("modifier_used"),
        billed_amount=specific["billed_amount"],
        denial_code=specific.get("denial_code", ""),
        denial_reason=specific.get("denial_reason", ""),
        correct_balance=specific.get("correct_balance", 0.0),
        correct_refund=specific.get("correct_refund", 0.0),
        has_scheduling_deposit=specific.get("has_scheduling_deposit", False),
        scheduling_deposit=specific.get("scheduling_deposit", 0.0),
        appeal_reason_correct=specific.get("appeal_reason_correct", ""),
        appeal_reason_set=specific.get("appeal_reason_set", []),
        billed_amount_oon=specific.get("billed_amount_oon", 0.0),
        qpa_amount=specific.get("qpa_amount", 0.0),
        gfe_fabricated_date=specific.get("gfe_fabricated_date", ""),
        processor_timestamp=specific.get("processor_timestamp", ""),
        gfe_backdate_days=specific.get("gfe_backdate_days", 0),
        initial_patient_state=specific.get("initial_patient_state", "frustrated"),
        rate_limited_at_start=specific.get("rate_limited_at_start", []),
        rate_limit_cooldown_steps=4,
        distractor_artifact_type=specific.get("distractor_artifact_type"),
        patient_name=specific.get("patient_name", ""),
        patient_complaint=specific.get("patient_complaint", ""),
    )
