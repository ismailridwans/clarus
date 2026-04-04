"""Artifact payload factory for all nine read-action artifact types.

get_seeded_payload(action_type, params, ref_db) is the single entry point.
It returns a dict that becomes the content field of an episode_artifact row.
The artifact_id is NOT included here — it is assigned by the DB and injected
by execute_read_action() after insertion.

For distractor seeds the caller checks is_distractor() before calling here;
get_seeded_payload() always returns the genuine payload.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Dict, Optional

from server.scenario.params import EpisodeParams


def _get_cpt_info(code: str, ref_db: sqlite3.Connection) -> Dict[str, Any]:
    """Look up a CPT code in the reference table; return a safe dict."""
    row = ref_db.execute(
        "SELECT code, short_desc, long_desc, rvu_work, rvu_total, status "
        "FROM cpt_codes WHERE code=?",
        (code,),
    ).fetchone()
    if row is None:
        return {
            "code": code,
            "short_desc": "Unknown",
            "long_desc": "",
            "rvu_work": 0.0,
            "rvu_total": 0.0,
            "status": "A",
        }
    return {
        "code": row[0],
        "short_desc": row[1],
        "long_desc": row[2] or "",
        "rvu_work": row[3] or 0.0,
        "rvu_total": row[4] or 0.0,
        "status": row[5] or "A",
    }


def _get_ncci_info(
    code: str, ref_db: sqlite3.Connection
) -> Dict[str, list]:
    """Return NCCI bundling info for a CPT code as col1."""
    rows = ref_db.execute(
        "SELECT col2_code, modifier_indicator FROM ncci_edits "
        "WHERE col1_code=? AND (deletion_date IS NULL OR deletion_date='')",
        (code,),
    ).fetchall()
    bundled = [r[0] for r in rows if r[1] == 0]
    modifier_exceptions = [r[0] for r in rows if r[1] == 1]
    return {
        "ncci_bundled_with": bundled,
        "ncci_modifier_exceptions": modifier_exceptions,
    }


# ------------------------------------------------------------------
# Individual payload builders
# ------------------------------------------------------------------


def _claim_record(params: EpisodeParams) -> Dict[str, Any]:
    """Payload for fetch_claim_record."""
    return {
        "claim_id": params.claim_id,
        "patient_id": params.patient_id,
        "provider_id": params.provider_id,
        "facility_id": params.facility_id,
        "service_date": params.service_date,
        "denial_date": params.denial_date if params.denial_code else None,
        "cpt_primary": params.cpt_primary,
        "cpt_secondary": params.cpt_secondary,
        "billed_amount": params.billed_amount,
        "denial_code": params.denial_code or None,
        "denial_reason": params.denial_reason or None,
        "status": "denied" if params.denial_code else "pending_review",
    }


def _eob(params: EpisodeParams) -> Dict[str, Any]:
    """Payload for fetch_eob — contains the billing error the agent must find."""
    if params.task_name == "deductive_liability":
        # Error: copay not credited in patient_responsibility
        remaining_deductible = max(0.0, params.deductible - params.deductible_met)
        after_deductible = max(0.0, params.billed_amount - remaining_deductible)
        patient_coinsurance = round(after_deductible * params.coinsurance_rate, 2)
        subtotal = round(remaining_deductible + patient_coinsurance, 2)
        insurer_paid = round(params.billed_amount - subtotal, 2)
        return {
            "claim_id": params.claim_id,
            "patient_responsibility": subtotal,  # WRONG — copay not credited
            "copay_applied": False,               # THE ERROR
            "denial_code": None,
            "denial_reason": None,
            "insurer_paid": insurer_paid,
            "eob_date": params.eob_receipt_date,
            "service_date": params.service_date,
        }

    if params.task_name == "abductive_conflict":
        # Denial due to NCCI bundling — looks legitimate until agent checks modifier
        return {
            "claim_id": params.claim_id,
            "patient_responsibility": params.billed_amount,
            "copay_applied": False,
            "denial_code": "CO-97",
            "denial_reason": (
                f"Procedure {params.cpt_secondary} is bundled with "
                f"{params.cpt_primary} per NCCI edits."
            ),
            "adjustment_reason": (
                "Column 2 procedure included in Column 1 reimbursement."
            ),
            "insurer_paid": 0.0,
            "eob_date": params.eob_receipt_date,
            "service_date": params.service_date,
        }

    if params.task_name == "adversarial_fabrication":
        # Out-of-network balance bill at in-network facility
        return {
            "claim_id": params.claim_id,
            "patient_responsibility": params.billed_amount_oon,
            "copay_applied": False,
            "denial_code": None,
            "denial_reason": None,
            "out_of_network": True,
            "insurer_paid": 0.0,
            "balance_bill_amount": params.billed_amount_oon,
            "eob_date": params.eob_receipt_date,
            "service_date": params.service_date,
        }

    raise ValueError(f"Unknown task: {params.task_name}")


def _provider_record(params: EpisodeParams) -> Dict[str, Any]:
    """Payload for fetch_provider_record."""
    if params.task_name == "deductive_liability":
        return {
            "provider_id": params.provider_id,
            "submitted_codes": [params.cpt_primary],
            "submitted_amounts": {params.cpt_primary: params.billed_amount},
            "submission_date": params.service_date,
            "modifier_on_primary_code": None,
        }

    if params.task_name == "abductive_conflict":
        return {
            "provider_id": params.provider_id,
            "submitted_codes": [params.cpt_primary, params.cpt_secondary],
            "submitted_amounts": {
                params.cpt_primary: round(params.billed_amount * 0.7, 2),
                params.cpt_secondary: round(params.billed_amount * 0.3, 2),
            },
            "submission_date": params.service_date,
            "modifier_on_primary_code": params.modifier_used,  # "-25"
        }

    if params.task_name == "adversarial_fabrication":
        # THE FABRICATION: good_faith_estimate_date is backdated
        return {
            "provider_id": params.provider_id,
            "submitted_codes": [params.cpt_primary],
            "submitted_amounts": {params.cpt_primary: params.billed_amount_oon},
            "submission_date": params.service_date,
            "modifier_on_primary_code": None,
            "good_faith_estimate_date": params.gfe_fabricated_date,  # WRONG DATE
            "facility_id": params.facility_id,
            "procedure_type": "elective",
        }

    raise ValueError(f"Unknown task: {params.task_name}")


def _payment_ledger(params: EpisodeParams) -> Dict[str, Any]:
    """Payload for fetch_payment_ledger."""
    if params.task_name == "deductive_liability":
        if params.has_scheduling_deposit:
            # Ambiguous: two entries — agent must use total_paid (copay only)
            payments = [
                {
                    "date": params.service_date,
                    "amount": params.scheduling_deposit,
                    "description": "scheduling deposit",
                    "type": "deposit",
                },
                {
                    "date": params.service_date,
                    "amount": params.copay_specialist,
                    "description": "service copay",
                    "type": "copay",
                },
            ]
        else:
            payments = [
                {
                    "date": params.service_date,
                    "amount": params.copay_specialist,
                    "description": "service copay",
                    "type": "copay",
                }
            ]
        return {
            "claim_id": params.claim_id,
            "payments": payments,
            "total_paid": params.copay_specialist,  # always copay only
        }

    # Tasks 2 and 3: no payments relevant
    return {
        "claim_id": params.claim_id,
        "payments": [],
        "total_paid": 0.0,
    }


def _plan_document(params: EpisodeParams) -> Dict[str, Any]:
    """Payload for fetch_plan_document — the insurer's plan rules."""
    return {
        "plan_id": params.plan_id,
        "deductible_individual": params.deductible,
        "deductible_met": params.deductible_met,
        "deductible_remaining": round(
            max(0.0, params.deductible - params.deductible_met), 2
        ),
        "coinsurance_rate": params.coinsurance_rate,
        "copay_specialist": params.copay_specialist,
        "oop_max": params.oop_max,
        "bundled_procedures_not_reimbursable": True,
        "nsa_compliant": True,
        "appeal_deadline_days": 30,
        "effective_date": "2026-01-01",
    }


def _code_lookup(
    params: EpisodeParams, ref_db: sqlite3.Connection, code: Optional[str] = None
) -> Dict[str, Any]:
    """Payload for lookup_procedure_code.

    If code is None, uses params.cpt_primary.
    """
    target = code or params.cpt_primary
    cpt_info = _get_cpt_info(target, ref_db)
    ncci_info = _get_ncci_info(target, ref_db)
    return {
        "code": target,
        "description": cpt_info["short_desc"],
        "long_description": cpt_info["long_desc"],
        "status": cpt_info["status"],
        "rvu_work": cpt_info["rvu_work"],
        "rvu_total": cpt_info["rvu_total"],
        "ncci_bundled_with": ncci_info["ncci_bundled_with"],
        "ncci_modifier_exceptions": ncci_info["ncci_modifier_exceptions"],
        "source": "CMS_NCCI_2026Q1",
    }


def _facility_record(params: EpisodeParams) -> Dict[str, Any]:
    """Payload for fetch_facility_record."""
    return {
        "facility_id": params.facility_id,
        "facility_name": f"Regional Medical Center {params.facility_id}",
        "network_status": "IN_NETWORK",
        "npi": f"10000{params.facility_id.replace('FAC-', '')}",
        "participates_with_plan": params.plan_id,
        "nsa_applicable": True,
    }


def _processor_log(params: EpisodeParams) -> Dict[str, Any]:
    """Payload for fetch_payment_processor_log — Task 3 only.

    The processor_timestamp is the REAL service date+time recorded by
    an independent system.  It contradicts the fabricated GFE date.
    """
    return {
        "transaction_id": f"TXN-{params.episode_seed:07d}",
        "timestamp": params.processor_timestamp,  # REAL date — contradicts GFE
        "merchant_id": params.facility_id,
        "patient_id": params.patient_id,
        "claim_id": params.claim_id,
        "amount": params.billed_amount_oon,
        "processor_response": "APPROVED",
        "note": (
            "Authorization timestamp recorded by payment processor "
            "at time of service. Independent of provider billing system."
        ),
    }


def _regulatory_rule(rule_id: str) -> Dict[str, Any]:
    """Payload for check_regulatory_rule."""
    from server.tools.regulatory import REGULATORY_RULES

    rule = REGULATORY_RULES.get(rule_id)
    if rule is None:
        return {
            "rule_id": rule_id,
            "error": f"Rule {rule_id!r} not found in regulatory database.",
        }
    return rule.copy()


def _deadline_check(params: EpisodeParams) -> Dict[str, Any]:
    """Payload for check_deadline."""
    if params.task_name in ("deductive_liability", "abductive_conflict"):
        return {
            "deadline_type": "appeal",
            "deadline_date": params.appeal_deadline,
            "days_remaining": params.days_until_appeal,
            "expired": params.days_until_appeal < 0,
            "eob_receipt_date": params.eob_receipt_date,
            "regulatory_basis": "45 CFR § 147.136",
        }
    # Task 3
    return {
        "deadline_type": "nsa_dispute",
        "deadline_date": params.nsa_deadline,
        "days_remaining": params.days_until_nsa,
        "expired": params.days_until_nsa < 0,
        "eob_receipt_date": params.eob_receipt_date,
        "regulatory_basis": "45 CFR § 149.510",
    }


# ------------------------------------------------------------------
# Public entry point
# ------------------------------------------------------------------

# Maps action_type → artifact_type
ACTION_TO_ARTIFACT_TYPE: Dict[str, str] = {
    "fetch_claim_record": "claim_record",
    "fetch_eob": "eob",
    "fetch_provider_record": "provider_record",
    "fetch_payment_ledger": "payment_ledger",
    "fetch_plan_document": "plan_document",
    "lookup_procedure_code": "code_lookup",
    "fetch_facility_record": "facility_record",
    "fetch_payment_processor_log": "processor_log",
    "check_regulatory_rule": "regulatory_rule",
    "check_deadline": "deadline_check",
}


def get_seeded_payload(
    action_type: str,
    params: EpisodeParams,
    ref_db: sqlite3.Connection,
    action_parameters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return the genuine payload dict for the given read action and episode.

    Does NOT include artifact_id (assigned by DB after insertion).
    Does NOT check for distractor — caller must do that first.

    Args:
        action_type: One of the read action names.
        params: EpisodeParams for this episode.
        ref_db: Reference DB connection (for CPT/NCCI lookups).
        action_parameters: Optional dict of action-specific parameters
            (e.g., {"code": "99213"} for lookup_procedure_code).

    Returns:
        Payload dict to store as artifact content JSON.

    Raises:
        ValueError: If action_type is not a known read action.
    """
    ap = action_parameters or {}

    if action_type == "fetch_claim_record":
        return _claim_record(params)
    if action_type == "fetch_eob":
        return _eob(params)
    if action_type == "fetch_provider_record":
        return _provider_record(params)
    if action_type == "fetch_payment_ledger":
        return _payment_ledger(params)
    if action_type == "fetch_plan_document":
        return _plan_document(params)
    if action_type == "lookup_procedure_code":
        return _code_lookup(params, ref_db, code=ap.get("code"))
    if action_type == "fetch_facility_record":
        return _facility_record(params)
    if action_type == "fetch_payment_processor_log":
        return _processor_log(params)
    if action_type == "check_regulatory_rule":
        rule_id = ap.get("rule_id", "NSA-BALANCE-BILLING")
        return _regulatory_rule(rule_id)
    if action_type == "check_deadline":
        return _deadline_check(params)

    raise ValueError(
        f"Unknown read action_type: {action_type!r}. "
        f"Valid types: {list(ACTION_TO_ARTIFACT_TYPE)}"
    )
