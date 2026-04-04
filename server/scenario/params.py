"""EpisodeParams dataclass — canonical single schema for all tasks.

All three task generators return an instance of this class.  Every field
is populated explicitly; no field is left at its default unless it genuinely
does not apply to that task.  Dates are always explicit strings (ISO-8601);
deadlines are always derived from those strings, never from ad-hoc proxies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EpisodeParams:
    """Complete episode configuration derived deterministically from seed."""

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------
    task_name: str
    episode_seed: int

    claim_id: str    # f"CLM-{seed:05d}"
    patient_id: str  # f"PAT-{seed % 500:04d}"
    provider_id: str # f"PRV-{seed % 100:03d}"
    facility_id: str # f"FAC-{seed % 20:02d}"
    plan_id: str     # from plan_templates table

    # ------------------------------------------------------------------
    # Dates — ALL explicit, never proxied
    # ------------------------------------------------------------------
    service_date: str       # "2026-01-15"
    denial_date: str        # service_date + 14 days
    eob_receipt_date: str   # denial_date + 3 days
    appeal_deadline: str    # eob_receipt_date + 30 days (45 CFR § 147.136)
    nsa_deadline: str       # eob_receipt_date + 30 days (45 CFR § 149.510)
    days_until_appeal: int  # (appeal_deadline - TODAY).days; may be negative
    days_until_nsa: int     # (nsa_deadline - TODAY).days; may be negative

    # ------------------------------------------------------------------
    # Plan parameters (queried from plan_templates)
    # ------------------------------------------------------------------
    deductible: float
    deductible_met: float
    coinsurance_rate: float
    copay_specialist: float
    oop_max: float

    # ------------------------------------------------------------------
    # CPT codes
    # ------------------------------------------------------------------
    cpt_primary: str
    cpt_secondary: Optional[str] = None   # Task 2 only
    modifier_used: Optional[str] = None   # Task 2 only (e.g. "-25")

    # ------------------------------------------------------------------
    # Task 1 — Deductive Liability
    # ------------------------------------------------------------------
    billed_amount: float = 0.0      # Total claim amount submitted by provider
    denial_code: str = ""
    denial_reason: str = ""
    correct_balance: float = 0.0    # Correct patient share (pre-computed)
    correct_refund: float = 0.0     # Correct refund amount (pre-computed)
    # Payment ledger ambiguity flag (15% of seeds)
    has_scheduling_deposit: bool = False
    scheduling_deposit: float = 0.0

    # ------------------------------------------------------------------
    # Task 2 — Abductive Conflict
    # ------------------------------------------------------------------
    appeal_reason_correct: str = ""
    appeal_reason_set: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Task 3 — Adversarial Fabrication
    # ------------------------------------------------------------------
    billed_amount_oon: float = 0.0    # Out-of-network balance bill (inflated)
    qpa_amount: float = 0.0           # Correct QPA from nsa_qpa_rates table
    gfe_fabricated_date: str = ""     # service_date - backdate_days (the lie)
    processor_timestamp: str = ""     # service_date + time (independent truth)
    gfe_backdate_days: int = 0

    # ------------------------------------------------------------------
    # Environment configuration
    # ------------------------------------------------------------------
    initial_patient_state: str = "frustrated"
    rate_limited_at_start: List[str] = field(default_factory=list)
    rate_limit_cooldown_steps: int = 4
    distractor_artifact_type: Optional[str] = None
    patient_name: str = ""
    patient_complaint: str = ""
