"""Static regulatory rules dictionary for Clarus.

These are verbatim summaries of 45 CFR regulations and CMS NCCI policy.
The check_regulatory_rule action looks up entries from this dict by rule_id.
Each entry mirrors the structure of a regulatory_rule artifact payload.
"""

from __future__ import annotations

from typing import Dict, Any


REGULATORY_RULES: Dict[str, Dict[str, Any]] = {
    # ------------------------------------------------------------------
    # NSA (No Surprises Act) — 45 CFR §§ 149.110–149.710
    # ------------------------------------------------------------------
    "NSA-BALANCE-BILLING": {
        "rule_id": "NSA-BALANCE-BILLING",
        "title": "No Surprises Act — Balance Billing Prohibition",
        "citation": "45 CFR § 149.410",
        "summary": (
            "A provider who is a participating provider with respect to a "
            "health plan may not charge a participant, beneficiary, or enrollee "
            "more than the cost-sharing amount for emergency services or certain "
            "non-emergency services furnished at in-network facilities. "
            "For non-emergency services at in-network facilities, the provider "
            "must provide a Good Faith Estimate (GFE) at least one business day "
            "before the scheduled service."
        ),
        "applies_to": ["non_emergency", "in_network_facility"],
        "does_not_apply_to": ["emergency_services", "out_of_network_facility"],
        "enforcement": "Patient may initiate independent dispute resolution (IDR) process.",
    },
    "NSA-GFE": {
        "rule_id": "NSA-GFE",
        "title": "No Surprises Act — Good Faith Estimate Requirements",
        "citation": "45 CFR § 149.610",
        "summary": (
            "Providers must furnish a Good Faith Estimate (GFE) to uninsured "
            "and self-pay patients upon request, and to scheduled patients at "
            "least one business day before the service. The GFE must include "
            "expected charges for the primary service and co-providers. "
            "Providers may not backdate GFEs; the GFE date must precede the "
            "service date and must be the date the estimate was actually prepared."
        ),
        "applies_to": ["scheduled_services", "uninsured", "self_pay"],
        "does_not_apply_to": ["emergency_services"],
        "enforcement": "Backdated GFEs are fraudulent and subject to IDR and civil penalties.",
    },
    "NSA-IDR": {
        "rule_id": "NSA-IDR",
        "title": "No Surprises Act — Independent Dispute Resolution",
        "citation": "45 CFR § 149.510",
        "summary": (
            "After receiving an Explanation of Benefits (EOB), a patient has "
            "30 days to initiate the federal independent dispute resolution (IDR) "
            "process for out-of-network charges at in-network facilities. "
            "The IDR entity uses the Qualifying Payment Amount (QPA) as the "
            "presumptive benchmark. The QPA is the median in-network rate for "
            "the same or similar service in the same geographic area."
        ),
        "deadline_from_eob": 30,
        "deadline_unit": "days",
        "benchmark": "QPA (median in-network rate via MPFS)",
        "applies_to": ["non_emergency", "in_network_facility", "out_of_network_provider"],
        "does_not_apply_to": ["emergency_services"],
    },
    "NSA-EMERGENCY-EXCEPTION": {
        "rule_id": "NSA-EMERGENCY-EXCEPTION",
        "title": "No Surprises Act — Emergency Services Exception",
        "citation": "45 CFR § 149.110(a)",
        "summary": (
            "The balance billing prohibitions of the No Surprises Act apply to "
            "emergency services without prior authorization. However, the NSA "
            "emergency exception (§ 149.110) covers emergency services only — "
            "defined as services needed to treat an emergency medical condition "
            "requiring immediate medical attention. Elective, scheduled, "
            "non-emergency procedures are NOT covered by this exception and "
            "remain subject to all NSA balance billing protections."
        ),
        "applies_to": ["emergency_services"],
        "does_not_apply_to": [
            "elective_procedures",
            "scheduled_non_emergency",
            "non_emergency_in_network",
        ],
        "note": (
            "A provider citing the emergency exception for an elective scheduled "
            "procedure is misapplying the regulation. The exception does not "
            "override NSA protections for non-emergency services."
        ),
    },
    # ------------------------------------------------------------------
    # NCCI (National Correct Coding Initiative)
    # ------------------------------------------------------------------
    "NCCI-BUNDLING": {
        "rule_id": "NCCI-BUNDLING",
        "title": "NCCI — Procedure-to-Procedure Bundling Edits",
        "citation": "CMS NCCI Policy Manual, Chapter 1",
        "summary": (
            "CMS NCCI edits identify pairs of CPT/HCPCS codes that should not "
            "be billed together because one is considered a component of the "
            "other (Column 2 bundled into Column 1). Payment for the Column 1 "
            "code is considered to include payment for the Column 2 code. "
            "Modifier Indicator 0 = modifier cannot override bundling. "
            "Modifier Indicator 1 = a valid NCCI-associated modifier (e.g., -25) "
            "may be appended to indicate the services are distinct and the "
            "Column 2 service should be separately payable."
        ),
        "modifier_indicator": {
            "0": "Bundling cannot be overridden by any modifier.",
            "1": "Modifier -25, -57, -59, -91, XE, XP, XS, or XU may override bundling.",
            "9": "Not applicable.",
        },
    },
    "NCCI-MODIFIER-25": {
        "rule_id": "NCCI-MODIFIER-25",
        "title": "NCCI — Modifier -25 Override",
        "citation": "CMS NCCI Policy Manual, Chapter 3; CPT Modifier -25",
        "summary": (
            "Modifier -25 (Significant, Separately Identifiable Evaluation and "
            "Management Service by the Same Physician on the Same Day of the "
            "Procedure or Other Service) indicates that the E/M service provided "
            "is above and beyond the other service provided. When appended to an "
            "E/M code that is Column 1 in an NCCI edit pair with "
            "modifier_indicator=1, it overrides the bundling edit and allows "
            "separate reimbursement for the Column 2 procedure. The insurer "
            "must process both codes for payment."
        ),
        "applies_to": ["em_codes_with_modifier_indicator_1"],
        "effect": "Overrides NCCI bundling for modifier_indicator=1 pairs.",
    },
    # ------------------------------------------------------------------
    # ACA Appeal Rights — 45 CFR § 147.136
    # ------------------------------------------------------------------
    "ACA-APPEAL": {
        "rule_id": "ACA-APPEAL",
        "title": "ACA — Internal Appeal Rights",
        "citation": "45 CFR § 147.136",
        "summary": (
            "Under the Affordable Care Act, enrollees have the right to appeal "
            "a health plan's denial of a claim or coverage. Plans must allow "
            "internal appeals and, for urgent care, expedited internal appeals. "
            "The enrollee must file an internal appeal within 180 days of "
            "receiving an adverse benefit determination (EOB). For billing "
            "disputes involving calculation errors, the appeal window from "
            "EOB receipt is typically 30 days per plan contract terms."
        ),
        "standard_appeal_deadline_days": 30,
        "urgent_care_deadline_days": 72,
        "applies_to": ["commercial_insurance", "aca_compliant_plans"],
    },
}
