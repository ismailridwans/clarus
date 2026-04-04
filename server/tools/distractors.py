"""Distractor artifact payloads for Clarus.

Distractors are misleading artifacts that waste the agent's API budget
without contributing to grader score.  Each distractor looks superficially
relevant but contains explicit signals that it does not apply to the case.

The structural reward for fetching a distractor is -0.01.  Grader checks
are not satisfied by distractor fetches.
"""

from __future__ import annotations

from typing import Optional


# Maps artifact_type → distractor payload content dict.
# These are returned by get_distractor_payload() when the agent fetches
# a resource that is the episode's designated distractor type.
DISTRACTOR_PAYLOADS: dict = {
    "regulatory_rule": {
        "rule_id": "MEDICAID-ONLY",
        "title": "Medicaid Billing Rule (Not Applicable)",
        "citation": "42 CFR § 447.45",
        "summary": (
            "This rule governs Medicaid fee-for-service billing timelines. "
            "It applies to Medicaid only — not to commercial ACA plans. "
            "Not relevant to this case."
        ),
        "note": "Applies to Medicaid only. Not relevant here.",
    },
    "code_lookup": {
        "code": "RETIRED-CODE",
        "description": "Retired procedure code — invalid since 2021.",
        "status": "INACTIVE",
        "ncci_bundled_with": [],
        "ncci_modifier_exceptions": [],
        "note": "Code retired 2021. Not relevant to this case.",
        "source": "CMS_CPT_ARCHIVE",
    },
    "payment_ledger": {
        "payments": [],
        "total_paid": 0.0,
        "note": "No payments found — different account number. Not relevant.",
    },
    "plan_document": {
        "plan_id": "PLAN-WRONG",
        "section": "vision_benefits",
        "content": (
            "Vision care benefits: $150 annual allowance for frames and lenses. "
            "This section covers vision services only."
        ),
        "note": "Vision benefits section. Not relevant to medical billing dispute.",
    },
    "facility_record": {
        "facility_id": "FAC-WRONG",
        "facility_name": "Wrong Facility Medical Center",
        "network_status": "OUT_OF_NETWORK",
        "npi": "0000000000",
        "note": "Wrong facility — not the treating facility for this claim.",
    },
}


def is_distractor(
    artifact_type: str,
    distractor_artifact_type: Optional[str],
) -> bool:
    """Return True if this fetch should return a distractor payload.

    Args:
        artifact_type: The artifact type being fetched.
        distractor_artifact_type: The episode's designated distractor type,
            or None if the episode has no distractor.

    Returns:
        True if the fetch should return distractor content.
    """
    if distractor_artifact_type is None:
        return False
    return artifact_type == distractor_artifact_type


def get_distractor_payload(artifact_type: str) -> dict:
    """Return the distractor payload for the given artifact type.

    Args:
        artifact_type: One of the distractor-capable artifact types.

    Returns:
        Dict payload to include in the artifact content.

    Raises:
        KeyError: If no distractor is defined for the given artifact type.
    """
    if artifact_type not in DISTRACTOR_PAYLOADS:
        raise KeyError(
            f"No distractor payload defined for artifact_type={artifact_type!r}. "
            f"Valid types: {list(DISTRACTOR_PAYLOADS)}"
        )
    return DISTRACTOR_PAYLOADS[artifact_type].copy()
