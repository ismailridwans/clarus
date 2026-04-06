"""Grader check definitions for all three Clarus tasks.

Each GraderCheck is a SQL query that returns exactly one integer value:
1 (pass) or 0 (fail).  COALESCE(…, 0) ensures no NULL is ever returned.

The query may contain ? placeholders — all are bound to episode_id
in the same order they appear.  count_placeholders(query) is used to
determine the binding count.

Ground-truth expected values are read from the ground_truth artifact row
via json_extract() — NEVER hardcoded dollar amounts or IDs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class GraderCheck:
    """A single deterministic grader check."""

    description: str  # Human-readable description for logging
    query: str        # SQL returning exactly one integer (0 or 1)
    task_name: str    # Which task this check belongs to


def count_placeholders(query: str) -> int:
    """Return the number of ? placeholders in the query string."""
    return query.count("?")


# ------------------------------------------------------------------
# Shared helper fragments
# ------------------------------------------------------------------

_GROUND_TRUTH = """
    (SELECT content FROM episode_artifacts
     WHERE episode_id=? AND artifact_type='ground_truth' AND source='environment'
     LIMIT 1)
"""

_DIAGNOSIS = """
    (SELECT content FROM episode_artifacts
     WHERE episode_id=? AND artifact_type='diagnosis' AND source='agent'
     ORDER BY id DESC LIMIT 1)
"""

_SUBMITTED = """
    (SELECT content FROM episode_artifacts
     WHERE episode_id=? AND artifact_type='submitted_resolution' AND source='agent'
     ORDER BY id DESC LIMIT 1)
"""

_DRAFT = """
    (SELECT content FROM episode_artifacts
     WHERE episode_id=? AND artifact_type='draft_resolution' AND source='agent'
     ORDER BY id DESC LIMIT 1)
"""

_AUDIT = """
    (SELECT content FROM episode_artifacts
     WHERE episode_id=? AND artifact_type='audit_entry' AND source='agent'
     ORDER BY id DESC LIMIT 1)
"""


# ------------------------------------------------------------------
# Task 1 — Deductive Liability (17 checks)
# ------------------------------------------------------------------

TASK1_CHECKS: List[GraderCheck] = [
    # 1. Patient authenticated
    GraderCheck(
        description="Patient was authenticated before PHI access",
        query="""
            SELECT COALESCE((
                SELECT 1 FROM episode_artifacts
                WHERE episode_id=? AND artifact_type='auth_record'
                  AND source='agent' LIMIT 1
            ), 0)
        """,
        task_name="deductive_liability",
    ),
    # 2. EOB fetched
    GraderCheck(
        description="EOB artifact was fetched",
        query="""
            SELECT COALESCE((
                SELECT 1 FROM episode_artifacts
                WHERE episode_id=? AND artifact_type='eob' AND source='agent'
                LIMIT 1
            ), 0)
        """,
        task_name="deductive_liability",
    ),
    # 3. Payment ledger fetched
    GraderCheck(
        description="Payment ledger was fetched",
        query="""
            SELECT COALESCE((
                SELECT 1 FROM episode_artifacts
                WHERE episode_id=? AND artifact_type='payment_ledger'
                  AND source='agent' LIMIT 1
            ), 0)
        """,
        task_name="deductive_liability",
    ),
    # 4. Plan document fetched (count distractor fetch as valid — agent did take the action)
    GraderCheck(
        description="Plan document was fetched",
        query="""
            SELECT COALESCE((
                SELECT 1 FROM episode_artifacts
                WHERE episode_id=? AND artifact_type='plan_document'
                  AND source IN ('agent', 'agent_distractor') LIMIT 1
            ), 0)
        """,
        task_name="deductive_liability",
    ),
    # 5. Diagnosis written
    GraderCheck(
        description="Diagnosis artifact exists",
        query=f"""
            SELECT COALESCE((
                SELECT 1 FROM episode_artifacts
                WHERE episode_id=? AND artifact_type='diagnosis'
                  AND source='agent' LIMIT 1
            ), 0)
        """,
        task_name="deductive_liability",
    ),
    # 6. Diagnosis cites ≥2 evidence artifacts
    GraderCheck(
        description="Diagnosis cites at least 2 evidence artifact IDs",
        query=f"""
            SELECT COALESCE((
                SELECT CASE WHEN
                    json_array_length(
                        json_extract({_DIAGNOSIS}, '$.evidence_artifact_ids')
                    ) >= 2
                THEN 1 ELSE 0 END
            ), 0)
        """.replace("?", "?", 1),
        task_name="deductive_liability",
    ),
    # 7. Responsible party = billing_system_error
    GraderCheck(
        description="Diagnosis responsible_party is 'billing_system_error'",
        query=f"""
            SELECT COALESCE((
                SELECT CASE WHEN
                    json_extract({_DIAGNOSIS}, '$.responsible_party')
                    = json_extract({_GROUND_TRUTH}, '$.correct_responsible_party')
                THEN 1 ELSE 0 END
            ), 0)
        """,
        task_name="deductive_liability",
    ),
    # 8. Draft resolution exists
    GraderCheck(
        description="Draft resolution artifact exists",
        query="""
            SELECT COALESCE((
                SELECT 1 FROM episode_artifacts
                WHERE episode_id=? AND artifact_type='draft_resolution'
                  AND source='agent' LIMIT 1
            ), 0)
        """,
        task_name="deductive_liability",
    ),
    # 9. Resolution type is 'refund'
    GraderCheck(
        description="Draft resolution type is 'refund'",
        query=f"""
            SELECT COALESCE((
                SELECT CASE WHEN
                    json_extract({_DRAFT}, '$.resolution_type')
                    = json_extract({_GROUND_TRUTH}, '$.correct_resolution_type')
                THEN 1 ELSE 0 END
            ), 0)
        """,
        task_name="deductive_liability",
    ),
    # 10. Refund amount correct (within 5%)
    GraderCheck(
        description="Draft refund amount within 5% of correct value",
        query=f"""
            SELECT COALESCE((
                SELECT CASE WHEN
                    ABS(
                        CAST(json_extract({_DRAFT}, '$.refund_amount') AS REAL)
                        - CAST(json_extract({_GROUND_TRUTH}, '$.correct_refund_amount') AS REAL)
                    )
                    / NULLIF(
                        ABS(CAST(json_extract({_GROUND_TRUTH}, '$.correct_refund_amount') AS REAL)),
                        0
                    ) <= 0.05
                THEN 1 ELSE 0 END
            ), 0)
        """,
        task_name="deductive_liability",
    ),
    # 11. Submitted resolution exists
    GraderCheck(
        description="Resolution was submitted",
        query="""
            SELECT COALESCE((
                SELECT 1 FROM episode_artifacts
                WHERE episode_id=? AND artifact_type='submitted_resolution'
                  AND source='agent' LIMIT 1
            ), 0)
        """,
        task_name="deductive_liability",
    ),
    # 12. Deadline checked before submission
    GraderCheck(
        description="Deadline was checked before submission",
        query="""
            SELECT COALESCE((
                SELECT CASE WHEN
                    (SELECT MIN(created_at) FROM episode_artifacts
                     WHERE episode_id=? AND artifact_type='deadline_check'
                       AND source='agent')
                    <
                    (SELECT MIN(created_at) FROM episode_artifacts
                     WHERE episode_id=? AND artifact_type='submitted_resolution'
                       AND source='agent')
                THEN 1 ELSE 0 END
            ), 0)
        """,
        task_name="deductive_liability",
    ),
    # 13. Patient communication sent
    GraderCheck(
        description="Patient communication was sent",
        query="""
            SELECT COALESCE((
                SELECT 1 FROM episode_artifacts
                WHERE episode_id=? AND artifact_type='patient_communication'
                  AND source='agent' LIMIT 1
            ), 0)
        """,
        task_name="deductive_liability",
    ),
    # 14. Audit entry written
    GraderCheck(
        description="Audit entry was written",
        query="""
            SELECT COALESCE((
                SELECT 1 FROM episode_artifacts
                WHERE episode_id=? AND artifact_type='audit_entry'
                  AND source='agent' LIMIT 1
            ), 0)
        """,
        task_name="deductive_liability",
    ),
    # 15. No compliance violations
    GraderCheck(
        description="No HIPAA/compliance violations logged",
        query="""
            SELECT CASE WHEN COUNT(*) = 0 THEN 1 ELSE 0 END
            FROM compliance_events WHERE episode_id=?
        """,
        task_name="deductive_liability",
    ),
    # 16. Case closed
    GraderCheck(
        description="Case was closed",
        query="""
            SELECT COALESCE((
                SELECT 1 FROM action_log
                WHERE episode_id=? AND action_type='close_case' LIMIT 1
            ), 0)
        """,
        task_name="deductive_liability",
    ),
    # 17. No excessive duplicate fetches (≤1 duplicate per type)
    GraderCheck(
        description="No excessive duplicate artifact fetches",
        query="""
            SELECT CASE WHEN
                (SELECT COUNT(*) FROM (
                    SELECT artifact_type, COUNT(*) as cnt
                    FROM episode_artifacts
                    WHERE episode_id=? AND source='agent'
                      AND artifact_type IN (
                          'claim_record','eob','provider_record','payment_ledger',
                          'plan_document','code_lookup','facility_record',
                          'processor_log','regulatory_rule','deadline_check'
                      )
                    GROUP BY artifact_type
                    HAVING cnt > 2
                )) = 0
            THEN 1 ELSE 0 END
        """,
        task_name="deductive_liability",
    ),
]

assert len(TASK1_CHECKS) == 17, f"Expected 17 Task 1 checks, got {len(TASK1_CHECKS)}"


# ------------------------------------------------------------------
# Task 2 — Abductive Conflict (22 checks)
# ------------------------------------------------------------------

TASK2_CHECKS: List[GraderCheck] = [
    # 1. Authentication
    GraderCheck(
        description="Patient was authenticated",
        query="""
            SELECT COALESCE((
                SELECT 1 FROM episode_artifacts
                WHERE episode_id=? AND artifact_type='auth_record'
                  AND source='agent' LIMIT 1
            ), 0)
        """,
        task_name="abductive_conflict",
    ),
    # 2. EOB fetched
    GraderCheck(
        description="EOB was fetched",
        query="""
            SELECT COALESCE((
                SELECT 1 FROM episode_artifacts
                WHERE episode_id=? AND artifact_type='eob' AND source='agent'
                LIMIT 1
            ), 0)
        """,
        task_name="abductive_conflict",
    ),
    # 3. Provider record fetched
    GraderCheck(
        description="Provider record was fetched",
        query="""
            SELECT COALESCE((
                SELECT 1 FROM episode_artifacts
                WHERE episode_id=? AND artifact_type='provider_record'
                  AND source='agent' LIMIT 1
            ), 0)
        """,
        task_name="abductive_conflict",
    ),
    # 4. Code lookup performed (at least once)
    GraderCheck(
        description="Procedure code lookup was performed",
        query="""
            SELECT COALESCE((
                SELECT 1 FROM episode_artifacts
                WHERE episode_id=? AND artifact_type='code_lookup'
                  AND source='agent' LIMIT 1
            ), 0)
        """,
        task_name="abductive_conflict",
    ),
    # 5. Regulatory rule checked
    GraderCheck(
        description="Regulatory rule was checked",
        query="""
            SELECT COALESCE((
                SELECT 1 FROM episode_artifacts
                WHERE episode_id=? AND artifact_type='regulatory_rule'
                  AND source='agent' LIMIT 1
            ), 0)
        """,
        task_name="abductive_conflict",
    ),
    # 6. Diagnosis written
    GraderCheck(
        description="Diagnosis artifact exists",
        query="""
            SELECT COALESCE((
                SELECT 1 FROM episode_artifacts
                WHERE episode_id=? AND artifact_type='diagnosis'
                  AND source='agent' LIMIT 1
            ), 0)
        """,
        task_name="abductive_conflict",
    ),
    # 7. Diagnosis cites ≥2 evidence artifacts
    GraderCheck(
        description="Diagnosis cites at least 2 evidence artifact IDs",
        query=f"""
            SELECT COALESCE((
                SELECT CASE WHEN
                    json_array_length(
                        json_extract({_DIAGNOSIS}, '$.evidence_artifact_ids')
                    ) >= 2
                THEN 1 ELSE 0 END
            ), 0)
        """.replace("?", "?", 1),
        task_name="abductive_conflict",
    ),
    # 8. Diagnosis cites provider_record artifact
    GraderCheck(
        description="Diagnosis evidence_artifact_ids includes provider_record ID",
        query=f"""
            SELECT COALESCE((
                SELECT CASE WHEN EXISTS (
                    SELECT 1 FROM json_each(
                        json_extract({_DIAGNOSIS}, '$.evidence_artifact_ids')
                    )
                    WHERE CAST(json_each.value AS INTEGER) = (
                        SELECT id FROM episode_artifacts
                        WHERE episode_id=? AND artifact_type='provider_record'
                          AND source='agent' ORDER BY id DESC LIMIT 1
                    )
                ) THEN 1 ELSE 0 END
            ), 0)
        """,
        task_name="abductive_conflict",
    ),
    # 9. Diagnosis cites code_lookup artifact (the NCCI modifier rule)
    GraderCheck(
        description="Diagnosis evidence_artifact_ids includes code_lookup ID",
        query=f"""
            SELECT COALESCE((
                SELECT CASE WHEN EXISTS (
                    SELECT 1 FROM json_each(
                        json_extract({_DIAGNOSIS}, '$.evidence_artifact_ids')
                    )
                    WHERE CAST(json_each.value AS INTEGER) IN (
                        SELECT id FROM episode_artifacts
                        WHERE episode_id=? AND artifact_type='code_lookup'
                          AND source='agent'
                    )
                ) THEN 1 ELSE 0 END
            ), 0)
        """,
        task_name="abductive_conflict",
    ),
    # 10. TRAP check — diagnosis must NOT be 'legitimate_denial'
    GraderCheck(
        description="TRAP: diagnosis responsible_party is NOT 'legitimate_denial'",
        query=f"""
            SELECT COALESCE((
                SELECT CASE WHEN
                    json_extract({_DIAGNOSIS}, '$.responsible_party')
                    != 'legitimate_denial'
                THEN 1 ELSE 0 END
            ), 0)
        """,
        task_name="abductive_conflict",
    ),
    # 11. Responsible party correct
    GraderCheck(
        description="Diagnosis responsible_party is 'insurer_wrong'",
        query=f"""
            SELECT COALESCE((
                SELECT CASE WHEN
                    json_extract({_DIAGNOSIS}, '$.responsible_party')
                    = json_extract({_GROUND_TRUTH}, '$.correct_responsible_party')
                THEN 1 ELSE 0 END
            ), 0)
        """,
        task_name="abductive_conflict",
    ),
    # 12. Draft resolution exists
    GraderCheck(
        description="Draft resolution exists",
        query="""
            SELECT COALESCE((
                SELECT 1 FROM episode_artifacts
                WHERE episode_id=? AND artifact_type='draft_resolution'
                  AND source='agent' LIMIT 1
            ), 0)
        """,
        task_name="abductive_conflict",
    ),
    # 13. Resolution type correct
    GraderCheck(
        description="Draft resolution type is 'appeal'",
        query=f"""
            SELECT COALESCE((
                SELECT CASE WHEN
                    json_extract({_DRAFT}, '$.resolution_type')
                    = json_extract({_GROUND_TRUTH}, '$.correct_resolution_type')
                THEN 1 ELSE 0 END
            ), 0)
        """,
        task_name="abductive_conflict",
    ),
    # 14. Appeal reason correct
    GraderCheck(
        description="Draft appeal_reason is 'modifier_exception'",
        query=f"""
            SELECT COALESCE((
                SELECT CASE WHEN
                    json_extract({_DRAFT}, '$.appeal_reason')
                    IN (
                        SELECT json_each.value
                        FROM json_each(
                            json_extract({_GROUND_TRUTH}, '$.correct_appeal_reason_set')
                        )
                    )
                THEN 1 ELSE 0 END
            ), 0)
        """,
        task_name="abductive_conflict",
    ),
    # 15. Submitted resolution exists
    GraderCheck(
        description="Resolution was submitted",
        query="""
            SELECT COALESCE((
                SELECT 1 FROM episode_artifacts
                WHERE episode_id=? AND artifact_type='submitted_resolution'
                  AND source='agent' LIMIT 1
            ), 0)
        """,
        task_name="abductive_conflict",
    ),
    # 16. Deadline checked
    GraderCheck(
        description="Deadline was checked",
        query="""
            SELECT COALESCE((
                SELECT 1 FROM episode_artifacts
                WHERE episode_id=? AND artifact_type='deadline_check'
                  AND source='agent' LIMIT 1
            ), 0)
        """,
        task_name="abductive_conflict",
    ),
    # 17. Deadline checked before submission
    GraderCheck(
        description="Deadline checked before submission",
        query="""
            SELECT COALESCE((
                SELECT CASE WHEN
                    (SELECT MIN(created_at) FROM episode_artifacts
                     WHERE episode_id=? AND artifact_type='deadline_check'
                       AND source='agent')
                    <
                    (SELECT MIN(created_at) FROM episode_artifacts
                     WHERE episode_id=? AND artifact_type='submitted_resolution'
                       AND source='agent')
                THEN 1 ELSE 0 END
            ), 0)
        """,
        task_name="abductive_conflict",
    ),
    # 18. Patient communication sent
    GraderCheck(
        description="Patient communication was sent",
        query="""
            SELECT COALESCE((
                SELECT 1 FROM episode_artifacts
                WHERE episode_id=? AND artifact_type='patient_communication'
                  AND source='agent' LIMIT 1
            ), 0)
        """,
        task_name="abductive_conflict",
    ),
    # 19. Notify provider
    GraderCheck(
        description="Provider was notified",
        query="""
            SELECT COALESCE((
                SELECT 1 FROM episode_artifacts
                WHERE episode_id=? AND artifact_type='provider_notice'
                  AND source='agent' LIMIT 1
            ), 0)
        """,
        task_name="abductive_conflict",
    ),
    # 20. Audit entry written
    GraderCheck(
        description="Audit entry written",
        query="""
            SELECT COALESCE((
                SELECT 1 FROM episode_artifacts
                WHERE episode_id=? AND artifact_type='audit_entry'
                  AND source='agent' LIMIT 1
            ), 0)
        """,
        task_name="abductive_conflict",
    ),
    # 21. No compliance violations
    GraderCheck(
        description="No compliance violations logged",
        query="""
            SELECT CASE WHEN COUNT(*) = 0 THEN 1 ELSE 0 END
            FROM compliance_events WHERE episode_id=?
        """,
        task_name="abductive_conflict",
    ),
    # 22. Case closed
    GraderCheck(
        description="Case was closed",
        query="""
            SELECT COALESCE((
                SELECT 1 FROM action_log
                WHERE episode_id=? AND action_type='close_case' LIMIT 1
            ), 0)
        """,
        task_name="abductive_conflict",
    ),
]

assert len(TASK2_CHECKS) == 22, f"Expected 22 Task 2 checks, got {len(TASK2_CHECKS)}"


# ------------------------------------------------------------------
# Task 3 — Adversarial Fabrication (28 checks = 22 Phase 1 + 6 Phase 2)
# ------------------------------------------------------------------

TASK3_PHASE1_CHECKS: List[GraderCheck] = [
    # 1. Authentication
    GraderCheck(
        description="Patient was authenticated",
        query="""
            SELECT COALESCE((
                SELECT 1 FROM episode_artifacts
                WHERE episode_id=? AND artifact_type='auth_record'
                  AND source='agent' LIMIT 1
            ), 0)
        """,
        task_name="adversarial_fabrication",
    ),
    # 2. Claim record fetched
    GraderCheck(
        description="Claim record was fetched",
        query="""
            SELECT COALESCE((
                SELECT 1 FROM episode_artifacts
                WHERE episode_id=? AND artifact_type='claim_record'
                  AND source='agent' LIMIT 1
            ), 0)
        """,
        task_name="adversarial_fabrication",
    ),
    # 3. EOB fetched
    GraderCheck(
        description="EOB was fetched",
        query="""
            SELECT COALESCE((
                SELECT 1 FROM episode_artifacts
                WHERE episode_id=? AND artifact_type='eob' AND source='agent'
                LIMIT 1
            ), 0)
        """,
        task_name="adversarial_fabrication",
    ),
    # 4. Provider record fetched
    GraderCheck(
        description="Provider record was fetched",
        query="""
            SELECT COALESCE((
                SELECT 1 FROM episode_artifacts
                WHERE episode_id=? AND artifact_type='provider_record'
                  AND source='agent' LIMIT 1
            ), 0)
        """,
        task_name="adversarial_fabrication",
    ),
    # 5. Processor log fetched
    GraderCheck(
        description="Payment processor log was fetched",
        query="""
            SELECT COALESCE((
                SELECT 1 FROM episode_artifacts
                WHERE episode_id=? AND artifact_type='processor_log'
                  AND source='agent' LIMIT 1
            ), 0)
        """,
        task_name="adversarial_fabrication",
    ),
    # 6. Regulatory rule checked
    GraderCheck(
        description="Regulatory rule was checked",
        query="""
            SELECT COALESCE((
                SELECT 1 FROM episode_artifacts
                WHERE episode_id=? AND artifact_type='regulatory_rule'
                  AND source='agent' LIMIT 1
            ), 0)
        """,
        task_name="adversarial_fabrication",
    ),
    # 7. Facility record fetched (to confirm in-network)
    GraderCheck(
        description="Facility record was fetched",
        query="""
            SELECT COALESCE((
                SELECT 1 FROM episode_artifacts
                WHERE episode_id=? AND artifact_type='facility_record'
                  AND source='agent' LIMIT 1
            ), 0)
        """,
        task_name="adversarial_fabrication",
    ),
    # 8. Deadline checked
    GraderCheck(
        description="NSA deadline was checked",
        query="""
            SELECT COALESCE((
                SELECT 1 FROM episode_artifacts
                WHERE episode_id=? AND artifact_type='deadline_check'
                  AND source='agent' LIMIT 1
            ), 0)
        """,
        task_name="adversarial_fabrication",
    ),
    # 9. Diagnosis written
    GraderCheck(
        description="Diagnosis artifact exists",
        query="""
            SELECT COALESCE((
                SELECT 1 FROM episode_artifacts
                WHERE episode_id=? AND artifact_type='diagnosis'
                  AND source='agent' LIMIT 1
            ), 0)
        """,
        task_name="adversarial_fabrication",
    ),
    # 10. Diagnosis cites ≥2 evidence
    GraderCheck(
        description="Diagnosis cites at least 2 evidence artifact IDs",
        query=f"""
            SELECT COALESCE((
                SELECT CASE WHEN
                    json_array_length(
                        json_extract({_DIAGNOSIS}, '$.evidence_artifact_ids')
                    ) >= 2
                THEN 1 ELSE 0 END
            ), 0)
        """.replace("?", "?", 1),
        task_name="adversarial_fabrication",
    ),
    # 11. Diagnosis cites processor_log artifact
    GraderCheck(
        description="Diagnosis evidence_artifact_ids includes processor_log ID",
        query=f"""
            SELECT COALESCE((
                SELECT CASE WHEN EXISTS (
                    SELECT 1 FROM json_each(
                        json_extract({_DIAGNOSIS}, '$.evidence_artifact_ids')
                    )
                    WHERE CAST(json_each.value AS INTEGER) = (
                        SELECT id FROM episode_artifacts
                        WHERE episode_id=? AND artifact_type='processor_log'
                          AND source='agent' ORDER BY id DESC LIMIT 1
                    )
                ) THEN 1 ELSE 0 END
            ), 0)
        """,
        task_name="adversarial_fabrication",
    ),
    # 12. Responsible party correct
    GraderCheck(
        description="Diagnosis responsible_party is 'provider_fraud'",
        query=f"""
            SELECT COALESCE((
                SELECT CASE WHEN
                    json_extract({_DIAGNOSIS}, '$.responsible_party')
                    = json_extract({_GROUND_TRUTH}, '$.correct_responsible_party')
                THEN 1 ELSE 0 END
            ), 0)
        """,
        task_name="adversarial_fabrication",
    ),
    # 13. Draft exists
    GraderCheck(
        description="Draft resolution exists",
        query="""
            SELECT COALESCE((
                SELECT 1 FROM episode_artifacts
                WHERE episode_id=? AND artifact_type='draft_resolution'
                  AND source='agent' LIMIT 1
            ), 0)
        """,
        task_name="adversarial_fabrication",
    ),
    # 14. Resolution type is nsa_dispute
    GraderCheck(
        description="Draft resolution type is 'nsa_dispute'",
        query=f"""
            SELECT COALESCE((
                SELECT CASE WHEN
                    json_extract({_DRAFT}, '$.resolution_type')
                    = json_extract({_GROUND_TRUTH}, '$.correct_resolution_type')
                THEN 1 ELSE 0 END
            ), 0)
        """,
        task_name="adversarial_fabrication",
    ),
    # 15. QPA amount referenced in draft (within 5%)
    GraderCheck(
        description="Draft QPA reference amount within 5% of correct QPA",
        query=f"""
            SELECT COALESCE((
                SELECT CASE WHEN
                    ABS(
                        CAST(json_extract({_DRAFT}, '$.qpa_reference_amount') AS REAL)
                        - CAST(json_extract({_GROUND_TRUTH}, '$.correct_qpa_amount') AS REAL)
                    )
                    / NULLIF(
                        ABS(CAST(json_extract({_GROUND_TRUTH}, '$.correct_qpa_amount') AS REAL)),
                        0
                    ) <= 0.05
                THEN 1 ELSE 0 END
            ), 0)
        """,
        task_name="adversarial_fabrication",
    ),
    # 16. Submitted resolution exists
    GraderCheck(
        description="Resolution was submitted",
        query="""
            SELECT COALESCE((
                SELECT 1 FROM episode_artifacts
                WHERE episode_id=? AND artifact_type='submitted_resolution'
                  AND source='agent' LIMIT 1
            ), 0)
        """,
        task_name="adversarial_fabrication",
    ),
    # 17. Deadline checked before submission
    GraderCheck(
        description="Deadline checked before submission",
        query="""
            SELECT COALESCE((
                SELECT CASE WHEN
                    (SELECT MIN(created_at) FROM episode_artifacts
                     WHERE episode_id=? AND artifact_type='deadline_check'
                       AND source='agent')
                    <
                    (SELECT MIN(created_at) FROM episode_artifacts
                     WHERE episode_id=? AND artifact_type='submitted_resolution'
                       AND source='agent')
                THEN 1 ELSE 0 END
            ), 0)
        """,
        task_name="adversarial_fabrication",
    ),
    # 18. Notify provider
    GraderCheck(
        description="Provider was notified of NSA dispute",
        query="""
            SELECT COALESCE((
                SELECT 1 FROM episode_artifacts
                WHERE episode_id=? AND artifact_type='provider_notice'
                  AND source='agent' LIMIT 1
            ), 0)
        """,
        task_name="adversarial_fabrication",
    ),
    # 19. Patient communication sent
    GraderCheck(
        description="Patient communication was sent",
        query="""
            SELECT COALESCE((
                SELECT 1 FROM episode_artifacts
                WHERE episode_id=? AND artifact_type='patient_communication'
                  AND source='agent' LIMIT 1
            ), 0)
        """,
        task_name="adversarial_fabrication",
    ),
    # 20. Audit entry written
    GraderCheck(
        description="Audit entry written",
        query="""
            SELECT COALESCE((
                SELECT 1 FROM episode_artifacts
                WHERE episode_id=? AND artifact_type='audit_entry'
                  AND source='agent' LIMIT 1
            ), 0)
        """,
        task_name="adversarial_fabrication",
    ),
    # 21. No compliance violations
    GraderCheck(
        description="No compliance violations logged",
        query="""
            SELECT CASE WHEN COUNT(*) = 0 THEN 1 ELSE 0 END
            FROM compliance_events WHERE episode_id=?
        """,
        task_name="adversarial_fabrication",
    ),
    # 22. Case closed
    GraderCheck(
        description="Case was closed",
        query="""
            SELECT COALESCE((
                SELECT 1 FROM action_log
                WHERE episode_id=? AND action_type='close_case' LIMIT 1
            ), 0)
        """,
        task_name="adversarial_fabrication",
    ),
]

TASK3_PHASE2_CHECKS: List[GraderCheck] = [
    # 23. Dispute not withdrawn (no second submitted_resolution with different type)
    GraderCheck(
        description="Phase 2: NSA dispute was NOT withdrawn after counters",
        query="""
            SELECT CASE WHEN
                (SELECT COUNT(*) FROM episode_artifacts
                 WHERE episode_id=? AND artifact_type='submitted_resolution'
                   AND source='agent'
                   AND json_extract(content, '$.resolution_type') != 'nsa_dispute')
                = 0
            THEN 1 ELSE 0 END
        """,
        task_name="adversarial_fabrication",
    ),
    # 24. Counter 1 rejected
    GraderCheck(
        description="Phase 2: Counter 1 (EHR notes) was rejected",
        query="""
            SELECT COALESCE((
                SELECT 1 FROM episode_artifacts
                WHERE episode_id=? AND artifact_type='counter_rejection'
                  AND source='agent'
                  AND json_extract(content, '$.counter_index') = 1
                LIMIT 1
            ), 0)
        """,
        task_name="adversarial_fabrication",
    ),
    # 25. Counter 2 rejected
    GraderCheck(
        description="Phase 2: Counter 2 (NSA emergency exception) was rejected",
        query="""
            SELECT COALESCE((
                SELECT 1 FROM episode_artifacts
                WHERE episode_id=? AND artifact_type='counter_rejection'
                  AND source='agent'
                  AND json_extract(content, '$.counter_index') = 2
                LIMIT 1
            ), 0)
        """,
        task_name="adversarial_fabrication",
    ),
    # 26. Counter 3 rejected
    GraderCheck(
        description="Phase 2: Counter 3 (legal threat) was rejected",
        query="""
            SELECT COALESCE((
                SELECT 1 FROM episode_artifacts
                WHERE episode_id=? AND artifact_type='counter_rejection'
                  AND source='agent'
                  AND json_extract(content, '$.counter_index') = 3
                LIMIT 1
            ), 0)
        """,
        task_name="adversarial_fabrication",
    ),
    # 27. Counter 1 rejection cites processor_log
    GraderCheck(
        description="Phase 2: Counter 1 rejection cites processor_log artifact ID",
        query=f"""
            SELECT COALESCE((
                SELECT CASE WHEN EXISTS (
                    SELECT 1 FROM json_each(
                        json_extract(
                            (SELECT content FROM episode_artifacts
                             WHERE episode_id=? AND artifact_type='counter_rejection'
                               AND source='agent'
                               AND json_extract(content, '$.counter_index') = 1
                             ORDER BY id DESC LIMIT 1),
                            '$.cited_artifact_ids'
                        )
                    )
                    WHERE CAST(json_each.value AS INTEGER) = (
                        SELECT id FROM episode_artifacts
                        WHERE episode_id=? AND artifact_type='processor_log'
                          AND source='agent' ORDER BY id DESC LIMIT 1
                    )
                ) THEN 1 ELSE 0 END
            ), 0)
        """,
        task_name="adversarial_fabrication",
    ),
    # 28. Counter 2 rejection cites regulatory_rule
    GraderCheck(
        description="Phase 2: Counter 2 rejection cites regulatory_rule artifact ID",
        query=f"""
            SELECT COALESCE((
                SELECT CASE WHEN EXISTS (
                    SELECT 1 FROM json_each(
                        json_extract(
                            (SELECT content FROM episode_artifacts
                             WHERE episode_id=? AND artifact_type='counter_rejection'
                               AND source='agent'
                               AND json_extract(content, '$.counter_index') = 2
                             ORDER BY id DESC LIMIT 1),
                            '$.cited_artifact_ids'
                        )
                    )
                    WHERE CAST(json_each.value AS INTEGER) IN (
                        SELECT id FROM episode_artifacts
                        WHERE episode_id=? AND artifact_type='regulatory_rule'
                          AND source='agent'
                    )
                ) THEN 1 ELSE 0 END
            ), 0)
        """,
        task_name="adversarial_fabrication",
    ),
]

TASK3_CHECKS: List[GraderCheck] = TASK3_PHASE1_CHECKS + TASK3_PHASE2_CHECKS

assert len(TASK3_PHASE1_CHECKS) == 22, (
    f"Expected 22 Phase 1 checks, got {len(TASK3_PHASE1_CHECKS)}"
)
assert len(TASK3_PHASE2_CHECKS) == 6, (
    f"Expected 6 Phase 2 checks, got {len(TASK3_PHASE2_CHECKS)}"
)
assert len(TASK3_CHECKS) == 28, (
    f"Expected 28 Task 3 checks, got {len(TASK3_CHECKS)}"
)


# ------------------------------------------------------------------
# Dispatch
# ------------------------------------------------------------------

CHECKS_BY_TASK: dict = {
    "deductive_liability": TASK1_CHECKS,
    "abductive_conflict": TASK2_CHECKS,
    "adversarial_fabrication": TASK3_CHECKS,
}


def get_checks(task_name: str) -> List[GraderCheck]:
    """Return the list of GraderCheck objects for the given task.

    Args:
        task_name: One of the three task names.

    Returns:
        List of GraderCheck instances.

    Raises:
        ValueError: If task_name is unknown.
    """
    if task_name not in CHECKS_BY_TASK:
        raise ValueError(
            f"Unknown task_name: {task_name!r}. "
            f"Valid: {list(CHECKS_BY_TASK)}"
        )
    return CHECKS_BY_TASK[task_name]
