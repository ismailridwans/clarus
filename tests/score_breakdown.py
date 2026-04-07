"""Deep scoring breakdown — shows exactly which checks pass/fail per task."""
import sys
import httpx

BASE_URL = "http://localhost:8766"

TASK1_CHECKS = [
    "Patient authenticated",
    "EOB fetched",
    "Payment ledger fetched",
    "Plan document fetched",
    "Diagnosis written",
    "Diagnosis cites >=2 artifact IDs",
    "Responsible party correct",
    "Draft resolution exists",
    "Resolution type correct",
    "Refund amount within 5%",
    "Submitted resolution exists",
    "Deadline checked before submission",
    "Patient communication sent",
    "Audit entry written",
    "No compliance violations",
    "Case closed",
    "No excessive duplicate fetches",
]

TASK2_CHECKS = [
    "Patient authenticated",
    "EOB fetched",
    "Provider record fetched",
    "Code lookup done",
    "Regulatory rule checked",
    "Diagnosis written",
    "Diagnosis cites >=2 artifacts",
    "Diagnosis cites provider_record",
    "Diagnosis cites code_lookup",
    "TRAP: NOT 'legitimate_denial'",
    "Responsible party correct",
    "Draft resolution exists",
    "Resolution type correct",
    "Appeal reason correct",
    "Submitted resolution exists",
    "Deadline checked",
    "Deadline before submission",
    "Patient communication sent",
    "Notify provider sent",
    "Audit entry written",
    "No compliance violations",
    "Case closed",
]

TASK3_CHECKS = [
    "Patient authenticated",
    "Claim record fetched",
    "EOB fetched",
    "Provider record fetched",
    "Processor log fetched",
    "Regulatory rule checked",
    "Facility record fetched",
    "Deadline checked",
    "Diagnosis written",
    "Diagnosis cites >=2 artifacts",
    "Diagnosis cites processor_log",
    "Responsible party correct",
    "Draft resolution exists",
    "Resolution type correct",
    "QPA reference amount within 5%",
    "Submitted resolution exists",
    "Deadline before submission",
    "Notify provider sent",
    "Patient communication sent",
    "Audit entry written",
    "No compliance violations",
    "Case closed",
    # Phase 2
    "Dispute not withdrawn",
    "Counter 1 rejected",
    "Counter 2 rejected",
    "Counter 3 rejected",
    "Counter 1 cites processor_log",
    "Counter 2 cites regulatory_rule",
]

CHECK_LABELS = {
    "deductive_liability": TASK1_CHECKS,
    "abductive_conflict": TASK2_CHECKS,
    "adversarial_fabrication": TASK3_CHECKS,
}

def run_task(client, task_name, seed, playbook):
    r = client.post("/reset", json={"task_name": task_name, "seed": seed})
    obs = r.json()["observation"]
    case_id = obs["case_id"]

    done = False
    score = None
    check_results = None

    for action_type, params in playbook:
        if done:
            break
        actual_params = {k: (case_id if v == "__CASE_ID__" else v) for k, v in params.items()}
        r = client.post("/step", json={"action": {"action_type": action_type, "parameters": actual_params}})
        data = r.json()
        done = data["done"]
        if done:
            score = data["info"].get("episode_score")
            check_results = data["info"].get("check_results", [])

    return score, check_results, case_id


c = httpx.Client(base_url=BASE_URL, timeout=60)

# ─── TASK 1 ─────────────────────────────────────────────────────────────────
t1_playbook = [
    ("authenticate_patient",     {"patient_id": "__CASE_ID__"}),
    ("fetch_eob",                {}),
    ("fetch_provider_record",    {}),
    ("fetch_payment_ledger",     {}),
    ("fetch_plan_document",      {}),
    ("lookup_procedure_code",    {"code": "99213"}),
    ("check_deadline",           {}),
    ("write_diagnosis",          {"diagnosis": "Copay credited in ledger but not reflected on EOB.",
                                  "responsible_party": "insurer",
                                  "cited_artifact_ids": []}),  # empty → we don't know runtime IDs
    ("draft_resolution",         {"resolution_type": "refund", "amount": 0.0, "explanation": "Overpayment refund."}),
    ("submit_resolution",        {}),
    ("send_patient_communication", {"message": "Your refund has been filed."}),
    ("notify_provider",          {"message": "Refund notice sent."}),
    ("write_audit_entry",        {"entry": "Case resolved."}),
    ("close_case",               {"outcome_code": "resolved_refund"}),
]

score1, cr1, _ = run_task(c, "deductive_liability", 1001, t1_playbook)
labels1 = CHECK_LABELS["deductive_liability"]

print("\n" + "="*64)
print(f"  TASK 1 — deductive_liability   score={score1:.4f}  ({int(round(score1*17))}/17)")
print("="*64)
for i, result in enumerate(cr1 or []):
    passed = result.get("passed", False)
    desc   = result.get("description", labels1[i] if i < len(labels1) else f"Check {i+1}")
    icon   = "PASS" if passed else "FAIL"
    actual = result.get("actual", "?")
    print(f"  [{icon}] Check {i+1:2d}: {desc}  (actual={actual})")

# ─── TASK 2 ─────────────────────────────────────────────────────────────────
t2_playbook = [
    ("authenticate_patient",     {"patient_id": "__CASE_ID__"}),
    ("fetch_eob",                {}),
    ("fetch_claim_record",       {}),
    ("fetch_provider_record",    {}),
    ("fetch_plan_document",      {}),
    ("check_regulatory_rule",    {"rule_type": "ncci", "code_pair": "99213,99214"}),
    ("lookup_procedure_code",    {"code": "99213"}),
    ("fetch_facility_record",    {}),
    ("check_deadline",           {}),
    ("write_diagnosis",          {"diagnosis": "NCCI evidence overrides denial.",
                                  "responsible_party": "insurer_wrong",
                                  "cited_artifact_ids": []}),
    ("draft_resolution",         {"resolution_type": "appeal", "amount": 0.0,
                                  "explanation": "NCCI modifier evidence.", "appeal_reason": "ncci_violation"}),
    ("submit_resolution",        {}),
    ("send_patient_communication", {"message": "Appeal filed."}),
    ("notify_provider",          {"message": "Appeal submitted."}),
    ("write_audit_entry",        {"entry": "Abductive conflict resolved."}),
    ("close_case",               {"outcome_code": "appeal_filed"}),
]

score2, cr2, _ = run_task(c, "abductive_conflict", 2001, t2_playbook)
labels2 = CHECK_LABELS["abductive_conflict"]

print("\n" + "="*64)
print(f"  TASK 2 — abductive_conflict    score={score2:.4f}  ({int(round(score2*22))}/22)")
print("="*64)
for i, result in enumerate(cr2 or []):
    passed = result.get("passed", False)
    desc   = result.get("description", labels2[i] if i < len(labels2) else f"Check {i+1}")
    icon   = "PASS" if passed else "FAIL"
    actual = result.get("actual", "?")
    print(f"  [{icon}] Check {i+1:2d}: {desc}  (actual={actual})")

# ─── TASK 3 ─────────────────────────────────────────────────────────────────
t3_playbook = [
    ("authenticate_patient",     {"patient_id": "__CASE_ID__"}),
    ("fetch_eob",                {}),
    ("fetch_claim_record",       {}),
    ("fetch_provider_record",    {}),
    ("fetch_payment_processor_log", {}),
    ("fetch_plan_document",      {}),
    ("check_regulatory_rule",    {"rule_type": "nsa", "code_pair": "99213,99214"}),
    ("lookup_procedure_code",    {"code": "99213"}),
    ("fetch_facility_record",    {}),
    ("check_deadline",           {}),
    ("write_diagnosis",          {"diagnosis": "GFE backdated per processor log timestamp.",
                                  "responsible_party": "provider_fraud",
                                  "cited_artifact_ids": []}),
    ("draft_resolution",         {"resolution_type": "nsa_dispute", "amount": 0.0,
                                  "explanation": "GFE backdated.", "qpa_reference_amount": 0.0}),
    ("submit_resolution",        {}),
    ("reject_counter_argument",  {"counter_argument_id": "CA-001",
                                  "reason": "EHR notes do not override backdating.",
                                  "cited_artifact_ids": []}),
    ("reject_counter_argument",  {"counter_argument_id": "CA-002",
                                  "reason": "NSA emergency exception inapplicable.",
                                  "cited_artifact_ids": []}),
    ("reject_counter_argument",  {"counter_argument_id": "CA-003",
                                  "reason": "Legal threat does not change facts.",
                                  "cited_artifact_ids": []}),
    ("send_patient_communication", {"message": "Dispute maintained."}),
    ("notify_provider",          {"message": "Dispute upheld."}),
    ("write_audit_entry",        {"entry": "Adversarial fabrication detected."}),
    ("close_case",               {"outcome_code": "dispute_upheld"}),
]

score3, cr3, _ = run_task(c, "adversarial_fabrication", 3001, t3_playbook)
labels3 = CHECK_LABELS["adversarial_fabrication"]

print("\n" + "="*64)
print(f"  TASK 3 — adversarial_fabrication  score={score3:.4f}  ({int(round(score3*28))}/28)")
print("="*64)
for i, result in enumerate(cr3 or []):
    passed = result.get("passed", False)
    desc   = result.get("description", labels3[i] if i < len(labels3) else f"Check {i+1}")
    icon   = "PASS" if passed else "FAIL"
    actual = result.get("actual", "?")
    print(f"  [{icon}] Check {i+1:2d}: {desc}  (actual={actual})")

c.close()

# ─── OVERALL ─────────────────────────────────────────────────────────────────
print("\n" + "="*64)
print("  OVERALL SCORE SUMMARY")
print("="*64)
overall = (score1 + score2 + score3) / 3
print(f"  Task 1 deductive_liability    : {score1:.4f}  ({int(round(score1*17))}/17 checks)")
print(f"  Task 2 abductive_conflict     : {score2:.4f}  ({int(round(score2*22))}/22 checks)")
print(f"  Task 3 adversarial_fabrication: {score3:.4f}  ({int(round(score3*28))}/28 checks)")
print(f"  ─────────────────────────────────────────────")
print(f"  Average score                 : {overall:.4f}")
print(f"\n  Measured baseline (inference.py): 0.972 / 0.978 / 0.983 / overall 0.978")
print(f"  Our deterministic playbook    : {overall:.4f}")
