"""Comprehensive live feature check — tests all endpoints, action types, tasks."""
import requests, json, sys

BASE = "http://127.0.0.1:8765"
errors = []
ok_count = 0


def s(label, cond, detail=""):
    global ok_count
    if cond:
        print(f"  [OK]   {label}")
        ok_count += 1
    else:
        msg = f"  [FAIL] {label}" + (f" -- {detail}" if detail else "")
        print(msg)
        errors.append(msg)


def step(action):
    r = requests.post(f"{BASE}/step", json={"action": action})
    r.raise_for_status()
    return r.json()


def reset(task=None, seed=None):
    body = {}
    if task:
        body["task_name"] = task
    if seed:
        body["seed"] = seed
    r = requests.post(f"{BASE}/reset", json=body)
    r.raise_for_status()
    return r.json()


# ──────────────────────────────────────────────────────────────────────────────
print("\n[1] STATIC ENDPOINTS")
# ──────────────────────────────────────────────────────────────────────────────
r = requests.get(f"{BASE}/health").json()
s("GET /health returns status=healthy", r.get("status") == "healthy")

r = requests.get(f"{BASE}/metadata").json()
s("GET /metadata has 3 tasks", len(r.get("tasks", [])) == 3)
s("GET /metadata version=1.0.0", r.get("version") == "1.0.0")
s("GET /metadata hf_space present", "hf_space" in r)

r = requests.get(f"{BASE}/schema").json()
s("GET /schema has action schema", "action" in r)
s("GET /schema has observation schema", "observation" in r)
s("GET /schema has state schema", "state" in r)

r = requests.get(f"{BASE}/", allow_redirects=True)
s("GET / returns 200 HTML", r.status_code == 200 and "text/html" in r.headers.get("content-type", ""))

r = requests.post(f"{BASE}/mcp", json={"id": 42, "method": "ping"}).json()
s("POST /mcp returns jsonrpc 2.0", r.get("jsonrpc") == "2.0" and r.get("id") == 42)

# ──────────────────────────────────────────────────────────────────────────────
print("\n[2] STATE BEFORE RESET")
# ──────────────────────────────────────────────────────────────────────────────
r = requests.get(f"{BASE}/state").json()
s("GET /state before reset has null episode_id", r.get("episode_id") is None)
s("GET /state has step_number=0", r.get("step_number") == 0)

# ──────────────────────────────────────────────────────────────────────────────
print("\n[3] GUARD RAILS")
# ──────────────────────────────────────────────────────────────────────────────
r = requests.post(f"{BASE}/step", json={"action": {"action_type": "fetch_eob", "parameters": {}}})
s("POST /step before reset returns 400", r.status_code == 400)

reset("deductive_liability", 1001)
res = step({"action_type": "fly_to_moon", "parameters": {}})
s("Unknown action_type returns action_error", bool(res["info"].get("action_error")))
s("Unknown action_type reward is negative", res["reward"] < 0)

# ──────────────────────────────────────────────────────────────────────────────
print("\n[4] TASK 1 -- deductive_liability  seed=1001")
# ──────────────────────────────────────────────────────────────────────────────
r1 = reset("deductive_liability", 1001)
s("Reset returns episode_id", bool(r1.get("episode_id")))
s("Reset task_name correct", r1.get("task_name") == "deductive_liability")
s("Reset seed correct", r1.get("seed") == 1001)
s("Obs has patient_complaint", bool(r1["observation"].get("patient_complaint")))
s("Obs has api_call_budget", r1["observation"].get("api_call_budget", 0) > 0)

obs = r1["observation"]

auth = step({"action_type": "authenticate_patient", "parameters": {"patient_id": obs["case_id"]}})
s("authenticate_patient: no error", not auth["info"].get("action_error"))
s("authenticate_patient: reward +0.05", abs(auth["reward"] - 0.05) < 0.001)
auth_result = auth["observation"]["last_action_result"]
s("authenticate_patient: authenticated=True", auth_result.get("authenticated") is True)

eob = step({"action_type": "fetch_eob", "parameters": {}})
s("fetch_eob: no error", not eob["info"].get("action_error"))
s("fetch_eob: reward +0.03", abs(eob["reward"] - 0.03) < 0.001)
eob_id = eob["observation"]["last_action_result"]["artifact_id"]

ledger = step({"action_type": "fetch_payment_ledger", "parameters": {}})
s("fetch_payment_ledger: no error", not ledger["info"].get("action_error"))
ledger_id = ledger["observation"]["last_action_result"]["artifact_id"]
total_paid = ledger["observation"]["last_action_result"].get("total_paid", 30.0)

plan = step({"action_type": "fetch_plan_document", "parameters": {}})
s("fetch_plan_document: no error", not plan["info"].get("action_error"))
plan_id = plan["observation"]["last_action_result"]["artifact_id"]

prov = step({"action_type": "fetch_provider_record", "parameters": {}})
s("fetch_provider_record: no error", not prov["info"].get("action_error"))
prov_id = prov["observation"]["last_action_result"]["artifact_id"]

diag = step({"action_type": "write_diagnosis", "parameters": {
    "responsible_party": "billing_system_error",
    "evidence_artifact_ids": [eob_id, ledger_id, plan_id],
    "diagnosis_text": "Copay was not credited in EOB — billing system error."
}})
s("write_diagnosis: no error", not diag["info"].get("action_error"))
s("write_diagnosis: reward positive", diag["reward"] > 0)
diag_id = diag["observation"]["last_action_result"]["artifact_id"]

draft = step({"action_type": "draft_resolution", "parameters": {
    "resolution_type": "refund",
    "refund_amount": round(float(total_paid), 2),
    "summary": "Refund copay not credited in EOB"
}})
s("draft_resolution: no error", not draft["info"].get("action_error"))
draft_id = draft["observation"]["last_action_result"]["artifact_id"]

dl = step({"action_type": "check_deadline", "parameters": {}})
s("check_deadline: no error", not dl["info"].get("action_error"))

sub = step({"action_type": "submit_resolution", "parameters": {"draft_artifact_id": draft_id}})
s("submit_resolution: no error", not sub["info"].get("action_error"))

comm = step({"action_type": "send_patient_communication", "parameters": {
    "message_type": "outcome",
    "message_text": "Your case has been resolved. A refund has been submitted."
}})
s("send_patient_communication: no error", not comm["info"].get("action_error"))
s("patient state transition occurred", comm["observation"]["last_action_result"].get("new_patient_state") in ("calm", "frustrated"))

audit = step({"action_type": "write_audit_entry", "parameters": {
    "summary": "Billing error identified and corrected",
    "outcome_code": "resolved"
}})
s("write_audit_entry: no error", not audit["info"].get("action_error"))

close1 = step({"action_type": "close_case", "parameters": {}})
s("Task1 close_case done=True", close1.get("done") is True)
s("Task1 episode_score present", close1["info"].get("episode_score") is not None)
score1 = close1["info"].get("episode_score", 0)
s(f"Task1 score >= 0.9 (got {score1:.3f})", score1 >= 0.9, f"score={score1}")
print(f"    Task 1 score: {score1:.4f}")
if close1["info"].get("check_results"):
    fails = [cr["description"] for cr in close1["info"]["check_results"] if not cr["passed"]]
    if fails:
        print(f"    Failing checks: {fails}")

# state after episode ends
r_state = requests.get(f"{BASE}/state").json()
s("GET /state after close shows done=True", r_state.get("done") is True)

# ──────────────────────────────────────────────────────────────────────────────
print("\n[5] TASK 2 -- abductive_conflict  seed=2001")
# ──────────────────────────────────────────────────────────────────────────────
r2 = reset("abductive_conflict", 2001)
s("Task2 reset OK", r2.get("task_name") == "abductive_conflict")
obs2 = r2["observation"]

auth2 = step({"action_type": "authenticate_patient", "parameters": {"patient_id": obs2["case_id"]}})
s("Task2 authenticate_patient: no error", not auth2["info"].get("action_error"))

eob2 = step({"action_type": "fetch_eob", "parameters": {}})
s("Task2 fetch_eob: no error", not eob2["info"].get("action_error"))
eob2_id = eob2["observation"]["last_action_result"]["artifact_id"]

prov2 = step({"action_type": "fetch_provider_record", "parameters": {}})
s("Task2 fetch_provider_record: no error", not prov2["info"].get("action_error"))
prov2_id = prov2["observation"]["last_action_result"]["artifact_id"]

code = step({"action_type": "lookup_procedure_code", "parameters": {}})
s("lookup_procedure_code: no error", not code["info"].get("action_error"))
code_id = code["observation"]["last_action_result"]["artifact_id"]

reg2 = step({"action_type": "check_regulatory_rule", "parameters": {"rule_id": "NCCI_MODIFIER_EXCEPTION"}})
s("Task2 check_regulatory_rule: no error", not reg2["info"].get("action_error"))
reg2_id = reg2["observation"]["last_action_result"]["artifact_id"]

diag2 = step({"action_type": "write_diagnosis", "parameters": {
    "responsible_party": "insurer_wrong",
    "evidence_artifact_ids": [eob2_id, prov2_id, code_id],
    "diagnosis_text": "Insurer wrongly denied claim; modifier exception applies."
}})
s("Task2 write_diagnosis: no error", not diag2["info"].get("action_error"))

draft2 = step({"action_type": "draft_resolution", "parameters": {
    "resolution_type": "appeal",
    "appeal_reason": "modifier_exception",
    "summary": "Appeal: modifier exception applies per regulatory rule"
}})
s("Task2 draft_resolution: no error", not draft2["info"].get("action_error"))
draft2_id = draft2["observation"]["last_action_result"]["artifact_id"]

dl2 = step({"action_type": "check_deadline", "parameters": {}})
s("Task2 check_deadline: no error", not dl2["info"].get("action_error"))

sub2 = step({"action_type": "submit_resolution", "parameters": {"draft_artifact_id": draft2_id}})
s("Task2 submit_resolution: no error", not sub2["info"].get("action_error"))

comm2 = step({"action_type": "send_patient_communication", "parameters": {
    "message_type": "outcome", "message_text": "Appeal submitted on your behalf."
}})
s("Task2 send_patient_communication: no error", not comm2["info"].get("action_error"))

notif2 = step({"action_type": "notify_provider", "parameters": {
    "notification_type": "appeal_filed", "message": "Appeal filed on behalf of patient."
}})
s("notify_provider: no error", not notif2["info"].get("action_error"))
s("notify_provider: provider_id present", bool(notif2["observation"]["last_action_result"].get("provider_id")))

audit2 = step({"action_type": "write_audit_entry", "parameters": {"summary": "Claim denial appealed"}})
s("Task2 write_audit_entry: no error", not audit2["info"].get("action_error"))

close2 = step({"action_type": "close_case", "parameters": {}})
s("Task2 close_case done=True", close2.get("done") is True)
score2 = close2["info"].get("episode_score", 0)
s(f"Task2 score >= 0.7 (got {score2:.3f})", score2 >= 0.7, f"score={score2}")
print(f"    Task 2 score: {score2:.4f}")
if close2["info"].get("check_results"):
    fails = [cr["description"] for cr in close2["info"]["check_results"] if not cr["passed"]]
    if fails:
        print(f"    Failing checks: {fails}")

# ──────────────────────────────────────────────────────────────────────────────
print("\n[6] TASK 3 -- adversarial_fabrication  seed=3001")
# ──────────────────────────────────────────────────────────────────────────────
r3 = reset("adversarial_fabrication", 3001)
s("Task3 reset OK", r3.get("task_name") == "adversarial_fabrication")
obs3 = r3["observation"]
s("Task3 obs has rate_limited_tools", "rate_limited_tools" in obs3)

auth3 = step({"action_type": "authenticate_patient", "parameters": {"patient_id": obs3["case_id"]}})
s("Task3 authenticate_patient: no error", not auth3["info"].get("action_error"))

claim = step({"action_type": "fetch_claim_record", "parameters": {}})
s("fetch_claim_record: no error", not claim["info"].get("action_error"))
claim_id = claim["observation"]["last_action_result"]["artifact_id"]

eob3 = step({"action_type": "fetch_eob", "parameters": {}})
s("Task3 fetch_eob: no error", not eob3["info"].get("action_error"))
eob3_id = eob3["observation"]["last_action_result"]["artifact_id"]

prov3 = step({"action_type": "fetch_provider_record", "parameters": {}})
s("Task3 fetch_provider_record: no error", not prov3["info"].get("action_error"))

# fetch_payment_processor_log (may be rate-limited initially; retry once)
proclog = step({"action_type": "fetch_payment_processor_log", "parameters": {}})
if proclog["info"].get("rate_limited"):
    # burn a step with fetch_eob then retry
    step({"action_type": "fetch_eob", "parameters": {}})
    proclog = step({"action_type": "fetch_payment_processor_log", "parameters": {}})
s("fetch_payment_processor_log: no error", not proclog["info"].get("action_error"),
  proclog["info"].get("action_error", ""))
proc_id = proclog["observation"]["last_action_result"].get("artifact_id")

fac = step({"action_type": "fetch_facility_record", "parameters": {}})
s("fetch_facility_record: no error", not fac["info"].get("action_error"))

dl3 = step({"action_type": "check_deadline", "parameters": {}})
s("Task3 check_deadline: no error", not dl3["info"].get("action_error"))

plan3 = step({"action_type": "fetch_plan_document", "parameters": {}})
s("Task3 fetch_plan_document: no error", not plan3["info"].get("action_error"))
qpa = plan3["observation"]["last_action_result"].get("qualifying_payment_amount", 500.0)

reg3 = step({"action_type": "check_regulatory_rule", "parameters": {"rule_id": "NSA_112"}})
s("Task3 check_regulatory_rule: no error", not reg3["info"].get("action_error"))
reg3_id = reg3["observation"]["last_action_result"]["artifact_id"]

diag3 = step({"action_type": "write_diagnosis", "parameters": {
    "responsible_party": "provider_fraud",
    "evidence_artifact_ids": [eob3_id, proc_id, reg3_id] if proc_id else [eob3_id, reg3_id],
    "diagnosis_text": "Provider submitted fabricated GFE; processor log contradicts date."
}})
s("Task3 write_diagnosis: no error", not diag3["info"].get("action_error"))

draft3 = step({"action_type": "draft_resolution", "parameters": {
    "resolution_type": "nsa_dispute",
    "qpa_reference_amount": round(float(qpa), 2),
    "nsa_violation_basis": "Good Faith Estimate date mismatch per processor log",
    "summary": "NSA dispute: fabricated GFE date"
}})
s("Task3 draft_resolution (nsa_dispute): no error", not draft3["info"].get("action_error"))
draft3_id = draft3["observation"]["last_action_result"]["artifact_id"]

sub3 = step({"action_type": "submit_resolution", "parameters": {"draft_artifact_id": draft3_id}})
s("Task3 submit_resolution: no error", not sub3["info"].get("action_error"))
# Phase 2 adversarial counters injected now

for ci in [1, 2, 3]:
    cited = [proc_id, reg3_id] if proc_id else [reg3_id]
    rej = step({"action_type": "reject_counter_argument", "parameters": {
        "counter_index": ci,
        "rejection_reasoning": f"Counter {ci} rejected: processor log is independent evidence.",
        "cited_artifact_ids": cited
    }})
    s(f"reject_counter_argument #{ci}: no error", not rej["info"].get("action_error"),
      rej["info"].get("action_error", ""))

comm3 = step({"action_type": "send_patient_communication", "parameters": {
    "message_type": "outcome",
    "message_text": "NSA dispute has been filed and maintained against provider."
}})
s("Task3 send_patient_communication: no error", not comm3["info"].get("action_error"))

notif3 = step({"action_type": "notify_provider", "parameters": {
    "notification_type": "nsa_dispute_filed",
    "message": "NSA dispute filed; all counter-arguments have been rejected."
}})
s("Task3 notify_provider: no error", not notif3["info"].get("action_error"))

audit3 = step({"action_type": "write_audit_entry", "parameters": {
    "summary": "NSA dispute maintained against provider fraud attempts",
    "outcome_code": "dispute_filed"
}})
s("Task3 write_audit_entry: no error", not audit3["info"].get("action_error"))

close3 = step({"action_type": "close_case", "parameters": {}})
s("Task3 close_case done=True", close3.get("done") is True)
score3 = close3["info"].get("episode_score", 0)
s(f"Task3 score >= 0.7 (got {score3:.3f})", score3 >= 0.7, f"score={score3}")
print(f"    Task 3 score: {score3:.4f}")
if close3["info"].get("check_results"):
    fails = [cr["description"] for cr in close3["info"]["check_results"] if not cr["passed"]]
    if fails:
        print(f"    Failing checks: {fails}")

# ──────────────────────────────────────────────────────────────────────────────
print("\n[7] EDGE CASES")
# ──────────────────────────────────────────────────────────────────────────────

# Duplicate fetch penalty
reset("deductive_liability", 1002)
step({"action_type": "authenticate_patient", "parameters": {"patient_id": "x"}})
step({"action_type": "fetch_eob", "parameters": {}})
dup = step({"action_type": "fetch_eob", "parameters": {}})
s("Duplicate fetch: no server crash", dup.get("reward") is not None)
s("Duplicate fetch: negative or zero reward", dup["reward"] <= 0)

# Default reset (no params)
rd = reset()
s("Default reset (no params) returns episode_id", bool(rd.get("episode_id")))
s("Default reset returns a known task", rd.get("task_name") in (
    "deductive_liability", "abductive_conflict", "adversarial_fabrication"))

# submit_resolution with bad draft_artifact_id
reset("deductive_liability", 1003)
step({"action_type": "authenticate_patient", "parameters": {"patient_id": "x"}})
bad_sub = step({"action_type": "submit_resolution", "parameters": {"draft_artifact_id": 99999}})
s("submit_resolution with bad draft_id returns action_error", bool(bad_sub["info"].get("action_error")))

# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 58)
print(f"  RESULT: {ok_count} OK,  {len(errors)} FAILED")
if errors:
    print("\nFAILURES:")
    for e in errors:
        print(f"  {e}")
print("=" * 58)

sys.exit(0 if not errors else 1)
