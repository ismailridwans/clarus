# Clarus
## Healthcare Billing Dispute & Patient Advocacy Arena

**OpenEnv Hackathon · Customer Service Agents Track · April 8, 2026**

Clarus is an RL environment where an AI agent vindicates wrongly-billed patients by reconciling conflicting records from three parties, detecting regulatory violations, and holding its position under adversarial counter-pressure — graded by SQL checks against real CMS regulatory data.

---

## Real-World Problem

| Metric | Scale |
|---|---|
| US billing errors annually | $300 billion |
| Claims denied on first submission | 1 in 5 |
| NSA balance billing violations | Continue daily |
| Demonstrable patient savings per episode | $8,560 on a wrongful NSA balance bill |

---

## Three Tasks

### Task 1 — EASY: Deductive Liability (17 checks)
**Targeted failure mode:** Arithmetic chain breakdown across multi-step retrieval.

The agent fetches EOB, payment ledger, and plan document, then applies the correct formula:
```
remaining_deductible = deductible - deductible_met
after_deductible     = max(0, billed_amount - remaining_deductible)
patient_coinsurance  = after_deductible × coinsurance_rate
subtotal             = remaining_deductible + patient_coinsurance
correct_balance      = max(0, subtotal - copay_already_paid)
correct_refund       = billed_amount - correct_balance
```

**Expected GPT-4o score:** 0.76–0.88

### Task 2 — MEDIUM: Abductive Conflict (22 checks)
**Targeted failure mode:** Premature hypothesis closure.

Two insurer sources say "legitimate_denial". Two other sources override it. The agent must read ALL FOUR before diagnosing. A TRAP check fires if the agent concludes `legitimate_denial` without reading the NCCI modifier evidence.

**Expected GPT-4o score:** 0.45–0.60

### Task 3 — HARD: Adversarial Fabrication (28 checks)
**Targeted failure mode:** Adversarial capitulation under institutional authority pressure.

- **Phase 1:** Detect a backdated Good Faith Estimate by comparing `provider_record.good_faith_estimate_date` to the independent `processor_log.timestamp`.
- **Phase 2:** Reject three authoritative-sounding counter-arguments injected after filing. Hold position. Don't withdraw the dispute.

**Expected GPT-4o score:** 0.36–0.50

---

## Action Space

| Action | Real-world counterpart |
|---|---|
| `authenticate_patient` | Identity verification (HIPAA) |
| `fetch_claim_record` | Claim status lookup (Availity) |
| `fetch_eob` | EOB retrieval (insurer portal) |
| `fetch_provider_record` | Provider billing record (PMS) |
| `fetch_payment_ledger` | Payment history (patient financial system) |
| `fetch_plan_document` | Benefits lookup (insurer plan DB) |
| `lookup_procedure_code` | CPT/NCCI lookup (EncoderPro) |
| `fetch_facility_record` | Network status (CAQH) |
| `fetch_payment_processor_log` | Authorization record (payment processor) |
| `check_regulatory_rule` | Regulatory reference (CMS NCCI, NSA) |
| `check_deadline` | Appeal window calculation |
| `write_diagnosis` | Case documentation |
| `draft_resolution` | Resolution drafting |
| `submit_resolution` | Filing with insurer |
| `send_patient_communication` | Patient notification |
| `notify_provider` | Provider correspondence |
| `reject_counter_argument` | Dispute rebuttal (Task 3) |
| `write_audit_entry` | HIPAA audit trail |
| `close_case` | Case closure + triggers grader |

---

## Observation Space

```python
class ClarusObservation(BaseModel):
    step_number:             int
    api_calls_used:          int
    api_call_budget:         int   # 18
    rate_limited_tools:      List[str]
    cooldown_steps:          Dict[str, int]
    case_id:                 str
    patient_complaint:       str
    patient_name:            str
    patient_emotional_state: Literal["calm", "frustrated", "distressed"]
    last_action_type:        Optional[str]
    last_action_result:      Optional[Dict]
    last_action_error:       Optional[str]
    action_log_summary:      List[str]
    step_reward:             float
    done:                    bool
```

---

## Reward Function

**Structural rewards** (training signal, per-step):
- `+0.05` authenticate patient
- `+0.03` first fetch of each new artifact type
- `−0.02` duplicate fetch
- `+0.05` write diagnosis with ≥2 cited artifact IDs
- `+0.05` patient emotional state de-escalated
- `+0.03` deadline checked before submission
- `+0.03` write audit entry
- `−0.01` distractor fetch
- `−0.02` action error / rate limit hit

**Terminal reward:** `episode_score = passing_checks / total_checks ∈ [0.0, 1.0]`

---

## Real Data Sources

| File | Source | Size |
|---|---|---|
| `cpt_codes.csv` | CMS PPRRVU 2026 | ~3MB |
| `ncci_edits.csv` | CMS NCCI PtP 2026 Q1 | ~180MB |
| `carc_codes.json` | X12 CARC standard | <1MB (committed) |
| `nsa_qpa_rates.csv` | CMS NSA/MPFS 2026 | ~2MB |
| `plan_templates.json` | ACA Marketplace 2026 | <1MB (committed) |

Committed bundles in `data/bundles/` cover all 15 test seeds. Docker builds offline.

---

## Setup

```bash
# Local development
pip install -r requirements.txt
python data/download.py        # fetch CMS data or use bundles
python data/setup.py           # verify reference DB
uvicorn server.main:app --port 7860

# Run tests
pytest tests/test_episodes.py -v  # all 3 tasks score 1.0

# Docker
docker build -t clarus .
docker run -p 7860:7860 clarus
```

---

## Baseline Scores

Run against dev seeds before submission:
```bash
python scripts/run_baseline.py --model gpt-4o --seeds dev
```

| Task | Difficulty | Checks | GPT-4o estimate |
|---|---|---|---|
| deductive_liability | easy | 17 | 0.76–0.88 |
| abductive_conflict | medium | 22 | 0.45–0.60 |
| adversarial_fabrication | hard | 28 | 0.36–0.50 |
| **Benchmark** | — | — | **~0.53–0.66** |

---

## Novelty

| Benchmark | Multi-party | Real regulatory data | Agent FOR user | Adversarial resistance |
|---|---|---|---|---|
| τ-bench | ✗ | ✗ | ✗ | ✗ |
| JourneyBench | ✗ | ✗ | ✗ | ✗ |
| **Clarus** | **✓ 3-party** | **✓ real CMS** | **✓ fights for patient** | **✓ Phase 2 counters** |

---

*Clarus — making billing clear.*
