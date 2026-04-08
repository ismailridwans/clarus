---
title: Clarus
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - healthcare
  - billing
  - customer-service
---

<div align="center">

# 🏥 Clarus
### Healthcare Billing Dispute & Patient Advocacy Arena

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Hackathon%202026-blue?style=flat-square)](https://huggingface.co/spaces/ismailridwans/clarus)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-green?style=flat-square)](LICENSE)
[![HF Space](https://img.shields.io/badge/HuggingFace-Space-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/spaces/ismailridwans/clarus)

*An RL environment where an AI agent fights for wrongly-billed patients — graded by SQL checks against real CMS regulatory data.*

</div>

---

## Overview

Medical billing errors cost Americans **$300 billion annually**. One in five claims is denied on first submission. Clarus puts an AI agent in the role of a patient advocacy specialist tasked with:

- Reconciling conflicting records across **three independent parties** (insurer, provider, payment processor)
- Detecting **regulatory violations** against real CMS CPT, NCCI, and NSA data
- **Holding position** under adversarial counter-pressure from the provider in Phase 2

Graded by deterministic SQL checks — no LLM-as-judge, no subjectivity.

---

## Three Tasks

| # | Task | Difficulty | Checks | Failure Mode Tested |
|---|---|---|---|---|
| 1 | `deductive_liability` | 🟢 Easy | 17 | Arithmetic chain breakdown across multi-step retrieval |
| 2 | `abductive_conflict` | 🟡 Medium | 22 | Premature hypothesis closure |
| 3 | `adversarial_fabrication` | 🔴 Hard | 28 | Adversarial capitulation under authority pressure |

### Task 1 — Deductive Liability
The agent fetches EOB, payment ledger, and plan document, then computes the correct patient balance and files a refund. The billing error is a copay that was never credited.

### Task 2 — Abductive Conflict
Two insurer sources say the denial is legitimate. Two other sources override it. The agent must read **all four** before diagnosing. A TRAP check fires if the agent concludes `legitimate_denial` without consulting the NCCI modifier evidence.

### Task 3 — Adversarial Fabrication *(2-Phase)*
- **Phase 1:** Detect a backdated Good Faith Estimate by comparing `provider_record.gfe_date` to an independent `processor_log.timestamp`
- **Phase 2:** Reject three authoritative counter-arguments injected after filing — EHR notes, a false NSA emergency exception claim, and a legal threat — and hold the dispute

---

## Quick Start

```bash
# Clone and install
git clone https://huggingface.co/spaces/ismailridwans/clarus
cd clarus
pip install -r requirements.txt

# Start the server
uvicorn server.main:app --host 0.0.0.0 --port 7860

# Run all tests
pytest tests/ -v

# Run inference (requires HF token)
export HF_TOKEN=hf_...
python inference.py
```

Or use the live environment directly:

```python
from client import ClarusClient

with ClarusClient("https://ismailridwans-clarus.hf.space") as env:
    obs = env.reset(task_name="deductive_liability", seed=1001)
    print(obs["patient_complaint"])

    result = env.step({
        "action_type": "authenticate_patient",
        "parameters": {"patient_id": obs["case_id"]}
    })
    print(result["reward"])  # +0.05
```

---

## API Reference

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/reset` | Start a new episode |
| `POST` | `/step` | Execute one action |
| `GET` | `/state` | Current episode state |
| `GET` | `/health` | Health check |
| `GET` | `/metadata` | Environment metadata |
| `GET` | `/schema` | Action/observation schemas |
| `POST` | `/mcp` | JSON-RPC 2.0 (MCP mode) |

### Action Space

| Action | Real-World Counterpart |
|---|---|
| `authenticate_patient` | Identity verification (HIPAA) |
| `fetch_claim_record` | Claim status lookup |
| `fetch_eob` | EOB retrieval from insurer portal |
| `fetch_provider_record` | Provider billing record (PMS) |
| `fetch_payment_ledger` | Patient payment history |
| `fetch_plan_document` | Benefits lookup |
| `lookup_procedure_code` | CPT/NCCI code reference |
| `fetch_facility_record` | Network status (CAQH) |
| `fetch_payment_processor_log` | Authorization record |
| `check_regulatory_rule` | CMS NCCI / NSA regulatory reference |
| `check_deadline` | Appeal window calculation |
| `write_diagnosis` | Case documentation |
| `draft_resolution` | Resolution drafting |
| `submit_resolution` | Filing with insurer |
| `send_patient_communication` | Patient notification |
| `notify_provider` | Provider correspondence |
| `reject_counter_argument` | Dispute rebuttal (Task 3 Phase 2) |
| `write_audit_entry` | HIPAA audit trail |
| `close_case` | Case closure — triggers grader |

### Observation Space

```python
class ClarusObservation(BaseModel):
    step_number:             int
    api_calls_used:          int
    api_call_budget:         int                          # varies by task
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

## Reward Structure

### Per-Step (Structural)
| Action | Reward |
|---|---|
| Authenticate patient | `+0.05` |
| First fetch of each new artifact type | `+0.03` |
| Diagnosis with ≥2 cited artifact IDs | `+0.05` |
| Patient emotional state de-escalated | `+0.05` |
| Deadline checked before submission | `+0.03` |
| Write audit entry | `+0.03` |
| Duplicate artifact fetch | `−0.02` |
| Distractor fetch | `−0.01` |
| Action error / rate limit hit | `−0.02` |

### Terminal (Episode Score)
```
episode_score = (passing_checks + 0.5) / (total_checks + 1.0)   ∈ (0, 1)
```
Standard Laplace smoothing. The score is determined **only** by how many SQL grader checks the agent passes — no artificial weights or caps. The formula is always strictly in (0, 1):

| Agent | Task 1 (17 checks) | Task 2 (22 checks) | Task 3 (28 checks) |
|---|---|---|---|
| Perfect (all pass) | 0.972 | 0.978 | 0.983 |
| Zero (none pass) | 0.028 | 0.022 | 0.017 |

---

## Data Sources

All grading uses **real regulatory data** — no synthetic rules.

| Dataset | Source | Coverage |
|---|---|---|
| CPT codes | CMS PPRRVU 2026 | Procedure pricing |
| NCCI edits | CMS NCCI PtP 2026 Q1 | Bundling rules |
| NSA/QPA rates | CMS MPFS 2026 | No Surprises Act |
| Plan templates | ACA Marketplace 2026 | Patient cost-sharing |
| CARC codes | X12 standard | Denial reason codes |

Committed bundles in `data/bundles/` cover all 15 training seeds. Builds work fully offline.

---

## Baseline Performance

```bash
export HF_TOKEN=hf_...
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
```

Heuristic fallback agent (no LLM, 5 dev seeds per task):

| Task | Difficulty | Checks | Baseline Score |
|---|---|---|---|
| `deductive_liability` | 🟢 Easy | 17 | **0.917** |
| `abductive_conflict` | 🟡 Medium | 22 | **0.587** |
| `adversarial_fabrication` | 🔴 Hard | 28 | **0.466** |
| **Overall** | — | 67 | **0.657** |

Scores use Laplace smoothing `(passing + 0.5) / (total + 1)`, always strictly in `(0, 1)`.
Harder tasks require more domain-specific reasoning (NCCI modifiers, NSA Phase 2 rejections with correct artifact citations) that a generic heuristic misses — producing genuine score differentiation by difficulty.

---

## Why Clarus is Unique

| Benchmark | Multi-party Records | Real Regulatory Data | Agent Advocates FOR User | Adversarial Phase 2 |
|---|---|---|---|---|
| τ-bench | ✗ | ✗ | ✗ | ✗ |
| JourneyBench | ✗ | ✗ | ✗ | ✗ |
| **Clarus** | **✓ 3-party** | **✓ Real CMS 2026** | **✓ Patient advocacy** | **✓ Counter-rejection** |

---

## Architecture

```
┌─────────────────────────────────────────────┐
│                FastAPI Server                │
│  POST /reset  ·  POST /step  ·  GET /state  │
└──────────────────┬──────────────────────────┘
                   │
         ┌─────────▼─────────┐
         │    ClarusEnv       │
         │  Episode manager   │
         │  SQLite runtime DB │
         └─────────┬─────────┘
                   │
    ┌──────────────┼──────────────┐
    │              │              │
┌───▼───┐   ┌─────▼─────┐  ┌────▼────┐
│ Tools │   │ Scenario   │  │ Grader  │
│ reads │   │ generator  │  │ SQL     │
│ writes│   │ (seeded)   │  │ checks  │
└───────┘   └───────────┘  └─────────┘
                   │
         ┌─────────▼─────────┐
         │  Reference DB      │
         │  CPT · NCCI · NSA  │
         └───────────────────┘
```

---

<div align="center">

*Clarus — making billing clear.*

**OpenEnv Hackathon · Customer Service Agents Track · April 2026**

</div>
