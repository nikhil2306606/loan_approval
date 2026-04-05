---
title: Loan Approval RL
emoji: 🏦
colorFrom: blue
colorTo: green
sdk: docker
python_version: "3.10"
pinned: false
tags:
  - openenv

---
# Loan Approval OpenEnv RL

A reinforcement learning environment where an agent acts as a bank loan officer, learning to maximize profit by correctly approving good applicants and rejecting risky ones.

---

## Environment Description & Motivation

Banks lose money in two ways: approving loans that default (costly bad debt), and rejecting applicants who would have repaid (missed revenue). This environment models that tradeoff. The agent receives a stream of loan applicants described by financial features and must decide whether to approve or reject each one. Reward is asymmetric — false approvals are penalized more heavily than missed opportunities, pushing the agent to learn risk-aware lending behavior.

---

## Observation Space

Each episode step presents an `ApplicantState` with the following fields:

| Field | Type | Description |
|---|---|---|
| `income` | float | Annual income in USD |
| `credit_score` | float | Credit score (300–850) |
| `debt` | float | Existing debt in USD |
| `loan_amount` | float | Requested loan amount in USD |
| `employment_status` | int | 1 = employed, 0 = unemployed |

---

## Action Space

The agent returns an `ApplicantAction`:

| Field | Type | Description |
|---|---|---|
| `approve` | bool | `True` to approve the loan, `False` to reject |

---

## Reward Function

| Situation | Reward |
|---|---|
| Approve a good applicant | +10.0 |
| Approve a risky applicant | -20.0 |
| Reject a good applicant | -5.0 |
| Reject a risky applicant | +2.0 |

An applicant is "good" if they score 3+ on: credit score > 650, debt < 50% of income, loan < 40% of income, employed.

---

## Tasks & Difficulty Levels

| Task | Grader | Description | Expected Difficulty |
|---|---|---|---|
| Easy | `grader_easy` | High-income, high-credit, low-debt applicants only. Most should be approved. | Low |
| Medium | `grader_medium` | Mixed profiles with moderate income and credit variability. | Medium |
| Hard | `grader_hard` | Wide income range (20k–100k), lower credit (500–750), high debt variability. Requires nuanced decisions. | High |

All graders return a normalized accuracy score in **[0.0, 1.0]**.

---

## Baseline Scores

Reproducible scores from `baseline.py` (1000 episodes per task):

| Agent | Easy | Medium | Hard |
|---|---|---|---|
| Rule-Based (Human) | 1.000 | 0.791 | 0.750 |
| Random | 0.531 | 0.490 | 0.508 |

---

## Setup & Installation

```bash
pip install -r requirements.txt
```

---

## How to Run

**Step 1:** Generate baseline inference scores.
```bash
python baseline.py
```

**Step 2:** Train the Q-Learning agent and export results.
```bash
python main.py
```

**Step 3:** Start the API server + dashboard.
```bash
python server.py
```

**Step 4:** Open the dashboard.
```
http://localhost:8000
```

**Step 5 (LLM Agent Inference):** Set environment variables, then run:
```bash
export API_BASE_URL="https://your-llm-endpoint"
export MODEL_NAME="your-model-name"
export HF_TOKEN="your-token"
python inference.py
```

---

## File Overview

| File | Purpose |
|---|---|
| `env.py` | Core OpenEnv environment with `step()`, `reset()`, `state()` |
| `agent.py` | RandomAgent, RuleBasedAgent, QLearningAgent |
| `graders.py` | `grader_easy`, `grader_medium`, `grader_hard` |
| `baseline.py` | Baseline inference script (reproducible scores) |
| `inference.py` | LLM agent inference script (uses OpenAI client) |
| `main.py` | Trains Q-Learning agent, exports results JSON |
| `server.py` | Flask API server exposing `/reset`, `/step`, `/state` |
| `openenv.yaml` | OpenEnv spec metadata |
| `Dockerfile` | Container build — trains model at build time, serves API |
| `index.html` / `style.css` / `app.js` | Web dashboard |
