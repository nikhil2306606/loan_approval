import os
import json
from typing import List, Optional
from openai import OpenAI
from env import LoanEnv, ApplicantAction

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME   = os.getenv("MODEL_NAME")   or "mistralai/Mistral-7B-Instruct-v0.3"
HF_TOKEN     = os.getenv("HF_TOKEN")     

TASK_NAME  = os.getenv("LOAN_ENV_TASK", "all")
BENCHMARK  = "loan_approval_rl"

MAX_EPISODES            = 50
SUCCESS_SCORE_THRESHOLD = 0.75

if not HF_TOKEN:
    raise EnvironmentError(
        "HF_TOKEN environment variable is not set. "
        "Please export HF_TOKEN=<your-huggingface-token> before running."
    )

client = OpenAI(
    base_url=API_BASE_URL, 
    api_key=HF_TOKEN,
)

SYSTEM_PROMPT = """You are a bank loan officer AI. You will be given a loan applicant's profile.
Your job is to decide whether to APPROVE or REJECT the loan.

Respond with ONLY a single JSON object like this:
{"approve": true}
or
{"approve": false}

Base your decision on:
- Credit score (higher is better, 700+ is good)
- Debt-to-income ratio (debt / income, lower is better, under 0.4 is safe)
- Loan-to-income ratio (loan_amount / income, under 0.5 is safe)
- Employment status (1 = employed, 0 = unemployed)

Respond ONLY with the JSON. No explanation."""


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


class LLMAgent:
    """Calls the LLM via OpenAI-compatible client to make approve/reject decisions."""

    def choose_action(self, applicant: dict) -> int:
        if not isinstance(applicant, dict):
            applicant = applicant.model_dump()

        user_msg = (
            f"Applicant profile:\n"
            f"  Annual Income: ${applicant['income']:,.0f}\n"
            f"  Credit Score: {applicant['credit_score']:.0f}\n"
            f"  Existing Debt: ${applicant['debt']:,.0f}\n"
            f"  Loan Requested: ${applicant['loan_amount']:,.0f}\n"
            f"  Employment Status: {'Employed' if applicant['employment_status'] == 1 else 'Unemployed'}\n"
            f"\nDecision (JSON only):"
        )

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                max_tokens=32,
                temperature=0.0,
            )
            text = response.choices[0].message.content.strip()
            text = text.strip("` \n").removeprefix("json").strip()
            decision = json.loads(text)
            return 1 if decision.get("approve", False) else 0

        except Exception as exc:
            err_str = str(exc)
            print(f"[DEBUG] Model request failed: {err_str}", flush=True)
            if any(code in err_str for code in ["401", "403", "Invalid username", "Unauthorized", "sufficient permissions"]):
                raise RuntimeError(
                    "\n\n[AUTH ERROR] HF_TOKEN is invalid or expired.\n"
                    "Fix: huggingface.co/settings/tokens → New token → Fine-grained\n"
                    "→ Enable 'Make calls to serverless Inference Providers'\n"
                ) from exc
            return 0


def run_task(agent: LLMAgent, difficulty: str, step_offset: int) -> tuple:
    """
    Run MAX_EPISODES episodes for one difficulty level.
    Emits [STEP] logs with globally-unique step numbers.
    Returns (accuracy_score, rewards_list).
    """
    env = LoanEnv(difficulty=difficulty)
    correct  = 0
    rewards: List[float] = []

    for episode in range(1, MAX_EPISODES + 1):
        state      = env.reset()
        state_dict = state.model_dump()
        action_val = agent.choose_action(state_dict)
        action     = ApplicantAction(approve=bool(action_val))

        is_good    = env.is_good_applicant(state)
        opt_action = True if is_good else False

        _, reward, done, _ = env.step(action)
        rewards.append(reward)

        if action.approve == opt_action:
            correct += 1

        action_str  = "approve" if action_val == 1 else "reject"
        global_step = step_offset + episode
        log_step(step=global_step, action=action_str, reward=reward, done=done, error=None)

    accuracy = correct / MAX_EPISODES
    return accuracy, rewards


def run_inference() -> None:
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    agent       = LLMAgent()
    all_rewards: List[float] = []
    scores      = {}
    step_offset = 0

    for difficulty in ["easy", "medium", "hard"]:
        score, rewards = run_task(agent, difficulty, step_offset)
        scores[difficulty] = round(score, 3)
        all_rewards.extend(rewards)
        step_offset += MAX_EPISODES

    total_steps = MAX_EPISODES * 3
    avg_score   = sum(scores.values()) / 3
    success     = avg_score >= SUCCESS_SCORE_THRESHOLD

    results = {"model": MODEL_NAME, "scores": scores}
    with open("inference_results.json", "w") as f:
        json.dump(results, f, indent=4)

    log_end(
        success=success,
        steps=total_steps,
        score=avg_score,
        rewards=all_rewards,
    )


if __name__ == "__main__":
    run_inference()