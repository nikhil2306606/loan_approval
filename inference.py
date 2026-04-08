import os
import json
from typing import List, Optional
from openai import OpenAI
from env import LoanEnv, ApplicantAction

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME   = os.getenv("MODEL_NAME")   or "mistralai/Mistral-7B-Instruct-v0.3"
HF_TOKEN     = os.getenv("HF_TOKEN")     

MAX_EPISODES = 50

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
"""


def log_start(task: str):
    print(f"[START] task={task}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool):
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error=null",
        flush=True,
    )


def log_end(task: str, score: float, steps: int):
    print(
        f"[END] task={task} score={score:.3f} steps={steps}",
        flush=True,
    )



class LLMAgent:
    def choose_action(self, applicant: dict) -> int:
        if not isinstance(applicant, dict):
            applicant = applicant.model_dump()

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": str(applicant)},
                ],
                max_tokens=32,
                temperature=0.0,
            )
            text = response.choices[0].message.content.strip()
            decision = json.loads(text)
            return 1 if decision.get("approve", False) else 0

        except Exception:
            return 0


def run_easy():
    return run_task("easy")


def run_medium():
    return run_task("medium")


def run_hard():
    return run_task("hard")


def run_task(difficulty: str) -> float:
    agent = LLMAgent()
    env = LoanEnv(difficulty=difficulty)

    correct = 0

    log_start(difficulty)

    for step in range(1, MAX_EPISODES + 1):
        state = env.reset()
        state_dict = state.model_dump()

        action_val = agent.choose_action(state_dict)
        action = ApplicantAction(approve=bool(action_val))

        is_good = env.is_good_applicant(state)
        optimal = True if is_good else False

        _, reward, done, _ = env.step(action)

        if action.approve == optimal:
            correct += 1

        action_str = "approve" if action_val == 1 else "reject"
        log_step(step, action_str, reward, done)

    score = correct / MAX_EPISODES

    log_end(difficulty, score, MAX_EPISODES)

    return score


if __name__ == "__main__":
    print("Easy:", run_easy())
    print("Medium:", run_medium())
    print("Hard:", run_hard())