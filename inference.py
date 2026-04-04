import os
import json
from openai import OpenAI
from env import LoanEnv, ApplicantAction
from graders import grader_easy, grader_medium, grader_hard

API_BASE_URL = os.environ.get("API_BASE_URL")
MODEL_NAME = os.environ.get("MODEL_NAME")
HF_TOKEN = os.environ.get("HF_TOKEN")

if not all([API_BASE_URL, MODEL_NAME, HF_TOKEN]):
    raise EnvironmentError(
        "Missing required environment variables. "
        "Please set API_BASE_URL, MODEL_NAME, and HF_TOKEN."
    )

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

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


class LLMAgent:
    """Agent that uses an LLM via the OpenAI-compatible client."""

    def choose_action(self, applicant):
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
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=32,
                temperature=0.0,
            )
            text = response.choices[0].message.content.strip()
            # Parse the JSON response
            decision = json.loads(text)
            return 1 if decision.get("approve", False) else 0
        except Exception:
            return 0


def run_inference():
    print("--- OpenEnv LLM Agent Inference ---")
    print(f"Model: {MODEL_NAME}")
    print(f"API:   {API_BASE_URL}")
    print()

    agent = LLMAgent()

    print("Running graders (1000 episodes each)...")
    score_easy = grader_easy(agent)
    print(f"  Easy:   {score_easy:.3f}")

    score_medium = grader_medium(agent)
    print(f"  Medium: {score_medium:.3f}")

    score_hard = grader_hard(agent)
    print(f"  Hard:   {score_hard:.3f}")

    results = {
        "model": MODEL_NAME,
        "scores": {
            "easy": score_easy,
            "medium": score_medium,
            "hard": score_hard,
        }
    }

    with open("inference_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print()
    print("Results saved to inference_results.json")
    print(f"Overall average: {(score_easy + score_medium + score_hard) / 3:.3f}")


if __name__ == "__main__":
    run_inference()
