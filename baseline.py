from agent import RuleBasedAgent, RandomAgent
from graders import EasyGrader, MediumGrader, HardGrader
import json

def run_baseline():
    print("--- Running OpenEnv Baseline Inference ---")

    easy_grader = EasyGrader()
    medium_grader = MediumGrader()
    hard_grader = HardGrader()

    baseline_agent = RuleBasedAgent()
    print("Evaluating Human-Rule Baseline Agent...")
    b_easy = easy_grader.grade(baseline_agent)
    b_medium = medium_grader.grade(baseline_agent)
    b_hard = hard_grader.grade(baseline_agent)
    
    print("\nEvaluating Random Baseline Agent...")
    rand_agent = RandomAgent()
    r_easy = easy_grader.grade(rand_agent)
    r_medium = medium_grader.grade(rand_agent)
    r_hard = hard_grader.grade(rand_agent)
    
    out = {
        "baseline_rules": {
            "easy": b_easy,
            "medium": b_medium,
            "hard": b_hard
        },
        "baseline_random": {
            "easy": r_easy,
            "medium": r_medium,
            "hard": r_hard
        }
    }
    
    with open("baseline_scores.json", "w") as f:
        json.dump(out, f, indent=4)
        
    print("\n Baseline inference complete. Reproducible scores saved to baseline_scores.json.")
    print(f"Rule Based Score (Hard): {b_hard}")
    print(f"Random Score (Hard):     {r_hard}")

if __name__ == "__main__":
    run_baseline()