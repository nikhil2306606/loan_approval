from agent import RuleBasedAgent, RandomAgent, QLearningAgent
from graders import grader_easy, grader_medium, grader_hard
import json

def run_baseline():
    print("--- Running OpenEnv Baseline Inference ---")
    
    baseline_agent = RuleBasedAgent()
    print("Evaluating Human-Rule Baseline Agent...")
    b_easy = grader_easy(baseline_agent)
    b_medium = grader_medium(baseline_agent)
    b_hard = grader_hard(baseline_agent)
    
    print("\nEvaluating Random Baseline Agent...")
    rand_agent = RandomAgent()
    r_easy = grader_easy(rand_agent)
    r_medium = grader_medium(rand_agent)
    r_hard = grader_hard(rand_agent)
    
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
        
    print("\n✅ Baseline inference complete. Reproducible scores saved to baseline_scores.json.")
    print(f"Rule Based Score (Hard): {b_hard}")
    print(f"Random Score (Hard):     {r_hard}")

if __name__ == "__main__":
    run_baseline()
