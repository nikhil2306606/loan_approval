import json
import random
from env import LoanEnv, ApplicantAction
from agent import RandomAgent, RuleBasedAgent, QLearningAgent

random.seed(42)

def run_experiment(agent_name, agent, difficulty, is_q_learning=False):
    env = LoanEnv(difficulty)
    episodes = 10000
    test_episodes = 1000

    if is_q_learning:
        agent.is_training = True
        for _ in range(episodes):
            state = env.reset()
            action_val = agent.choose_action(state)
            action = ApplicantAction(approve=bool(action_val))
            _, reward, _, _ = env.step(action)
            agent.learn(state, action_val, reward)
        agent.is_training = False

    # Evaluate
    total_reward = 0
    correct_choices = 0

    for _ in range(test_episodes):
        state = env.reset()
        action_val = agent.choose_action(state)
        action = ApplicantAction(approve=bool(action_val))

        is_good = env.is_good_applicant(state)
        best_action = 1 if is_good else 0

        if action_val == best_action:
            correct_choices += 1

        _, reward, _, _ = env.step(action)
        total_reward += reward

    return {
        "avg_reward": round(total_reward / test_episodes, 2),
        "accuracy": round(correct_choices / test_episodes, 2),
        "q_table": agent.q_table if is_q_learning else None
    }


if __name__ == "__main__":
    print("Running Hackathon Experiments...")
    difficulties = ["easy", "medium", "hard"]
    results = {
        "easy": [],
        "medium": [],
        "hard": []
    }

    q_table_to_export = {}

    for diff in difficulties:
        print(f"Testing on {diff}...")

        # 1. Random Agent
        rand_res = run_experiment("Random", RandomAgent(), diff)
        rand_res["agent"] = "Random Player"
        del rand_res["q_table"]

        # 2. Rule Based
        rule_res = run_experiment("RuleBased", RuleBasedAgent(), diff)
        rule_res["agent"] = "Human Rules"
        del rule_res["q_table"]

        # 3. Q-Learning
        q_res = run_experiment("Q-Learning", QLearningAgent(), diff, is_q_learning=True)
        q_res["agent"] = "AI (Q-Learning)"

        if diff == "hard":
            q_table_to_export = q_res["q_table"]

        del q_res["q_table"]
        results[diff] = [rand_res, rule_res, q_res]

    print("Saving results.json...")
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("Saving ai_model.json...")
    with open("ai_model.json", "w") as f:
        json.dump(q_table_to_export, f, indent=4)

    print("Done! Data saved for dashboard.")
