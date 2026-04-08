from env import LoanEnv, ApplicantAction
from agent import RuleBasedAgent


def evaluate_agent(agent, difficulty, episodes=1000):
    env = LoanEnv(difficulty)
    correct = 0

    for _ in range(episodes):
        state = env.reset()
        state_dict = state.model_dump()

        action_val = agent.choose_action(state_dict)
        action = ApplicantAction(approve=bool(action_val))

        is_good = env.is_good_applicant(state)
        optimal_action = True if is_good else False

        if action.approve == optimal_action:
            correct += 1

        env.step(action)

    score = correct / episodes

    if score >= 1.0:
        score = 0.999
    elif score <= 0.0:
        score = 0.001

    return score


class EasyGrader:
    def grade(self):
        agent = RuleBasedAgent()
        return evaluate_agent(agent, "easy")


class MediumGrader:
    def grade(self):
        agent = RuleBasedAgent()
        return evaluate_agent(agent, "medium")


class HardGrader:
    def grade(self):
        agent = RuleBasedAgent()
        return evaluate_agent(agent, "hard")