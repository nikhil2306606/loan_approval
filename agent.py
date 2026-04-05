import random

def discretize_state(applicant):
    if not isinstance(applicant, dict):
        applicant = applicant.model_dump()

    if applicant["income"] < 50000: inc = 0
    elif applicant["income"] < 80000: inc = 1
    else: inc = 2

    if applicant["credit_score"] < 600: cred = 0
    elif applicant["credit_score"] < 700: cred = 1
    else: cred = 2

    dti = applicant["debt"] / max(1, applicant["income"])
    debt_bucket = 0 if dti < 0.3 else 1

    lti = applicant["loan_amount"] / max(1, applicant["income"])
    loan_bucket = 0 if lti < 0.3 else 1

    return f"{inc}_{cred}_{debt_bucket}_{loan_bucket}"

class RandomAgent:
    def choose_action(self, state):
        return random.choice([0, 1])

class RuleBasedAgent:
    def choose_action(self, applicant):
        if not isinstance(applicant, dict):
            applicant = applicant.model_dump()

        income = applicant["income"]
        credit_score = applicant["credit_score"]
        debt = applicant["debt"]
        loan_amount = applicant["loan_amount"]

        debt_ratio = debt / income if income > 0 else 0
        loan_ratio = loan_amount / income

        if credit_score > 680 and debt_ratio < 0.4 and loan_ratio < 0.5:
            return 1
        if credit_score < 600:
            return 0
        if debt_ratio > 0.6:
            return 0
        return 1

class QLearningAgent:
    def __init__(self, alpha=0.1, epsilon=0.1):
        self.q_table = {}
        self.alpha = alpha
        self.epsilon = epsilon
        self.is_training = True

    def _get_q(self, state):
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0]
        return self.q_table[state]

    def choose_action(self, applicant):
        state = discretize_state(applicant)
        qs = self._get_q(state)

        if self.is_training and random.uniform(0, 1) < self.epsilon:
            return random.choice([0, 1])

        if qs[1] > qs[0]: return 1
        elif qs[0] > qs[1]: return 0
        else: return random.choice([0, 1])

    def learn(self, applicant, action, reward):
        state = discretize_state(applicant)
        qs = self._get_q(state)
        old_val = qs[action]
        qs[action] = old_val + self.alpha * (reward - old_val)
