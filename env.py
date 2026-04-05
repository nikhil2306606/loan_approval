import random
from pydantic import BaseModel
from typing import Tuple, Dict

class ApplicantState(BaseModel):
    income: float
    credit_score: float
    debt: float
    loan_amount: float
    employment_status: int  

class ApplicantAction(BaseModel):
    approve: bool

class LoanEnv:
    def __init__(self, difficulty="easy"):
        self.difficulty = difficulty
        self._state = None
        self.reset()
        
    def generate_applicant(self) -> ApplicantState:
        if self.difficulty == "easy":
            inc = random.randint(50000, 100000)
            cred = random.randint(700, 850)
            debt = random.randint(0, 20000)
            emp = 1
        elif self.difficulty == "medium":
            inc = random.randint(30000, 100000)
            cred = random.randint(600, 800)
            debt = random.randint(0, 40000)
            emp = random.choice([0, 1, 1])
        else: # hard
            inc = random.randint(20000, 100000)
            cred = random.randint(500, 750)
            debt = random.randint(0, 60000)
            emp = random.choice([0, 1])
            
        loan = random.randint(5000, 30000)
        return ApplicantState(
            income=inc, 
            credit_score=cred, 
            debt=debt, 
            loan_amount=loan, 
            employment_status=emp
        )

    def is_good_applicant(self, applicant: ApplicantState) -> bool:
        score = 0
        if applicant.credit_score > 650: score += 1
        if applicant.debt < applicant.income * 0.5: score += 1
        if applicant.loan_amount < applicant.income * 0.4: score += 1
        if applicant.employment_status == 1: score += 1
        return score >= 3
        
    def reset(self) -> ApplicantState:
        self._state = self.generate_applicant()
        return self._state
        
    def state(self) -> ApplicantState:
        return self._state
        
    def step(self, action: ApplicantAction) -> Tuple[ApplicantState, float, bool, Dict]:
        good = self.is_good_applicant(self._state)
        
        # Reward Logic from hackathon notes
        if action.approve:
            reward = 10.0 if good else -20.0
        else:
            reward = -5.0 if good else 2.0
            
        # Agent completes interaction in one step
        next_state = self.generate_applicant() 
        self._state = next_state
        return next_state, reward, True, {"is_good": good, "profit": reward}