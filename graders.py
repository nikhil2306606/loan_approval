from env import LoanEnv, ApplicantAction

def evaluate_agent(agent, difficulty, episodes=1000):
    env = LoanEnv(difficulty)
    correct = 0
    for _ in range(episodes):
        state = env.reset()
        
        # Backward compatibility with dict-based agents
        state_dict = state.dict()
        action_val = agent.choose_action(state_dict) 
        
        action = ApplicantAction(approve=bool(action_val))
        
        is_good = env.is_good_applicant(state)
        opt_action = True if is_good else False
        
        if action.approve == opt_action:
            correct += 1
            
        env.step(action)
        
    # Return normalized score 0.0 - 1.0 (Accuracy)
    return correct / episodes

def grader_easy(agent):
    return evaluate_agent(agent, "easy")

def grader_medium(agent):
    return evaluate_agent(agent, "medium")

def grader_hard(agent):
    return evaluate_agent(agent, "hard")
