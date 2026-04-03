# Integrating Q-Learning into the Loan Environment

To integrate Q-learning, we need to teach a program (the "agent") to learn from trial and error by playing the bank manager game thousands of times. 

## What is Q-Learning? (Simply Explained)

Imagine you have a giant cheat sheet (called a **Q-Table**). 
- Every **row** is a type of customer (e.g., "High Income, Low Debt, Good Credit").
- Every **column** is an action you can take ("Approve" or "Reject").
- The **numbers** inside the cheat sheet are the points you *expect* to get.

At first, the cheat sheet is completely blank (full of zeros). 
1. **Explore & Exploit**: When a customer walks in, you look at your cheat sheet. Sometimes you guess randomly to see what happens (Explore), and sometimes you pick the action that currently has the highest expected points (Exploit).
2. **Update**: After you make a choice and get your score, you use a math formula to update the cheat sheet. Over time, the cheat sheet becomes perfect!

## Proposed Changes

Because your `env.py` evaluates the customer and ends immediately (`done = True` right away), this is actually a simplified version of Q-Learning called a **Contextual Bandit**. 

Here is how we will modify your code to make the agent learn. We won't modify `env.py`, we will only modify `main.py`.

### 1. Discretization (Making "Buckets")
Your customers have exact numbers (`income = 51234`, `credit_score = 712`). A cheat sheet can't have a row for *every possible exact number*—it would be too big! So, we will group these numbers into simple categories (buckets):
- **Income**: Low (`<50k`), Medium (`50k-80k`), High (`>80k`)
- **Credit Score**: Poor (`<600`), Fair (`600-700`), Good (`>700`)
- **Debt-to-Income Ratio**: Safe (`<30%`), Risky (`>30%`)
- **Loan-to-Income Ratio**: Safe (`<30%`), Risky (`>30%`)

This turns fine-grained exact numbers into simple labels.

### 2. The Learning Loop
We will change `main.py` to:
- Play the game 10,000 times automatically.
- Start with an empty dictionary (our Q-Table).
- Look up the customer's "Bucket" in the dictionary.
- Decide to Approve or Reject (mostly picking the best known option, sometimes guessing randomly to learn).
- Update the dictionary based on the score received.

### [main.py]
#### [MODIFY] [main.py](file:///d:/parallel_minds/rl_hacakton/main.py)
We will completely replace the contents of this file with the Q-Learning training loop.

## User Review Required
> [!IMPORTANT]
> The current system requires categorizing continuous numbers into buckets (e.g., turning a credit score of 720 into simply "Good"). Do you approve of this plan to categorize numbers into "Low", "Medium", and "High" buckets so the AI can easily build its cheat sheet? Once you approve, I will write the code to let the AI learn!
