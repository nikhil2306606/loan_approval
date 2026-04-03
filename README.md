# 🏦 Loan Approval OpenEnv RL

An intelligent loan approval Reinforcement Learning system that learns to maximize profit while minimizing risk. Built strictly following the **OpenEnv Specification**, this project features a Contextual Bandit Q-Learning AI, automated benchmarking graders, and a stunning, interactive Web Dashboard.

---

## 📂 File Explanations
Here is a breakdown of every piece of the architecture:

### The Core Environment
* **`env.py`**: The core environment built on the OpenEnv spec. Uses strictly typed `pydantic` models (`ApplicantState`, `ApplicantAction`) and provides the `step()`, `reset()`, and `state()` API.
* **`agent.py`**: Contains three independent agent logic classes: `RandomAgent` (guesses randomly), `RuleBasedAgent` (human-crafted baseline rules), and our smart `QLearningAgent` (the AI).

### Execution Scripts
* **`baseline.py`**: The baseline inference script. It tests the Human vs Random agents offline to generate reproducible metrics.
* **`main.py`**: The primary Orchestrator loop. It aggressively trains the AI across 10,000 games on Easy/Medium/Hard, then tests it. It exports performance data to `results.json` and the AI's literal brain (Q-Table) to `ai_model.json`.
* **`graders.py`**: Contains three evaluation graders (`grader_easy`, `grader_medium`, `grader_hard`) that normalize agent performances to a strict `[0.0 to 1.0]` accuracy score.

### Deployment & UI
* **`openenv.yaml`**: The required OpenEnv metadata describing our tasks, metrics, and models.
* **`Dockerfile`**: Packages the app completely so it flawlessly builds and hosts the dashboard on Hugging Face Spaces.
* **`index.html`, `style.css`, `app.js`**: A custom-built, glassmorphism Web Application that visualizes the AI's training metrics in animated graphs, and runs a client-side JavaScript engine to let you test the AI live.

---

## ⚙️ Installation & Requirements
Before running anything, make sure you download the required Python libraries (mainly `pydantic` for the typed models):

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Execute the Project
Run these exact commands sequentially in your terminal to see the system work from start to finish:

**Step 1:** Generate the baseline inference scores.
```bash
python baseline.py
```

**Step 2:** Train the advanced AI models and generate the exported `.json` data dictionaries.
```bash
python main.py
```

**Step 3:** Spin up a local server to view the Web Application Dashboard.
```bash
python -m http.server
```

**Step 4:** Open your web browser and go to your locally hosted dashboard!
```text
http://localhost:8000
```

**Step 5 (Interactive Evaluation):** 
Once you open the webpage, view the comparative graphs. Then, **scroll to the bottom of the page**. You will see an interactive form where you can type in theoretical income/debt statistics. Click the "Evaluate" button, and our trained RL model will run an inference right in your browser to tell you exactly why it would Approve or Reject you!
