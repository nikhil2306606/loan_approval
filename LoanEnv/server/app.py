from flask import Flask, jsonify, request
from env import LoanEnv, ApplicantAction
import os

app = Flask(_name_)
_env = LoanEnv(difficulty="hard")

@app.route("/reset", methods=["POST", "GET"])
def reset():
    difficulty = request.args.get("difficulty", "hard")
    if difficulty not in ("easy", "medium", "hard"):
        return jsonify({"error": "difficulty must be easy, medium, or hard"}), 400
    _env.difficulty = difficulty
    state = _env.reset()
    return jsonify(state.model_dump())

@app.route("/step", methods=["POST", "GET"])
def step():
    data = request.get_json(force=True, silent=True) or {}
    if "approve" not in data:
        return jsonify({"error": "request body must include 'approve' (bool)"}), 400
    action = ApplicantAction(approve=bool(data["approve"]))
    next_state, reward, done, info = _env.step(action)
    return jsonify({"state": next_state.model_dump(), "reward": reward, "done": done, "info": info})

@app.route("/state", methods=["GET"])
def state():
    return jsonify(_env.state().model_dump())

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/")
def index():
    return jsonify({"name": "Loan Approval OpenEnv", "status": "ok", "endpoints": ["/reset", "/step", "/state", "/health"]})

if _name_ == "_main_":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)