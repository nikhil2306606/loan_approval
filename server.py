"""
server.py — OpenEnv HTTP API + Dashboard

Exposes the step()/reset()/state() endpoints required by the OpenEnv spec
and serves the static dashboard UI at /.

Endpoints:
  POST /reset?difficulty=easy|medium|hard  -> ApplicantState JSON
  POST /step                               -> {state, reward, done, info}
  GET  /state                              -> ApplicantState JSON
  GET  /health                             -> {"status": "ok"}
  GET  /                                   -> Dashboard HTML
"""
import os
import json
from flask import Flask, jsonify, request, send_from_directory
from env import LoanEnv, ApplicantAction

app = Flask(__name__, static_folder=".", static_url_path="")

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
    return jsonify({
        "state": next_state.model_dump(),
        "reward": reward,
        "done": done,
        "info": info
    })


@app.route("/state", methods=["GET"])
def state():
    return jsonify(_env.state().model_dump())


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/")
def index():
    return jsonify({
        "name": "Loan Approval OpenEnv",
        "status": "ok",
        "endpoints": ["/reset", "/step", "/state", "/health"]
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)