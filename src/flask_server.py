"""
flask_server.py — Exposes the local HF agent as an OpenAI-compatible REST endpoint.

Start this AFTER loading a model (see run_pipeline.py).
The server listens on http://localhost:5050/v1/chat/completions

Usage:
    python src/flask_server.py
or import start_flask_server() and call it from your script.
"""
import threading
from flask import Flask, request, jsonify

from src.config import CFG

flask_app = Flask(__name__)

# These are set by start_flask_server() at runtime
_agent = None


@flask_app.route("/v1/chat/completions", methods=["POST"])
def local_infer():
    data     = request.json
    messages = data.get("messages", [])

    # Build full conversation string for the model
    # Extract system + user messages properly
    system_content = ""
    user_content   = ""

    for m in messages:
        if m["role"] == "system":
            system_content = m["content"]
        elif m["role"] == "user":
            user_content = m["content"]

    # Use the last user message — handles both agent calls and judge calls
    final_input = user_content if user_content else messages[-1]["content"]

    # Pass through LLMAgent which handles chat template correctly
    result = _agent.run(final_input, context="")

    return jsonify({
        "choices": [{
            "message": {
                "role"   : "assistant",
                "content": result["answer"]
            }
        }],
        "model" : CFG["model_key"],
        "usage" : {"total_tokens": result["metadata"].get("tokens_used", 0)}
    })


@flask_app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": CFG["model_key"]})


def start_flask_server(agent, port: int = 5050):
    """Start the Flask server in a background daemon thread."""
    global _agent
    _agent = agent

    def run_flask():
        flask_app.run(host="0.0.0.0", port=port, use_reloader=False)

    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    print(f"✅ Local model server → http://localhost:{port}/v1/chat/completions")
    print(f"   Health check      → http://localhost:{port}/health")
    return flask_thread


if __name__ == "__main__":
    # Standalone usage — load model first
    import sys
    print("ℹ️  Import start_flask_server() and pass your agent instance.")
    print("   Or use run_pipeline.py to launch the full pipeline.")
    sys.exit(0)