"""JSON API for the agent."""
import uuid

from flask import Blueprint, jsonify, request, session

from app.agent.orchestrator import SchoolAgent
from app.models import ChatLog, db
from config import Config, provider_status

chat_bp = Blueprint("chat", __name__)

# In-memory conversation history, keyed by session id.
# Fine for learning; move to Redis or the chat_logs table for production.
_HISTORY: dict[str, list[dict]] = {}
_MAX_TURNS = 6  # keep the last 3 exchanges -- free models have small context windows


def _explain(e: Exception) -> str:
    """
    Turn a provider exception into something a human can act on.

    Free tiers fail in a small number of predictable ways, and the raw SDK
    message rarely says what to DO. Map the common ones; pass anything else
    through unchanged so you never hide a real bug.
    """
    name = type(e).__name__
    text = str(e)
    low = text.lower()

    if "authenticationerror" in name.lower() or "401" in text or "invalid api key" in low:
        return (f"The {Config.LLM_PROVIDER} API key was rejected. Check GROQ_API_KEY in .env "
                f"(no quotes, no trailing spaces), or set LLM_PROVIDER=mock to run without a key.")
    if "ratelimit" in name.lower() or "429" in text:
        return (f"{Config.LLM_PROVIDER} rate limit hit -- free tiers are capped per minute. "
                f"Wait a moment and retry, or switch LLM_PROVIDER to ollama/mock.")
    if "connection" in name.lower() or "connect" in low or "timed out" in low:
        return (f"Could not reach {Config.LLM_PROVIDER}. Check your internet connection"
                + (f", and that Ollama is running at {Config.OLLAMA_BASE_URL}."
                   if Config.LLM_PROVIDER == "ollama" else "."))
    if "not_found" in low and "model" in low or "does not exist" in low:
        return (f"The model name is wrong or retired. Check the *_MODEL value in .env for "
                f"provider '{Config.LLM_PROVIDER}'.")
    return f"{name}: {text}"


@chat_bp.post("/chat")
def chat():
    data = request.get_json(silent=True) or {}
    question = (data.get("message") or "").strip()
    role = data.get("role", "parent")

    if not question:
        return jsonify({"error": "message is required"}), 400

    sid = session.get("sid")
    if not sid:
        sid = session["sid"] = uuid.uuid4().hex

    # Fail fast with an actionable message rather than a 401 from the provider.
    st = provider_status()
    if not st["ready"]:
        return jsonify({"error": st["problem"], "fix": st["fix"]}), 503

    history = _HISTORY.setdefault(sid, [])

    try:
        result = SchoolAgent(role=role).ask(question, history=history)
    except Exception as e:
        return jsonify({"error": _explain(e)}), 502

    # Only plain text goes into history -- replaying raw tool blocks across
    # providers is fragile, and the answer already contains the useful facts.
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": result["answer"]})
    del history[:-_MAX_TURNS]

    db.session.add(
        ChatLog(
            session_id=sid,
            role_context=role,
            question=question,
            answer=result["answer"],
            tools_used=",".join(result["tools_used"]),
            latency_ms=result["latency_ms"],
        )
    )
    db.session.commit()

    return jsonify(result)


@chat_bp.post("/reset")
def reset():
    _HISTORY.pop(session.get("sid", ""), None)
    return jsonify({"ok": True})


@chat_bp.get("/health")
def health():
    """Also reports whether the LLM is actually usable -- check this first."""
    st = provider_status()
    # "ok" must mean the app can actually answer, not merely that Flask is up.
    return jsonify({"status": "ok" if st["ready"] else "llm_not_configured", **st})
