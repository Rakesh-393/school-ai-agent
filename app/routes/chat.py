"""JSON API for the agent."""
import re
import uuid

from flask import Blueprint, current_app, jsonify, request, session

from app.agent.orchestrator import get_agent
from app.models import ChatLog, db
from config import Config, provider_status

chat_bp = Blueprint("chat", __name__)

# In-memory conversation history, keyed by session id.
# Fine for learning; move to Redis or the chat_logs table for production.
_HISTORY: dict[str, list[dict]] = {}
_MAX_TURNS = 6  # keep the last 3 exchanges -- free models have small context windows


def _rate_limit_message(text: str) -> str:
    """
    A 429, in words the person chatting can act on.

    This text lands in the chat bubble, so it is written for whoever is asking,
    not for whoever deployed the app. The old message told them to "switch
    LLM_PROVIDER to ollama/mock", which is meaningless to a user and was the
    first thing shown in a live demo.

    Groq's error body already says which limit was hit and how long to wait --
    "on tokens per minute (TPM) ... Please try again in 6.585s" -- so pass on
    the wait rather than a vague "a moment". The per-day limits get their own
    wording, because telling someone to retry in a few seconds when the budget
    is gone until tomorrow sends them retrying for nothing.
    """
    current_app.logger.warning("LLM rate limited: %s", text[:400])
    low = text.lower()

    if "per day" in low or "(tpd)" in low or "(rpd)" in low:
        return ("The assistant has used up its free daily allowance and cannot "
                "answer until it resets. Please contact HR for anything urgent.")

    wait = re.search(r"try again in\s+(?:(\d+)m)?\s*([\d.]+)s", text)
    if wait:
        seconds = int(wait.group(1) or 0) * 60 + float(wait.group(2))
        return (f"The assistant is busy right now: the free AI plan allows only a "
                f"limited amount of work per minute. Please try again in about "
                f"{max(1, round(seconds))} seconds.")
    return ("The assistant is busy right now: the free AI plan allows only a "
            "limited amount of work per minute. Please try again in a minute.")


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
        return _rate_limit_message(text)
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
    # No default here: each domain has its own lowest-privilege role, and
    # hardcoding "parent" would silently downgrade an HR user to a role that
    # does not exist in the HRMS registry.
    role = data.get("role")

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
        result = get_agent(role).ask(question, history=history)
    except Exception as e:
        return jsonify({"error": _explain(e)}), 502

    # Only plain text goes into history -- replaying raw tool blocks across
    # providers is fragile, and the answer already contains the useful facts.
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": result["answer"]})
    del history[:-_MAX_TURNS]

    # Logging is best-effort, and deliberately AFTER the answer is in hand.
    #
    # The answer is the product; the log is bookkeeping. Letting bookkeeping
    # fail the request means a perfectly good reply is thrown away and the user
    # sees a 500 -- which is exactly what happened the first time this ran
    # against a database the app does not own: the agent answered correctly,
    # then the INSERT hit "Invalid object name 'chat_logs'" and the whole
    # response became a stack trace. Never trade a good answer for a log line.
    try:
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
    except Exception as e:                        # noqa: BLE001
        # Roll back or the session stays poisoned and the NEXT request fails too.
        db.session.rollback()
        current_app.logger.warning("Chat log not saved: %s", e)

    return jsonify(result)


@chat_bp.post("/reset")
def reset():
    _HISTORY.pop(session.get("sid", ""), None)
    return jsonify({"ok": True})

@chat_bp.post("/transcribe")
def transcribe():
    """Convert a short browser recording to text for the normal chat flow."""
    if "audio" not in request.files:
        return jsonify({"error": "No audio recording was received."}), 400

    if Config.LLM_PROVIDER != "groq" or not Config.GROQ_API_KEY:
        return jsonify({
            "error": "Voice transcription needs a configured Groq provider."
        }), 503

    try:
        from openai import OpenAI

        audio = request.files["audio"]
        client = OpenAI(
            api_key=Config.GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
        )
        result = client.audio.transcriptions.create(
            model=Config.GROQ_TRANSCRIPTION_MODEL,
            file=(audio.filename or "question.webm", audio.stream, audio.mimetype),
            response_format="json",
            language="en",
        )
        text = (result.text or "").strip()
        if not text:
            return jsonify({"error": "No speech was detected. Please try again."}), 422
        return jsonify({"text": text})
    except Exception as e:
        return jsonify({"error": _explain(e)}), 502


@chat_bp.get("/health")
def health():
    """Also reports whether the LLM is actually usable -- check this first."""
    st = provider_status()
    # "ok" must mean the app can actually answer, not merely that Flask is up.
    return jsonify({"status": "ok" if st["ready"] else "llm_not_configured", **st})
