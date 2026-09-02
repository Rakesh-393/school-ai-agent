"""Central config. Everything reads from .env so nothing is hardcoded."""
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret")

    # sqlite now, mysql/postgres later -- only this line changes
    SQLALCHEMY_DATABASE_URI = os.getenv(
        "DATABASE_URL", "sqlite:///" + os.path.join(BASE_DIR, "instance", "school.db")
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # --- llm ---
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()

    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    # gpt-oss models are REASONING models: they think before answering, which on
    # a tool-routing task is mostly wasted time. Measured on this project,
    # "low" cut a tool-routing call from 4300ms to 580ms with the same tool
    # choice. Raise to "medium"/"high" only if you see it picking wrong tools.
    GROQ_REASONING_EFFORT = os.getenv("GROQ_REASONING_EFFORT", "low")

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-5")

    # --- rag ---
    CHROMA_DIR = os.getenv("CHROMA_DIR", os.path.join(BASE_DIR, "instance", "chroma"))
    DOCS_DIR = os.path.join(BASE_DIR, "data", "docs")
    RAG_COLLECTION = "school_docs"
    RAG_TOP_K = 4

    # --- agent ---
    MAX_AGENT_STEPS = 5          # hard cap so a confused model can't loop forever
    FUZZY_ACCEPT = 88            # >= this score -> candidate for auto-accept
    FUZZY_MARGIN = 12            # ...but ONLY if it beats runner-up by this much
    FUZZY_SOLO = 75              # a single candidate this good is accepted alone
    FUZZY_SUGGEST = 65           # >= this score -> offer as a "did you mean"


# ---------------------------------------------------------------------------
# Startup self-check.
#
# The single most common way this project fails is an unset or still-placeholder
# API key: every /api/chat then dies with a raw 401 and a 500 page. Detecting it
# once, up front, with a message that says exactly what to edit, is worth far
# more than a stack trace per request.
# ---------------------------------------------------------------------------
PLACEHOLDERS = ("", "gsk_xxxxxxxxxxxxxxxxxxxx", "your-key-here", "changeme")


def provider_status() -> dict:
    """Is the selected LLM provider actually usable? Returns a report, never raises."""
    p = Config.LLM_PROVIDER

    if p == "mock":
        return {"provider": p, "ready": True,
                "note": "Scripted responses. No API key needed."}

    if p == "ollama":
        return {"provider": p, "ready": True, "model": Config.OLLAMA_MODEL,
                "note": f"Needs a local Ollama serving {Config.OLLAMA_MODEL} at {Config.OLLAMA_BASE_URL}."}

    keys = {
        "groq": ("GROQ_API_KEY", Config.GROQ_API_KEY, Config.GROQ_MODEL,
                 "https://console.groq.com/keys"),
        "gemini": ("GEMINI_API_KEY", Config.GEMINI_API_KEY, Config.GEMINI_MODEL,
                   "https://aistudio.google.com/apikey"),
        "anthropic": ("ANTHROPIC_API_KEY", Config.ANTHROPIC_API_KEY, Config.ANTHROPIC_MODEL,
                      "https://console.anthropic.com"),
    }
    if p not in keys:
        return {"provider": p, "ready": False,
                "problem": f"Unknown LLM_PROVIDER {p!r}.",
                "fix": "Set LLM_PROVIDER to one of: groq, gemini, ollama, anthropic, mock."}

    var, value, model, url = keys[p]
    if value.strip() in PLACEHOLDERS:
        return {
            "provider": p, "ready": False, "model": model,
            "problem": f"{var} is not set (still the placeholder from .env.example).",
            "fix": f"Get a free key at {url}, put it in .env as {var}=..., and restart. "
                   f"Or set LLM_PROVIDER=mock to run without any key.",
        }
    return {"provider": p, "ready": True, "model": model}
