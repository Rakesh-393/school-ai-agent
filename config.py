"""Central config. Everything reads from .env so nothing is hardcoded."""
import os
import re

from dotenv import load_dotenv

import dburi

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")

# Load .env by ABSOLUTE path, anchored to this file -- never bare load_dotenv().
#
# Bare load_dotenv() searches upward from the CURRENT WORKING DIRECTORY. That
# works when you run `python run.py` from the project folder and silently fails
# everywhere else: an IDE Run button with a different working directory, a cron
# entry, `python "Ai agent/run.py"` from the parent folder. The failure is
# nasty because it is not an error -- every os.getenv() just returns its
# default, so the app boots and reports a missing API key you can plainly see
# sitting in .env.
#
# Anchoring to __file__ makes config independent of where you launch from.
ENV_FOUND = os.path.isfile(ENV_PATH)
load_dotenv(ENV_PATH)


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret")
    SCHOOL_OFFICE_PHONE = os.getenv(
        "SCHOOL_OFFICE_PHONE", "+91-8886127373 / +91-9885136655"
    )

    # SQLite by default, SQL Server when you set DB_BACKEND=mssql. The whole
    # decision -- and every SQL Server quirk that goes with it -- lives in
    # dburi.py, so this stays one line no matter which database you point at.
    #
    # On the SQLite path, leave the relative path alone. Flask-SQLAlchemy
    # resolves a relative sqlite:/// URI against app.instance_path, which is
    # derived from the app package and is therefore already absolute and
    # cwd-independent -- so "sqlite:///school.db" reliably means
    # <project>/instance/school.db. "Helpfully" making it absolute against
    # BASE_DIR moves the database to the project root and hands you a
    # brand-new empty one. (Ask me how I know.)
    DB_BACKEND = dburi.backend()
    SQLALCHEMY_DATABASE_URI = dburi.database_uri()
    SQLALCHEMY_ENGINE_OPTIONS = dburi.engine_options()
    # ChatLog rides a separate bind so the transcript never has to live in a
    # database the agent only reads. Same URI when we own the database, so the
    # school demo keeps its logs in school.db exactly as before.
    SQLALCHEMY_BINDS = {"logs": dburi.log_store_uri()}
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # --- agent domain ---
    # "school" is the demo agent over the bundled SQLite school. "hrms" points
    # the same agent loop at an existing HR database, read-only. The domain
    # decides which tool registry and system prompt are loaded; nothing else.
    AGENT_DOMAIN = os.getenv("AGENT_DOMAIN", "school").strip().lower()

    # --- llm ---
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()

    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    GROQ_TRANSCRIPTION_MODEL = os.getenv("GROQ_TRANSCRIPTION_MODEL", "whisper-large-v3-turbo")
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
    # Same relative-path trap as the database: anchor to the project directory
    # so a run from elsewhere doesn't build a second, empty vector index.
    CHROMA_DIR = os.path.join(BASE_DIR, os.getenv("CHROMA_DIR", "instance/chroma"))
    DOCS_DIR = os.path.join(BASE_DIR, "data", "docs")
    RAG_COLLECTION = "school_docs"
    RAG_TOP_K = 4
    # Below this, the nearest passage is too weak to treat as an answer.
    RAG_MIN_SIMILARITY = float(os.getenv("RAG_MIN_SIMILARITY", "0.35"))

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
        # Distinguish "no .env at all" from "key really is a placeholder".
        # Reporting the wrong one sends you editing a file that is already
        # correct -- which is exactly what happened here once.
        if not ENV_FOUND:
            return {
                "provider": p, "ready": False, "model": model,
                "problem": f"No .env file found at {ENV_PATH}",
                "fix": "Copy .env.example to .env in the project root and put your key in it.",
            }
        return {
            "provider": p, "ready": False, "model": model,
            "problem": f"{var} is empty or still the .env.example placeholder.",
            "fix": f"Get a free key at {url}, set {var}=... in {ENV_PATH} "
                   f"(no quotes, no spaces around =), and restart the server. "
                   f"Or set LLM_PROVIDER=mock to run without any key.",
        }
    return {"provider": p, "ready": True, "model": model}


def database_status() -> dict:
    """
    Can we actually reach the configured database? Returns a report, never raises.

    Same bargain as provider_status(): the failures here are all configuration,
    and each one has a specific fix that a traceback does not name. Connecting
    once at startup and printing that fix beats a 500 page per request.
    """
    uri = Config.SQLALCHEMY_DATABASE_URI
    shown = dburi.safe_display_uri(uri)
    backend = Config.DB_BACKEND

    if backend != "mssql":
        return {"backend": backend, "ready": True, "uri": shown}

    try:
        import pyodbc  # noqa: F401
    except ImportError:
        return {
            "backend": backend, "ready": False, "uri": shown,
            "problem": "DB_BACKEND=mssql but the pyodbc driver is not installed.",
            "fix": "Run: pip install -r requirements.txt",
        }

    driver = dburi.pick_driver()
    installed = dburi.installed_sql_server_drivers()
    if driver not in installed:
        return {
            "backend": backend, "ready": False, "uri": shown,
            "problem": f"ODBC driver {driver!r} is not installed on this machine.",
            "fix": ("Installed SQL Server drivers: "
                    + (", ".join(installed) or "none")
                    + ". Set MSSQL_DRIVER to one of those in .env, or install the "
                      "Microsoft ODBC Driver for SQL Server."),
        }

    from sqlalchemy import create_engine, text
    from sqlalchemy.exc import SQLAlchemyError

    try:
        engine = create_engine(uri, **Config.SQLALCHEMY_ENGINE_OPTIONS)
        with engine.connect() as conn:
            version = conn.execute(text("SELECT @@VERSION")).scalar() or ""
        engine.dispose()
    except SQLAlchemyError as exc:
        return {
            "backend": backend, "ready": False, "uri": shown,
            "problem": f"Cannot connect to SQL Server: {_short_odbc_error(exc)}",
            "fix": _mssql_hint(exc),
        }

    return {
        "backend": backend, "ready": True, "uri": shown,
        "driver": driver,
        "server": version.splitlines()[0].strip() if version else "SQL Server",
    }


def _short_odbc_error(exc: Exception) -> str:
    """
    pyodbc errors arrive as a wall of driver noise. Keep the one sentence.

    A real one, in full:

        ('28000', "[28000] [Microsoft][ODBC Driver 17 for SQL Server][SQL
        Server]Login failed for user 'X'. (18456) (SQLDriverConnect); [28000]
        [Microsoft][ODBC Driver 17 for SQL Server]Invalid connection string
        attribute (0); [28000] ...Login failed for user 'X'. (18456)")

    Every ODBC diagnostic record is concatenated with "; ", and the driver
    repeats itself and throws in unrelated asides. Only the FIRST record is the
    cause -- reading from the end (the obvious thing to do) reliably reports
    "Invalid connection string attribute", which sends you auditing a
    connection string that was never the problem.
    """
    first = str(getattr(exc, "orig", exc)).split("; ")[0]
    first = re.sub(r"""^\(?'?[A-Za-z0-9]{5}'?,?\s*["']?""", "", first)   # tuple wrapper
    # [28000] [Microsoft][ODBC Driver 17 for SQL Server][SQL Server] -- note the
    # space after the SQLSTATE, which is why this allows whitespace between the
    # groups rather than requiring them to be adjacent.
    first = re.sub(r"^(\s*\[[^\]]*\])+\s*", "", first)
    # The message reaches us through repr(), so a Windows login prints as
    # DOMAIN\\user and line breaks as literal \r\n. Undo both.
    first = first.replace("\\r\\n", " ").replace("\\\\", "\\")
    return " ".join(first.split())[:300] or repr(exc)


def _mssql_hint(exc: Exception) -> str:
    """Map the handful of SQLSTATEs people actually hit onto what to edit."""
    msg = str(getattr(exc, "orig", exc))
    server = os.getenv("MSSQL_HOST", "localhost")
    user = os.getenv("MSSQL_USER", "").strip()

    if "18456" in msg or "Login failed" in msg:
        if not user:
            return ("The server answered and rejected your Windows account. That "
                    "account has no login on this instance. Either grant it one, "
                    "or switch to SQL Server authentication by setting MSSQL_USER "
                    "and MSSQL_PASSWORD in .env -- which also needs the server in "
                    "Mixed Mode, since a default install accepts Windows auth only.")
        return (f"The server answered and rejected the login {user!r}. Check "
                "MSSQL_PASSWORD, and that the server has SQL Server authentication "
                "enabled (Mixed Mode) -- a default install accepts Windows auth "
                "only. To use Windows auth instead, leave MSSQL_USER empty.")
    if "4060" in msg or "Cannot open database" in msg:
        db = os.getenv("MSSQL_DATABASE", "school")
        return (f"The login works but the database {db!r} does not exist or is not "
                f"visible to this user. Create it once: CREATE DATABASE [{db}];")
    if "certificate" in msg.lower() or "SSL Provider" in msg:
        return ("TLS rejected the server certificate. For a local or self-signed "
                "server set MSSQL_TRUST_SERVER_CERTIFICATE=yes; to disable "
                "encryption entirely set MSSQL_ENCRYPT=no.")
    if "IM002" in msg:
        return ("The ODBC driver name in MSSQL_DRIVER does not match anything "
                "installed. Installed: "
                + (", ".join(dburi.installed_sql_server_drivers()) or "none"))
    return (f"Could not reach {server}. Check that SQL Server is running, that "
            "TCP/IP is enabled in SQL Server Configuration Manager (it is OFF by "
            "default on Express), that MSSQL_HOST / MSSQL_INSTANCE / MSSQL_PORT "
            "are right, and that the firewall allows port 1433.")
