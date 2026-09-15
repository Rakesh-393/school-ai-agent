"""
Build the SQLAlchemy database URL.

Why this is its own module instead of one line in config.py: SQLite needs a
path, SQL Server needs a host, an instance, a driver, an auth mode and a TLS
policy -- and every one of those has a failure that looks like a different
problem. Keeping the assembly (and the diagnosis) in one place means the fix is
always "edit .env", never "read a traceback".

Two ways to point the app at a database, checked in this order:

  1. DB_BACKEND=mssql + the MSSQL_* settings. The normal path for SQL Server.
  2. DATABASE_URL -- a full SQLAlchemy URL, for a database this module knows
     nothing about, or when you want no assembly applied at all.

Neither set -> SQLite, exactly as before.

DB_BACKEND deliberately outranks DATABASE_URL. Every copy of this project
already carries DATABASE_URL=sqlite:///school.db from .env.example, so the
other order means setting DB_BACKEND=mssql appears to do nothing at all: the
app boots, says SQLite, and the SQL Server settings sit there being ignored
with no error to search for.
"""
import os
import re
import urllib.parse

from sqlalchemy.engine import URL

# Driver 18 flipped the default to Encrypt=yes; 17 and older default to no. We
# always send Encrypt explicitly so moving between driver versions cannot
# silently change whether the connection is encrypted.
_DRIVER_RE = re.compile(r"^ODBC Driver (\d+) for SQL Server$")
FALLBACK_DRIVER = "ODBC Driver 17 for SQL Server"

TRUE_WORDS = ("1", "true", "yes", "y", "on")


def _flag(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in TRUE_WORDS


def installed_sql_server_drivers() -> list[str]:
    """SQL Server ODBC drivers present on this machine, newest first."""
    try:
        import pyodbc
    except ImportError:
        return []

    found = []
    for name in pyodbc.drivers():
        m = _DRIVER_RE.match(name.strip())
        if m:
            found.append((int(m.group(1)), name.strip()))
    return [name for _, name in sorted(found, reverse=True)]


def pick_driver() -> str:
    """MSSQL_DRIVER if set, else the newest driver actually installed."""
    explicit = os.getenv("MSSQL_DRIVER", "").strip()
    if explicit:
        return explicit
    drivers = installed_sql_server_drivers()
    return drivers[0] if drivers else FALLBACK_DRIVER


def odbc_value(value: str) -> str:
    """
    Quote one ODBC connection-string value.

    A connection string is split on semicolons, so a password of "p;w{d}" is
    read as a password of "p" followed by junk -- and the login failure that
    follows blames the credentials, not the punctuation. Braces make the value
    opaque; a literal closing brace inside is doubled, which is the escape ODBC
    itself defines.
    """
    value = str(value)
    if not value:
        return value
    if any(c in value for c in ";{}") or value != value.strip():
        return "{" + value.replace("}", "}}") + "}"
    return value


def _server() -> str:
    """
    Combine host, named instance and port into one SQL Server address.

    The rules are unforgiving and are the #1 cause of "server not found":
      - a named instance uses a BACKSLASH:  localhost\\SQLEXPRESS
      - a port uses a COMMA, not a colon:   10.0.0.5,1433
      - you give one or the other. A named instance is resolved by SQL Browser
        on UDP 1434, so pinning a port alongside it usually just breaks it.
    """
    host = os.getenv("MSSQL_HOST", "localhost").strip()
    instance = os.getenv("MSSQL_INSTANCE", "").strip().lstrip("\\")
    port = os.getenv("MSSQL_PORT", "").strip()

    if instance:
        return f"{host}\\{instance}"
    if port:
        return f"{host},{port}"
    return host


def mssql_odbc_connect_string() -> str:
    """The raw ODBC connection string, exactly as pyodbc wants it."""
    parts = [
        f"DRIVER={{{pick_driver()}}}",
        f"SERVER={odbc_value(_server())}",
        f"DATABASE={odbc_value(os.getenv('MSSQL_DATABASE', 'school').strip())}",
    ]

    user = os.getenv("MSSQL_USER", "").strip()
    if user:
        # SQL Server authentication. odbc_value() handles the punctuation a
        # generated password loves: @ / : ? need nothing here (this is not a
        # URL), and ; { } get brace-quoted.
        parts += [
            f"UID={odbc_value(user)}",
            f"PWD={odbc_value(os.getenv('MSSQL_PASSWORD', ''))}",
        ]
    else:
        # No username -> Windows authentication as whoever runs the process.
        parts.append("Trusted_Connection=yes")

    parts.append("Encrypt=" + ("yes" if _flag("MSSQL_ENCRYPT", "yes") else "no"))
    if _flag("MSSQL_TRUST_SERVER_CERTIFICATE", "yes"):
        # Needed for the self-signed certificate that a default local install
        # generates. Turn it off against a server with a real certificate.
        parts.append("TrustServerCertificate=yes")

    timeout = os.getenv("MSSQL_TIMEOUT", "10").strip()
    if timeout:
        parts.append(f"Connection Timeout={timeout}")

    extra = os.getenv("MSSQL_ODBC_EXTRA", "").strip().strip(";")
    if extra:
        parts.append(extra)

    return ";".join(parts)


def _mssql_url() -> str:
    """
    Wrap the ODBC string in a SQLAlchemy URL via odbc_connect.

    Passing the whole ODBC string as one opaque parameter is deliberate. The
    hand-built alternative, mssql+pyodbc://user:pass@host/db?driver=..., breaks
    the moment a password contains @ / : or ?, or the host contains the
    backslash of a named instance. Those produce a login failure that looks
    like a wrong password, and people rotate a perfectly good password over it.
    """
    return URL.create(
        "mssql+pyodbc",
        query={"odbc_connect": mssql_odbc_connect_string()},
    ).render_as_string(hide_password=False)


def backend() -> str:
    """Which database this configuration is pointed at: 'mssql', 'sqlite', ..."""
    chosen = os.getenv("DB_BACKEND", "").strip().lower()
    if chosen:
        return chosen

    url = os.getenv("DATABASE_URL", "").strip()
    if url:
        scheme = url.split(":", 1)[0]
        # mssql+pyodbc, postgresql+psycopg -- the dialect is the part before "+".
        return "mssql" if scheme.startswith("mssql") else scheme.split("+", 1)[0]
    return "sqlite"


def owns_schema() -> bool:
    """
    May this app create and drop tables in the configured database?

    Only for a database the app owns. SQLite is a file this project created, so
    yes. Anything else is somebody's existing database until told otherwise, and
    the honest default is to keep our hands off it.

    WHY THIS EXISTS: create_all() runs on every boot. Pointed at a real
    company database it tried to create eleven tables inside it, and the run
    only failed because one of our table names, `subjects`, already existed
    there with different columns. That collision is what stopped it -- not any
    safety check, because there wasn't one. Worse, `seed-db --reset` calls
    drop_all(), which would have resolved `DROP TABLE subjects` to their real
    Subjects table. Set DB_CREATE_TABLES=yes to opt a non-SQLite database in.
    """
    override = os.getenv("DB_CREATE_TABLES", "").strip().lower()
    if override:
        return override in TRUE_WORDS
    return backend() == "sqlite"


def database_uri() -> str:
    """The SQLALCHEMY_DATABASE_URI to hand to Flask-SQLAlchemy."""
    if os.getenv("DB_BACKEND", "").strip().lower() == "mssql":
        return _mssql_url()

    explicit = os.getenv("DATABASE_URL", "").strip()
    if explicit:
        return explicit
    # Relative on purpose: Flask-SQLAlchemy resolves it against instance_path,
    # which is already absolute. See the note in config.py.
    return "sqlite:///school.db"


def log_store_uri() -> str:
    """
    Where ChatLog rows go.

    Chat logs are the only thing this app writes, and they have no business
    landing in a database it does not own -- an HR schema should not grow an
    agent's transcript table, and it would refuse to anyway. When the app owns
    the database they stay with everything else; otherwise they go to a local
    SQLite file beside the app.
    """
    if owns_schema():
        return database_uri()
    return "sqlite:///chat_logs.db"


def engine_options() -> dict:
    """
    Engine settings that only make sense for a networked database.

    SQLite is a local file: it has no connections to drop, so it gets nothing.
    """
    if backend() != "mssql":
        return {}

    opts = {
        # A pooled connection that a firewall, load balancer or server restart
        # killed while idle still looks alive to the pool. Without pre_ping the
        # next request gets a dead handle and a random "connection is closed"
        # error somewhere far from the cause.
        "pool_pre_ping": True,
        "pool_recycle": int(os.getenv("MSSQL_POOL_RECYCLE", "1800")),
        "pool_size": int(os.getenv("MSSQL_POOL_SIZE", "5")),
        "max_overflow": int(os.getenv("MSSQL_MAX_OVERFLOW", "10")),
    }
    if _flag("MSSQL_FAST_EXECUTEMANY", "yes"):
        # Turns seeding from one round trip per row into batched inserts.
        # Worth ~50x on the seed script's few thousand attendance rows.
        opts["fast_executemany"] = True
    return opts


def safe_display_uri(uri: str | None = None) -> str:
    """The URI with any password blanked, safe to print in a startup banner."""
    uri = uri or database_uri()
    shown = urllib.parse.unquote(uri)
    brace = r"\{(?:[^}]|\}\})*\}"
    shown = re.sub(r"(?i)(PWD=)(" + brace + r"|[^;]*)", r"\1***", shown)
    shown = re.sub(r"(?i)(://[^:/@]+):[^@]*@", r"\1:***@", shown)
    return shown
