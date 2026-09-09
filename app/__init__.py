"""Flask application factory + CLI commands."""
import sys

import click
from flask import Flask

from config import Config
from app.models import db

# Windows terminals default to cp1252, which cannot encode much of what an LLM
# emits -- narrow no-break spaces (U+202F), en/em dashes, curly quotes. Printing
# one raises UnicodeEncodeError and kills the command. Force UTF-8 and degrade
# gracefully instead of crashing. Harmless on macOS/Linux, where it is already
# UTF-8. Must run before any output.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def create_app(config_object=Config) -> Flask:
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object(config_object)

    db.init_app(app)

    from app.routes.web import web_bp
    from app.routes.chat import chat_bp

    app.register_blueprint(web_bp)
    app.register_blueprint(chat_bp, url_prefix="/api")

    _register_cli(app)

    with app.app_context():
        _prepare_database()

    _warn_if_misconfigured()

    return app


BANNER = "!" * 72


def _prepare_database() -> None:
    """
    Report on the database, then create any missing tables.

    Against SQLite this cannot really fail. Against SQL Server it fails in five
    or six ordinary ways -- server down, TCP/IP disabled, wrong instance,
    Windows-auth-only server, database not created yet -- and db.create_all()
    turns every one of them into the same 40-line driver traceback at import
    time, before the app can serve even a health check. So we ask first, print
    the specific fix, and let the app boot: the pages that need no database
    still work, and the ones that do fail with a clear message instead.
    """
    from config import database_status

    st = database_status()
    if not st["ready"]:
        print(f"\n{BANNER}\n  DATABASE NOT REACHABLE ({st['backend']})\n"
              f"  {st['uri']}\n\n  {st['problem']}\n  {st['fix']}\n{BANNER}\n")
        return

    print(f"  Database: {st['backend']} -> {st['uri']}")
    if st.get("server"):
        print(f"  {st['server']}")

    try:
        db.create_all()
    except Exception as exc:            # noqa: BLE001 -- report, never crash boot
        print(f"\n{BANNER}\n  Could not create tables: {exc}\n"
              f"  The connection works, so this is usually a permissions problem: "
              f"the login needs db_ddladmin (or db_owner) on this database.\n{BANNER}\n")


def _warn_if_misconfigured() -> None:
    """Say it once, loudly, at startup -- not once per failed request."""
    from config import provider_status

    st = provider_status()
    if st["ready"]:
        print(f"  LLM provider: {st['provider']} ({st.get('model', '-')})")
        if st.get("note"):
            print(f"  {st['note']}")
        return

    line = "!" * 72
    print(f"\n{line}\n  LLM NOT CONFIGURED\n  {st['problem']}\n  {st['fix']}\n{line}\n")


def _register_cli(app: Flask) -> None:
    @app.cli.command("seed-db")
    @click.option("--reset", is_flag=True, help="Drop all tables first.")
    def seed_db(reset):
        """Populate the database with a realistic demo school."""
        from app.seed import seed

        if reset:
            db.drop_all()
        db.create_all()
        stats = seed()
        click.echo(f"Seeded: {stats}")

    @app.cli.command("ingest-docs")
    def ingest_docs():
        """(Re)build the vector index from data/docs/."""
        from app.agent import rag

        click.echo("Embedding documents (first run downloads the ONNX model)...")
        click.echo(f"Done: {rag.ingest_directory()}")

    @app.cli.command("ask")
    @click.argument("question")
    @click.option("--role", default="parent", type=click.Choice(["parent", "teacher", "admin"]))
    def ask(question, role):
        """Ask the agent one question from the terminal."""
        from app.agent.orchestrator import SchoolAgent

        result = SchoolAgent(role=role).ask(question)
        for t in result["trace"]:
            mark = "ok" if t["ok"] else "!!"
            click.echo(f"  [{mark}] {t['tool']}({t['args']})")
        click.echo(f"\n{result['answer']}\n({result['latency_ms']} ms)")

    @app.cli.command("db-check")
    def db_check():
        """Test the database connection and show what is in it."""
        from sqlalchemy import inspect

        from config import Config, database_status

        st = database_status()
        click.echo(f"backend : {st['backend']}")
        click.echo(f"uri     : {st['uri']}")
        if not st["ready"]:
            click.echo(f"\nPROBLEM : {st['problem']}\nFIX     : {st['fix']}")
            raise SystemExit(1)

        if st.get("driver"):
            click.echo(f"driver  : {st['driver']}")
        if st.get("server"):
            click.echo(f"server  : {st['server']}")
        if Config.SQLALCHEMY_ENGINE_OPTIONS:
            click.echo(f"pool    : {Config.SQLALCHEMY_ENGINE_OPTIONS}")

        tables = sorted(inspect(db.engine).get_table_names())
        if not tables:
            click.echo("\ntables  : none yet -- run: flask seed-db")
            return

        click.echo(f"\n{'table':<20} rows")
        for name in tables:
            try:
                n = db.session.execute(
                    db.text(f"SELECT COUNT(*) FROM {db.engine.dialect.identifier_preparer.quote(name)}")
                ).scalar()
            except Exception as exc:                      # noqa: BLE001
                n = f"? ({exc.__class__.__name__})"
            click.echo(f"{name:<20} {n}")
