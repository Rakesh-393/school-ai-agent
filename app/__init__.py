"""Flask application factory + CLI commands."""
import click
from flask import Flask

from config import Config
from app.models import db


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
        db.create_all()

    _warn_if_misconfigured()

    return app


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
