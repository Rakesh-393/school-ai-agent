from flask import Blueprint, render_template

from app.agent.orchestrator import DOMAINS
from config import Config

web_bp = Blueprint("web", __name__)

ROLE_LABELS = {
    "parent": "Parent", "teacher": "Teacher", "admin": "Admin",
    "employee": "Employee", "hr": "HR",
}

# What the page says and offers, per domain. The role list is NOT here: it comes
# from the agent class itself, so the page can never offer a role the agent
# does not accept. It used to be hardcoded to the school roles, which meant that
# in HRMS mode nobody in the browser could be HR -- "parent" was silently turned
# into "employee", and every HR-only tool was unreachable from the page.
PAGES = {
    "school": {
        "title": "Paramita Schools Assistant",
        "subtitle": "School support",
        "welcome": "Hello, welcome to Paramita Schools.",
        "chips": [
            ("Campus information", "How many branches do you have?"),
            ("CBSE branches", "Which branch is CBSE?"),
            ("School policies", "What is the leave policy?"),
            ("Book a demo", "I would like to book a campus demo."),
        ],
    },
    "hrms": {
        "title": "Paramita HR Assistant",
        "subtitle": "HR support",
        "welcome": "Hello, I can answer questions about the organisation, or do a job for you.",
        "chips": [
            # One question and two jobs, so the difference is one click away.
            ("How many branches?", "How many branches do we have and in which country?"),
            ("Hiring report", "Prepare a hiring report for HR"),
            ("Organisation overview", "Give me an overview of the organisation"),
        ],
    },
}


@web_bp.get("/")
def index():
    domain = Config.AGENT_DOMAIN if Config.AGENT_DOMAIN in PAGES else "school"
    agent = DOMAINS.get(domain, DOMAINS["school"])
    return render_template(
        "chat.html",
        office_phone=Config.SCHOOL_OFFICE_PHONE,
        page=PAGES[domain],
        roles=[(r, ROLE_LABELS.get(r, r.title())) for r in agent.roles],
        default_role=agent.default_role,
    )
