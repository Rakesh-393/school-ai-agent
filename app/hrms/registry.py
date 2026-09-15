"""
The HRMS registry, shaped exactly like app/agent/tools.py's.

The orchestrator only ever calls tools_for_role() and execute(), so a toolkit
that exposes those two names is a drop-in replacement. That is the whole reason
the agent can change domain without the agent loop changing at all.

ROLES
  employee  -- org structure, subjects, grades, policies, links. Public-ish
               facts anyone inside the company may see.
  hr, admin -- the above plus headcount and vacancies.

Headcount is gated because "how many people work in each unit" is an aggregate
an employee arguably should not be able to mine. Widen it by adding "employee"
to a tool's allowed_for; that is the only change needed.
"""
from __future__ import annotations

from app.hrms import links, tools as t

EVERYONE = {"employee", "hr", "admin"}
STAFF = {"hr", "admin"}

NO_ARGS = {"type": "object", "properties": {}, "additionalProperties": False}

TOOLS: list[dict] = [
    {
        "name": "get_organisation_info",
        "description": "The organisation's name, code, address, city, state and country.",
        "fn": t.get_organisation_info,
        "allowed_for": EVERYONE,
        "input_schema": NO_ARGS,
    },
    {
        "name": "list_units",
        "description": (
            "The branches / campuses / units, with the city, state and country of each, "
            "and the total count. Use for 'how many branches', 'which campuses', "
            "'which country are you in'."
        ),
        "fn": t.list_units,
        "allowed_for": EVERYONE,
        "input_schema": {
            "type": "object",
            "properties": {
                "country": {"type": "string", "description": "Only units in this country."},
                "state": {"type": "string", "description": "Only units in this state."},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "list_sub_units",
        "description": "Sub-units / departments, optionally only those under one unit.",
        "fn": t.list_sub_units,
        "allowed_for": EVERYONE,
        "input_schema": {
            "type": "object",
            "properties": {"unit": {"type": "string", "description": "Unit name to filter by."}},
            "additionalProperties": False,
        },
    },
    {
        "name": "list_locations",
        "description": "Every country, state and city the organisation operates in.",
        "fn": t.list_locations,
        "allowed_for": EVERYONE,
        "input_schema": NO_ARGS,
    },
    {
        "name": "list_subjects",
        "description": "Subjects configured in the system, with their codes.",
        "fn": t.list_subjects,
        "allowed_for": EVERYONE,
        "input_schema": NO_ARGS,
    },
    {
        "name": "list_grades",
        "description": "Grades / classes, with the stream and NEP stage of each.",
        "fn": t.list_grades,
        "allowed_for": EVERYONE,
        "input_schema": NO_ARGS,
    },
    {
        "name": "list_academic_years",
        "description": "Academic years with their start and end dates.",
        "fn": t.list_academic_years,
        "allowed_for": EVERYONE,
        "input_schema": NO_ARGS,
    },
    {
        "name": "list_designations",
        "description": "Job titles, with the function and staff category each belongs to.",
        "fn": t.list_designations,
        "allowed_for": EVERYONE,
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": (
                        "Staff category, e.g. 'Managerial', 'House Keeping', "
                        "'Grade 06 & Grade 08'. Not 'teaching' -- use teaching_only for that."
                    ),
                },
                "teaching_only": {
                    "type": "boolean",
                    "description": "true for academic roles only, false for non-teaching only.",
                },
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "list_departments",
        "description": "The functions and staff categories that make up the org structure.",
        "fn": t.list_departments,
        "allowed_for": EVERYONE,
        "input_schema": NO_ARGS,
    },
    {
        "name": "get_headcount",
        "description": (
            "How many active employees there are, optionally broken down by unit, "
            "designation, category or teaching/non-teaching. Counts only, never names. "
            "HR and admin only."
        ),
        "fn": t.get_headcount,
        "allowed_for": STAFF,
        "input_schema": {
            "type": "object",
            "properties": {
                "group_by": {
                    "type": "string",
                    "description": "One of: unit, designation, category, teaching. Omit for a total.",
                },
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "list_vacancies",
        "description": "Open vacancies with designation and unit, plus the link to apply.",
        "fn": t.list_vacancies,
        "allowed_for": STAFF,
        "input_schema": {
            "type": "object",
            "properties": {"unit": {"type": "string", "description": "Unit name to filter by."}},
            "additionalProperties": False,
        },
    },
    {
        "name": "list_policies",
        "description": (
            "Policy documents on record: name, category and effective date. "
            "Returns the list, not the policy text."
        ),
        "fn": t.list_policies,
        "allowed_for": EVERYONE,
        "input_schema": {
            "type": "object",
            "properties": {"category": {"type": "string", "description": "Policy category."}},
            "additionalProperties": False,
        },
    },
    {
        "name": "get_resource_link",
        "description": (
            "The real URL for a form, portal or contact -- the job application form, the "
            "HRMS login, the HR helpdesk. ALWAYS use this instead of writing a URL "
            "yourself. Omit the topic to list everything on record."
        ),
        "fn": links.resource_link,
        "allowed_for": EVERYONE,
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {"type": "string",
                          "description": "What they want, e.g. 'application form', 'hrms login'."},
            },
            "additionalProperties": False,
        },
    },
]


def _allow_null_on_optional(tools: list[dict]) -> list[dict]:
    """
    Let an optional argument be sent as null.

    Same reason as the school toolkit: providers in strict-schema mode require
    every property to be listed as required, so the model expresses "I am not
    filtering" by sending null. execute() strips the nulls again.
    """
    for tool in tools:
        schema = tool["input_schema"]
        required = set(schema.get("required", []))
        for name, prop in schema.get("properties", {}).items():
            ty = prop.get("type")
            if name not in required and isinstance(ty, str) and ty != "null":
                prop["type"] = [ty, "null"]
    return tools


TOOLS = _allow_null_on_optional(TOOLS)


def tools_for_role(role: str) -> list[dict]:
    """Only advertise what this role may call -- the model can't misuse what it can't see."""
    return [
        {k: v for k, v in t.items() if k in ("name", "description", "input_schema")}
        for t in TOOLS
        if role in t["allowed_for"]
    ]


def execute(name: str, args: dict, role: str) -> dict:
    spec = next((t for t in TOOLS if t["name"] == name), None)
    if spec is None:
        return {"error": "unknown_tool", "message": f"No tool named {name!r}."}
    if role not in spec["allowed_for"]:
        # Second gate: tools_for_role already hid it, but a model can hallucinate a name.
        return {
            "error": "forbidden",
            "message": f"A '{role}' is not permitted to use {name}. Tell the user politely.",
        }
    try:
        allowed = set(spec["input_schema"]["properties"])
        clean = {k: v for k, v in args.items() if k in allowed and v is not None}
        return spec["fn"](**clean)
    except TypeError as e:
        return {"error": "bad_arguments", "message": str(e)}
    except Exception as e:  # never let a tool crash the whole request
        return {"error": "tool_failed", "message": f"{type(e).__name__}: {e}"}
