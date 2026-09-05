"""
THE TOOL REGISTRY -- the agent's entire capability surface.

DESIGN RULE: the LLM never writes SQL. It picks a function name and fills in
typed arguments. Every DB query in this file is a hand-written, parameterised
SQLAlchemy query. Consequences:

  * no SQL injection surface at all
  * no accidental `SELECT * FROM students` dumping the whole school
  * every tool is unit-testable without an LLM
  * role-based access control is enforceable (see `allowed_for`)

Adding a capability = adding one entry to TOOLS. Nothing else changes.
"""
from __future__ import annotations

import calendar
import re
from datetime import date, timedelta

from sqlalchemy import case, func

from app.agent import rag
from app.agent.resolver import resolve
from config import Config
from app.models import (
    Attendance,
    Branch,
    ClassSection,
    Exam,
    FeeInvoice,
    Homework,
    Mark,
    Student,
    Subject,
    TimetableSlot,
    db,
)


# ---------------------------------------------------------------- helpers
def _student_extra(s: Student) -> dict:
    return {
        "class": s.klass.label if s.klass else "-",
        "branch": s.klass.branch.name if s.klass and s.klass.branch else "-",
        "admission_no": s.admission_no,
    }


def _find_student(name_or_admission: str):
    """
    Resolve a student. Returns (Student, None) or (None, error_payload).

    Exact admission number wins immediately -- it is unambiguous by definition,
    so we never fuzzy-match when the user gives an ID.
    """
    q = (name_or_admission or "").strip()
    if not q:
        return None, {
            "error": "missing_student",
            "message": "Ask the user for the student's name.",
        }

    exact = Student.query.filter(func.lower(Student.admission_no) == q.lower()).first()
    if exact:
        return exact, None

    students = Student.query.all()
    r = resolve(q, students, label_attr="full_name", extra_fn=_student_extra)
    if r.status != "matched":
        return None, r.as_tool_payload()
    return db.session.get(Student, r.best.id), None


def _find_branch(name: str):
    """
    Resolve a branch by name, code, locality or nickname.

    Branches get `alias_attr` because an institution has many names: the official
    one ("Explorica Premium School"), the locality people actually say
    ("Jyothinagar"), the board ("the Cambridge one"), the short code ("EXP").
    Matching only the official name would fail on nearly every real question.
    """
    branches = Branch.query.all()
    r = resolve(
        name,
        branches,
        label_attr="name",
        alias_attr="alias_list",
        extra_fn=lambda b: {"locality": b.locality, "board": b.board},
    )
    if r.status != "matched":
        payload = r.as_tool_payload()
        payload.setdefault("available_branches", [b.name for b in branches])
        return None, payload
    return db.session.get(Branch, r.best.id), None


def _branch_dict(b: Branch, with_counts: bool = False) -> dict:
    out = {
        "name": b.name,
        "code": b.code,
        "locality": b.locality,
        "city": b.city,
        "state": b.state,
        "board": b.board,
        "grades": b.grades,
        "address": b.address,
        "phone": b.phone,
        "email": b.email,
        "website": b.website,
        "principal": b.principal,
        "established": b.established,
    }
    if with_counts:
        out["classes"] = len(b.sections)
        out["students"] = sum(len(s.students) for s in b.sections)
    return {k: v for k, v in out.items() if v not in (None, "")}


def _find_subject(name: str):
    subjects = Subject.query.all()
    r = resolve(name, subjects, label_attr="name", extra_fn=lambda s: {"code": s.code})
    if r.status != "matched":
        return None, r.as_tool_payload()
    return db.session.get(Subject, r.best.id), None


ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7,
         "viii": 8, "ix": 9, "x": 10, "xi": 11, "xii": 12}


def _find_class(label: str, branch: str | None = None):
    """
    Match a class. Accepts '7B', '7-B', 'grade 7 b', 'class VII B', '7th B'.

    NOTE: this is NOT fuzzy-matched first, and that is deliberate. Class labels
    are *structured* ("grade" + "section"), so a regex parse is both exact and
    cheap. Fuzzy matching on short codes is dangerous -- WRatio("7A", "7B") is
    50, and WRatio("7B", "Grade 7-B") is low enough to miss entirely. Reach for
    fuzzy matching only when the data is genuinely free text, like a person's
    name. Structure first, fuzz second.

    MULTI-BRANCH: "Grade 7-B" exists at several campuses, so a class name alone
    is no longer a unique key. If `branch` is given we scope to it; if not and
    more than one campus has that class, we return `ambiguous_class` so the
    agent asks *which campus* rather than silently picking the first row.
    """
    text = (label or "").strip().lower()
    sections = ClassSection.query.all()

    if branch:
        b, err = _find_branch(branch)
        if err:
            return None, err
        sections = [c for c in sections if c.branch_id == b.id]

    m = re.search(r"(\d{1,2}|[ivx]+)\s*(?:st|nd|rd|th)?\s*[-–/ ]?\s*([a-z])\b", text)
    if m:
        raw_grade, section = m.group(1), m.group(2).upper()
        grade = str(ROMAN.get(raw_grade, raw_grade))
        hits = [c for c in sections if c.grade == grade and c.section.upper() == section]
        if len(hits) == 1:
            return hits[0], None
        if len(hits) > 1:
            return None, {
                "error": "ambiguous_class",
                "query": label,
                "message": f"{len(hits)} campuses have {hits[0].label}. Ask the user which branch.",
                "did_you_mean": [
                    {"class": c.label, "branch": c.branch.name, "room": c.room} for c in hits
                ],
            }

    # Fallback: fuzzy on the human label, for inputs like "seven b".
    r = resolve(label, sections, label_attr="label",
                extra_fn=lambda c: {"branch": c.branch.name, "room": c.room})
    if r.status != "matched":
        payload = r.as_tool_payload()
        payload.setdefault(
            "available_classes", sorted({f"{c.label} @ {c.branch.name}" for c in sections})
        )
        return None, payload
    return db.session.get(ClassSection, r.best.id), None


def _latest_attendance_day(student_id: int | None = None) -> date | None:
    """Most recent day we hold attendance for (optionally for one student)."""
    q = db.session.query(func.max(Attendance.day))
    if student_id is not None:
        q = q.filter(Attendance.student_id == student_id)
    return q.scalar()


def _month_range(month: str | None, anchor: date | None = None):
    """
    'july' / '2025-07' / None -> (first_day, last_day).

    `anchor` is what "no month given" means. It defaults to today, but callers
    pass the latest day they actually hold data for.

    WHY: attendance is recorded up to the last school day, not up to today. On
    the 2nd of a new month, "what's Kabir's attendance?" with month=None asked
    for the new month, found zero rows, and the agent reported no records for a
    student with four months of history. Defaulting to the current *calendar*
    month is only correct if your data is always current -- which it never is.
    """
    today = anchor or date.today()
    if not month:
        y, m = today.year, today.month
    else:
        text = month.strip().lower()
        try:
            if "-" in text:
                y, m = (int(x) for x in text.split("-")[:2])
            else:
                names = {n.lower(): i for i, n in enumerate(calendar.month_name) if n}
                abbr = {n.lower(): i for i, n in enumerate(calendar.month_abbr) if n}
                m = names.get(text) or abbr.get(text[:3])
                if not m:
                    raise ValueError(text)
                y = today.year
        except (ValueError, TypeError):
            y, m = today.year, today.month
    return date(y, m, 1), date(y, m, calendar.monthrange(y, m)[1])


# ---------------------------------------------------------------- tool impls
def list_branches(board: str | None = None) -> dict:
    """
    How many branches, and which ones.

    Returns the COUNT explicitly. If you only return a list and let the model
    count it, it will occasionally miscount -- language models are unreliable at
    counting list items. Compute aggregates in Python, always.
    """
    branches = Branch.query.order_by(Branch.id).all()

    matched_board = None
    if board:
        boards = sorted({b.board for b in branches if b.board})
        fake = [type("B", (), {"id": i, "full_name": x})() for i, x in enumerate(boards)]
        r = resolve(board, fake)
        if r.status == "matched":
            matched_board = boards[r.best.id]
            branches = [b for b in branches if b.board == matched_board]
        else:
            return {
                "error": "unknown_board",
                "query": board,
                "message": "No branch uses that board.",
                "available_boards": boards,
            }

    return {
        "total_branches": len(branches),
        "filtered_by_board": matched_board,
        "branches": [_branch_dict(b, with_counts=True) for b in branches],
    }


def get_branch_info(branch_name: str) -> dict:
    """Full details of one campus: address, board, grades, phone, principal."""
    b, err = _find_branch(branch_name)
    if err:
        return err
    return _branch_dict(b, with_counts=True)


def get_student_profile(student_name: str) -> dict:
    s, err = _find_student(student_name)
    if err:
        return err
    return {
        "name": s.full_name,
        "admission_no": s.admission_no,
        "class": s.klass.label if s.klass else None,
        "branch": s.klass.branch.name if s.klass and s.klass.branch else None,
        "class_teacher": s.klass.class_teacher if s.klass else None,
        "roll_no": s.roll_no,
        "date_of_birth": str(s.date_of_birth) if s.date_of_birth else None,
        "guardian": s.guardian_name,
        "guardian_phone": s.guardian_phone,
        "bus_route": s.bus_route,
    }


def get_attendance_summary(student_name: str, month: str | None = None) -> dict:
    s, err = _find_student(student_name)
    if err:
        return err

    latest = _latest_attendance_day(s.id)
    start, end = _month_range(month, anchor=latest)
    rows = (
        Attendance.query.filter(
            Attendance.student_id == s.id,
            Attendance.day.between(start, end),
        )
        .order_by(Attendance.day)
        .all()
    )
    if not rows:
        # Say what we DO have. "No records" with no further detail sends the
        # agent off to invent an explanation; a concrete range lets it offer
        # the user a period that will actually work.
        earliest = db.session.query(func.min(Attendance.day)).filter(
            Attendance.student_id == s.id
        ).scalar()
        return {
            "student": s.full_name,
            "period": f"{start} to {end}",
            "message": "No attendance records for this period.",
            "records_available_from": str(earliest) if earliest else None,
            "records_available_to": str(latest) if latest else None,
        }

    counts: dict[str, int] = {}
    for r in rows:
        counts[r.status] = counts.get(r.status, 0) + 1

    present = counts.get("present", 0) + counts.get("late", 0)
    pct = round(present / len(rows) * 100, 1)

    return {
        "student": s.full_name,
        "class": s.klass.label if s.klass else None,
        "period": f"{start} to {end}",
        "school_days": len(rows),
        "breakdown": counts,
        "attendance_percent": pct,
        "absent_dates": [str(r.day) for r in rows if r.status == "absent"],
        "flag": "BELOW_75_PERCENT" if pct < 75 else "OK",
    }


def get_marks(
    student_name: str, exam_name: str | None = None, subject: str | None = None
) -> dict:
    s, err = _find_student(student_name)
    if err:
        return err

    q = Mark.query.filter(Mark.student_id == s.id)

    if exam_name:
        r = resolve(exam_name, Exam.query.all(), label_attr="name")
        if r.status == "matched":
            q = q.filter(Mark.exam_id == r.best.id)
    if subject:
        subj, serr = _find_subject(subject)
        if serr:
            return serr
        q = q.filter(Mark.subject_id == subj.id)

    rows = q.all()
    if not rows:
        return {"student": s.full_name, "message": "No marks recorded for that filter."}

    items = [
        {
            "exam": m.exam.name,
            "subject": m.subject.name,
            "scored": m.scored,
            "out_of": m.max_marks,
            "percent": round(m.scored / m.max_marks * 100, 1),
            "grade": m.grade,
        }
        for m in rows
    ]
    total = sum(i["scored"] for i in items)
    out_of = sum(i["out_of"] for i in items)
    return {
        "student": s.full_name,
        "class": s.klass.label if s.klass else None,
        "results": items,
        "overall_percent": round(total / out_of * 100, 1) if out_of else None,
    }


def get_fee_status(student_name: str) -> dict:
    s, err = _find_student(student_name)
    if err:
        return err

    invoices = (
        FeeInvoice.query.filter_by(student_id=s.id).order_by(FeeInvoice.due_date).all()
    )
    if not invoices:
        return {"student": s.full_name, "message": "No fee invoices on record."}

    def status_of(inv):
        if inv.balance <= 0:
            return "paid"
        if inv.due_date and inv.due_date < date.today():
            return "overdue"
        return "pending"

    items = [
        {
            "term": i.term,
            "head": i.head,
            "amount": i.amount,
            "paid": i.paid,
            "balance": i.balance,
            "due_date": str(i.due_date) if i.due_date else None,
            "status": status_of(i),
        }
        for i in invoices
    ]
    return {
        "student": s.full_name,
        "invoices": items,
        "total_outstanding": round(sum(i["balance"] for i in items), 2),
        "currency": "INR",
    }


def get_homework(class_name: str, days: int = 7, branch: str | None = None) -> dict:
    k, err = _find_class(class_name, branch)
    if err:
        return err

    since = date.today() - timedelta(days=days)
    rows = (
        Homework.query.filter(
            Homework.class_section_id == k.id, Homework.assigned_on >= since
        )
        .order_by(Homework.due_on)
        .all()
    )
    return {
        "class": k.label,
        "branch": k.branch.name if k.branch else None,
        "window_days": days,
        "homework": [
            {
                "subject": h.subject.name,
                "assigned_on": str(h.assigned_on),
                "due_on": str(h.due_on) if h.due_on else None,
                "details": h.details,
            }
            for h in rows
        ]
        or "No homework assigned in this window.",
    }


def get_timetable(class_name: str, weekday: str | None = None, branch: str | None = None) -> dict:
    k, err = _find_class(class_name, branch)
    if err:
        return err

    q = TimetableSlot.query.filter_by(class_section_id=k.id)
    if weekday:
        days = list(calendar.day_name)
        fake = [type("Day", (), {"id": i, "full_name": d})() for i, d in enumerate(days)]
        r = resolve(weekday, fake)
        if r.status == "matched":
            q = q.filter(TimetableSlot.weekday == days[r.best.id])

    rows = q.order_by(TimetableSlot.weekday, TimetableSlot.period).all()
    grouped: dict[str, list] = {}
    for slot in rows:
        grouped.setdefault(slot.weekday, []).append(
            {
                "period": slot.period,
                "time": slot.starts_at,
                "subject": slot.subject.name if slot.subject else "Free",
                "teacher": slot.subject.teacher if slot.subject else None,
            }
        )
    return {
        "class": k.label,
        "branch": k.branch.name if k.branch else None,
        "room": k.room,
        "timetable": grouped or "No timetable configured.",
    }


def get_class_attendance_report(
    class_name: str, month: str | None = None, branch: str | None = None
) -> dict:
    """Teacher/admin view: who in this class is below 75%."""
    k, err = _find_class(class_name, branch)
    if err:
        return err

    start, end = _month_range(month, anchor=_latest_attendance_day())
    rows = (
        db.session.query(
            Student.full_name,
            func.count(Attendance.id).label("total"),
            func.sum(
                case((Attendance.status.in_(["present", "late"]), 1), else_=0)
            ).label("present"),
        )
        .join(Attendance, Attendance.student_id == Student.id)
        .filter(Student.class_section_id == k.id, Attendance.day.between(start, end))
        .group_by(Student.id)
        .all()
    )
    report = [
        {
            "student": n,
            "school_days": t,
            "present": int(p or 0),
            "percent": round((p or 0) / t * 100, 1),
        }
        for n, t, p in rows
    ]
    report.sort(key=lambda r: r["percent"])
    return {
        "class": k.label,
        "branch": k.branch.name if k.branch else None,
        "period": f"{start} to {end}",
        "students": report,
        "below_75_percent": [r["student"] for r in report if r["percent"] < 75],
    }


def search_school_documents(query: str) -> dict:
    """RAG: policies, circulars, handbook -- anything that lives in prose."""
    hits = rag.search(query)
    if not hits:
        return {
            "error": "no_documents",
            "message": "No school documents indexed. Run: flask ingest-docs",
        }

    top = hits[0]
    text = top["text"].lower()
    asks_for_amount = any(word in query.lower() for word in
                          ("amount", "cost", "fee", "fees", "price", "how much"))
    asks_about_admission = "admission" in query.lower()
    has_amount = bool(re.search(r"(?:rs|inr|₹)\s*[\d,]+|\b\d[\d,]{2,}\b", text))
    has_admission_amount = "admission" in text and has_amount
    answerable = top["similarity"] >= Config.RAG_MIN_SIMILARITY and (
        not asks_for_amount or has_amount
    ) and (
        not asks_about_admission or has_admission_amount
    )
    return {
        "query": query,
        "passages": hits,
        "answerable": answerable,
        "instruction": (
            "Answer the exact question only when answerable is true. If false, "
            "say the school documents do not contain that detail; do not turn a "
            "related policy into an answer. For general questions, answer from "
            "your general knowledge and label it as general guidance. Cite a "
            "school source only when you use its passage. Offer to help book a "
            "campus demo for missing admissions information."
        ),
    }


# ---------------------------------------------------------------- registry
# `allowed_for` is the access-control list. A parent must never be able to call
# the class-wide report -- that would leak other families' data.
BRANCH_ARG = {
    "type": "string",
    "description": "Campus name or locality, e.g. 'Heritage', 'Jyothinagar', 'World School'. Omit if the user did not say.",
}

TOOLS: list[dict] = [
    {
        "name": "list_branches",
        "description": (
            "How many branches/campuses the school group has, and the details of each. "
            "Use for 'how many branches do you have', 'which campuses are there', "
            "'which branch is CBSE', 'list your schools'."
        ),
        "fn": list_branches,
        "allowed_for": {"parent", "teacher", "admin"},
        "input_schema": {
            "type": "object",
            "properties": {
                "board": {
                    "type": "string",
                    "description": "Optional filter: 'CBSE', 'State', 'Cambridge'. Omit to list all branches.",
                }
            },
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_branch_info",
        "description": (
            "Full details of ONE campus: address, board, grades offered, phone, email, "
            "website, principal, year established, and how many students it has."
        ),
        "fn": get_branch_info,
        "allowed_for": {"parent", "teacher", "admin"},
        "input_schema": {
            "type": "object",
            "properties": {
                "branch_name": {
                    "type": "string",
                    "description": "Branch name, code or locality as the user typed it, e.g. 'Explorica', 'Alugunoor'.",
                }
            },
            "required": ["branch_name"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_student_profile",
        "description": "Basic details of one student: class, roll number, guardian, bus route, DOB.",
        "fn": get_student_profile,
        "allowed_for": {"parent", "teacher", "admin"},
        "input_schema": {
            "type": "object",
            "properties": {
                "student_name": {
                    "type": "string",
                    "description": "Student's name or admission number, as the user typed it.",
                }
            },
            "required": ["student_name"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_attendance_summary",
        "description": "Attendance for ONE student for a month: present/absent counts, percentage, absent dates.",
        "fn": get_attendance_summary,
        "allowed_for": {"parent", "teacher", "admin"},
        "input_schema": {
            "type": "object",
            "properties": {
                "student_name": {
                    "type": "string",
                    "description": "Student's name or admission number.",
                },
                "month": {
                    "type": "string",
                    "description": "Month like 'July' or '2025-07'. Omit for the current month.",
                },
            },
            "required": ["student_name"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_marks",
        "description": "Exam marks and grades for one student. Optionally filter by exam or subject.",
        "fn": get_marks,
        "allowed_for": {"parent", "teacher", "admin"},
        "input_schema": {
            "type": "object",
            "properties": {
                "student_name": {"type": "string"},
                "exam_name": {
                    "type": "string",
                    "description": "e.g. 'Term 1', 'Mid Term'.",
                },
                "subject": {"type": "string", "description": "e.g. 'Maths', 'Science'."},
            },
            "required": ["student_name"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_fee_status",
        "description": "Fee invoices, amounts paid, outstanding balance and overdue status for one student.",
        "fn": get_fee_status,
        "allowed_for": {"parent", "admin"},
        "input_schema": {
            "type": "object",
            "properties": {"student_name": {"type": "string"}},
            "required": ["student_name"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_homework",
        "description": "Homework assigned to a class in the last N days.",
        "fn": get_homework,
        "allowed_for": {"parent", "teacher", "admin"},
        "input_schema": {
            "type": "object",
            "properties": {
                "class_name": {
                    "type": "string",
                    "description": "e.g. '7B', 'Grade 7-B'.",
                },
                "days": {
                    "type": "integer",
                    "description": "Look-back window in days. Default 7.",
                },
                "branch": BRANCH_ARG,
            },
            "required": ["class_name"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_timetable",
        "description": "Class timetable, optionally for one weekday.",
        "fn": get_timetable,
        "allowed_for": {"parent", "teacher", "admin"},
        "input_schema": {
            "type": "object",
            "properties": {
                "class_name": {"type": "string"},
                "weekday": {
                    "type": "string",
                    "description": "e.g. 'Monday'. Omit for the full week.",
                },
                "branch": BRANCH_ARG,
            },
            "required": ["class_name"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_class_attendance_report",
        "description": "Whole-class attendance report with a list of students below 75%. Staff only.",
        "fn": get_class_attendance_report,
        "allowed_for": {"teacher", "admin"},
        "input_schema": {
            "type": "object",
            "properties": {
                "class_name": {"type": "string"},
                "month": {"type": "string"},
                "branch": BRANCH_ARG,
            },
            "required": ["class_name"],
            "additionalProperties": False,
        },
    },
    {
        "name": "search_school_documents",
        "description": (
            "Search school policy documents, circulars and the handbook. Use for questions about "
            "RULES, POLICIES, PROCEDURES, holidays or events -- anything not stored as student data."
        ),
        "fn": search_school_documents,
        "allowed_for": {"parent", "teacher", "admin"},
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The user's question, rephrased for semantic search.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
]


def _allow_null_on_optional(tools: list[dict]) -> list[dict]:
    """
    Every OPTIONAL parameter must also accept null. This is not cosmetic.

    Models fill unused optional arguments with null constantly -- it is their
    natural way of saying "not applicable". Some providers (Groq, notably)
    validate the model's tool call against your schema SERVER-SIDE and reject
    the whole request if a field typed as a bare "string" arrives as null:

        400 tool_use_failed - `/board`: expected string, but got null

    You cannot stop the model doing it, so accept it: declare optional
    properties as ["string", "null"], then strip the nulls in execute() so the
    Python default applies. Done here in a loop rather than by hand on each
    schema, so tools you add later are covered automatically.
    """
    for t in tools:
        schema = t["input_schema"]
        required = set(schema.get("required", []))
        for name, prop in schema["properties"].items():
            ty = prop.get("type")
            # isinstance guard also makes this idempotent, which matters because
            # BRANCH_ARG is one dict shared by three tools.
            if name not in required and isinstance(ty, str) and ty != "null":
                prop["type"] = [ty, "null"]
    return tools


TOOLS = _allow_null_on_optional(TOOLS)


def tools_for_role(role: str) -> list[dict]:
    """Only advertise tools this role may call -- the model can't misuse what it can't see."""
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
        # Filter to declared keys in case the model invents an extra argument,
        # and drop nulls so the function's own default applies. Passing
        # days=None straight through would blow up in timedelta(days=None).
        allowed = set(spec["input_schema"]["properties"])
        clean = {k: v for k, v in args.items() if k in allowed and v is not None}
        return spec["fn"](**clean)
    except TypeError as e:
        return {"error": "bad_arguments", "message": str(e)}
    except Exception as e:  # never let a tool crash the whole request
        return {"error": "tool_failed", "message": f"{type(e).__name__}: {e}"}
