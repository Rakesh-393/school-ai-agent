"""
Exercise the tools and the fuzzy resolver WITHOUT calling any LLM.

Run this first, and every time you add a tool. If a tool is wrong here, no
amount of prompt engineering upstream will save you.

    .venv\\Scripts\\python.exe scripts\\test_tools.py
"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from app.agent import tools as toolkit
from app.agent.resolver import resolve
from app.models import Student


def show(title, payload):
    print(f"\n--- {title} ---")
    print(json.dumps(payload, indent=2, default=str)[:900])


app = create_app()
with app.app_context():

    # ============================================================
    # 0. BRANCHES + ALIAS MATCHING
    # ============================================================
    print("=" * 62)
    print("BRANCHES  (alias-aware fuzzy matching)")
    print("=" * 62)

    r = toolkit.execute("list_branches", {}, "parent")
    print(f"  total_branches = {r['total_branches']}")
    for b in r["branches"]:
        print(f"    {b['code']:<5} {b['name']:<34} {b['board']:<17} {b['students']:>2} students")

    print("\n  filtered by board 'CBSE':")
    for b in toolkit.execute("list_branches", {"board": "cbse"}, "parent")["branches"]:
        print(f"    {b['name']}")

    print("\n  one row, many names -- every probe below must land on the right campus:")
    for probe in ["explorica", "jyothinagar", "the cambridge one", "heritage",
                  "padmanagar", "alugunoor", "iris", "ponnam", "PHS", "Hogwarts"]:
        res = toolkit.execute("get_branch_info", {"branch_name": probe}, "parent")
        print(f"    {probe:<20} -> {res.get('name') or res['error']}")

    show("class that exists at 5 campuses -> agent must ask which",
         toolkit.execute("get_homework", {"class_name": "7A"}, "parent"))

    show("same class, scoped to one campus",
         toolkit.execute("get_homework", {"class_name": "7A", "branch": "explorica"}, "parent"))

    # ============================================================
    # 1. FUZZY MATCHING in isolation
    # ============================================================
    students = Student.query.all()
    print("=" * 62)
    print("FUZZY RESOLVER  (rapidfuzz WRatio + token_set_ratio)")
    print("=" * 62)

    for probe in [
        "Aarav Sharma",   # exact          -> matched
        "arav sharma",    # typo           -> matched
        "sharma aarav",   # wrong order    -> token_set handles it
        "diya",           # partial        -> matched
        "aarav",          # AMBIGUOUS: Aarav Sharma vs Aaravi Sharma
        "kabirr menonn",  # double typos   -> matched
        "Zaphod Beeblebrox",  # nonsense   -> not_found
    ]:
        r = resolve(
            probe,
            students,
            label_attr="full_name",
            extra_fn=lambda s: {"class": s.klass.label, "branch": s.klass.branch.code},
        )
        top = ", ".join(f"{c.label}={c.score:.0f}" for c in r.candidates[:3])
        print(f"  {probe:<20} -> {r.status:<10} [{top}]")

    # ============================================================
    # 2. TOOLS
    # ============================================================
    print("\n" + "=" * 62)
    print("TOOLS")
    print("=" * 62)

    show("attendance, typo'd name", toolkit.execute(
        "get_attendance_summary", {"student_name": "kabirr menon", "month": "July"}, "parent"))

    show("ambiguous name -> agent must ask", toolkit.execute(
        "get_attendance_summary", {"student_name": "aarav"}, "parent"))

    show("marks filtered by subject", toolkit.execute(
        "get_marks", {"student_name": "Diya Patel", "exam_name": "Term 1", "subject": "maths"}, "parent"))

    # Admission numbers bypass fuzzy matching entirely -- an ID is exact by definition.
    from app.models import Student as _S
    adm = _S.query.filter_by(full_name="Kabir Menon").first().admission_no
    show(f"fees, looked up by admission no {adm}",
         toolkit.execute("get_fee_status", {"student_name": adm}, "parent"))

    show("ACCESS CONTROL: parent blocked from class report", toolkit.execute(
        "get_class_attendance_report", {"class_name": "7A", "branch": "PHS"}, "parent"))

    show("same tool, teacher allowed", toolkit.execute(
        "get_class_attendance_report",
        {"class_name": "7A", "branch": "PHS", "month": "July"}, "teacher"))

    show("RAG", toolkit.execute(
        "search_school_documents", {"query": "how do I apply for leave"}, "parent"))

    # ============================================================
    # 3. WHAT THE LLM ACTUALLY SEES
    # ============================================================
    print("\n" + "=" * 62)
    print("TOOL VISIBILITY BY ROLE")
    print("=" * 62)
    for role in ("parent", "teacher", "admin"):
        names = [t["name"] for t in toolkit.tools_for_role(role)]
        print(f"  {role:<8} ({len(names)}): {', '.join(names)}")
