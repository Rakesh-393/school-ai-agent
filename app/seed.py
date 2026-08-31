"""
Demo data for Paramita Schools.

Deliberately includes two students with very similar names
("Aarav Sharma" and "Aaravi Sharma") so you can watch the fuzzy resolver
correctly refuse to guess and ask a clarifying question instead.
"""
import random
from datetime import date, timedelta

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

random.seed(7)

# Everything is anchored to TODAY, never to hardcoded calendar dates.
# Hardcoding "July 2025" means the demo silently returns "no records" the
# moment the real year rolls over -- a bug you will hit in every seeded
# project you ever write.
TODAY = date.today()
AY_START_YEAR = TODAY.year if TODAY.month >= 4 else TODAY.year - 1   # Indian AY: Apr-Mar
AY = f"{AY_START_YEAR}-{str(AY_START_YEAR + 1)[2:]}"                 # e.g. "2026-27"
ATTENDANCE_DAYS = 120                                               # ~4 months back from today

# The five campuses, from paramitaschools.in and the individual branch sites.
# `aliases` is the important field -- every other way a parent might name the
# branch. Add to it whenever a real user's phrasing fails to resolve; that is
# cheaper and far more reliable than loosening the fuzzy thresholds.
BRANCHES = [
    dict(
        name="Paramita High School", code="PHS", locality="Mankammathota",
        board="State (TS SSC)", grades="Pre-K to Grade 10",
        address="Mankammathota, Karimnagar, Telangana - 505001",
        phone="+91-8886127373 / +91-9885136655",
        email="info@paramitaschools.in", website="paramitahighschool.in",
        established="18 June 1996",
        aliases="paramita high|high school|mankammathota|ssc branch|state board branch|main branch",
    ),
    dict(
        name="Paramita School, Ponnam Complex", code="PPC", locality="Ponnam Complex",
        board="State (TS SSC)", grades="Pre-K to Grade 10",
        address="Ponnam Complex, Karimnagar, Telangana - 505001",
        phone="+91-8886127373 / +91-9885136655",
        email="info@paramitaschools.in", website="paramitaschools.in",
        aliases="ponnam|ponnam complex|paramita ponnam|state board",
    ),
    dict(
        name="Paramita Heritage School", code="PHER", locality="Padmanagar",
        board="CBSE", grades="Grade 1 to Grade 10",
        address="Vemulawada Road, Beside Walmart, Padmanagar, Karimnagar - 505001",
        phone="+91-8886127373", email="info@paramitaschools.in",
        website="paramitaheritageschool.in",
        principal="Parankusham Gopikrishna (M.Sc, B.Ed.)",
        established="2 June 2005",
        aliases="heritage|paramita heritage|padmanagar|cbse branch|walmart branch|xseed",
    ),
    dict(
        name="Paramita World School", code="PWS", locality="Alugunoor",
        board="CBSE", grades="Grade 1 to Grade 10",
        address="#5-13/5/1, Hyderabad Highway Road, Alugunoor, Karimnagar, Telangana - 505001",
        phone="+91-8886127373 / +91-9885136655",
        email="info@paramitaschools.in", website="paramitaworldschool.in",
        aliases="world school|iris|iris world school|alugunoor|highway branch",
    ),
    dict(
        name="Explorica Premium School", code="EXP", locality="Jyothinagar",
        board="Cambridge (CIE)", grades="Toddlers to Grade 7",
        address="2-10-1473, Jyothi Nagar, Karimnagar, Telangana - 505001",
        phone="+91 91609 36655", email="contact@explorica.in", website="explorica.in",
        aliases="explorica|jyothinagar|jyothi nagar|cambridge|cambridge branch|premium school",
    ),
]

SUBJECTS = [
    ("Mathematics", "MATH", "Mrs. Priya Nair"),
    ("Science", "SCI", "Mr. Rohan Das"),
    ("English", "ENG", "Ms. Fatima Sheikh"),
    ("Social Studies", "SST", "Mr. Vikram Rao"),
    ("Hindi", "HIN", "Mrs. Sunita Verma"),
    ("Computer Science", "CS", "Mr. Arjun Pillai"),
]

NAMES = [
    # The first two are deliberately near-identical: this is the pair that proves
    # the resolver refuses to guess between two real children.
    "Aarav Sharma", "Aaravi Sharma",
    "Diya Patel", "Kabir Menon", "Ishaan Reddy", "Ananya Iyer", "Vihaan Gupta",
    "Meera Krishnan", "Rohan Joshi", "Saanvi Bose", "Arjun Nair", "Zara Khan",
    "Advait Kulkarni", "Riya Chatterjee", "Neel Bhatt", "Tara Desai",
    "Kiaan Malhotra", "Anika Roy", "Dev Agarwal", "Myra Sinha",
    "Aditya Rao", "Ira Bhandari", "Yash Deshmukh", "Nitara Ghosh", "Reyansh Pillai",
    "Kyra Saxena", "Ayaan Mirza", "Prisha Kaur", "Vivaan Shetty", "Amaira Jain",
    "Shaurya Chauhan", "Navya Kapoor", "Atharv Naik", "Sara Thomas", "Rudra Yadav",
    "Aadhya Mishra", "Krish Bansal", "Pari Sethi", "Veer Solanki", "Kiara Dutta",
]

# Which grade+section pairs each branch runs. Grade 7-A exists at every campus
# on purpose -- that collision is what makes `_find_class` ask which branch.
SECTIONS_BY_BRANCH = {
    "PHS":  [("7", "A"), ("7", "B"), ("8", "A")],
    "PPC":  [("7", "A"), ("8", "A")],
    "PHER": [("7", "A"), ("7", "B"), ("8", "A")],
    "PWS":  [("7", "A"), ("7", "B")],
    "EXP":  [("6", "A"), ("7", "A")],
}


def seed() -> dict:
    if Student.query.first():
        return {"skipped": "database already seeded -- use --reset to rebuild"}

    # --- branches ---
    branches = [Branch(**b) for b in BRANCHES]
    db.session.add_all(branches)
    db.session.flush()
    by_code = {b.code: b for b in branches}

    # --- classes, per branch ---
    teachers = ["Mrs. Priya Nair", "Mr. Rohan Das", "Ms. Fatima Sheikh",
                "Mr. Vikram Rao", "Mrs. Sunita Verma", "Mr. Arjun Pillai"]
    sections = []
    for code, pairs in SECTIONS_BY_BRANCH.items():
        for n, (grade, sec) in enumerate(pairs):
            sections.append(ClassSection(
                branch_id=by_code[code].id, grade=grade, section=sec,
                class_teacher=teachers[(len(sections) + n) % len(teachers)],
                room=f"{code}-{grade}{sec}",
            ))
    db.session.add_all(sections)

    subjects = [Subject(name=n, code=c, teacher=t) for n, c, t in SUBJECTS]
    db.session.add_all(subjects)

    exams = [
        Exam(name="Unit Test 1", academic_year=AY, starts_on=TODAY - timedelta(days=95)),
        Exam(name="Term 1", academic_year=AY, starts_on=TODAY - timedelta(days=35)),
    ]
    db.session.add_all(exams)
    db.session.flush()

    # --- students ---
    students = []
    for i, name in enumerate(NAMES):
        sec = sections[i % len(sections)]
        s = Student(
            admission_no=f"PAR{AY_START_YEAR}{i + 1:03d}",
            full_name=name,
            roll_no=(i // len(sections)) + 1,
            date_of_birth=date(2012, (i % 12) + 1, (i % 27) + 1),
            gender="F" if i % 3 == 0 else "M",
            class_section_id=sec.id,
            guardian_name=f"{name.split()[1]} (parent)",
            guardian_phone=f"+91 98{random.randint(10000000, 99999999)}",
            guardian_email=f"{name.split()[0].lower()}.parent@example.com",
            bus_route=f"Route {(i % 5) + 1}",
        )
        students.append(s)
    db.session.add_all(students)
    db.session.flush()

    # --- attendance: the last ATTENDANCE_DAYS calendar days, weekdays only ---
    day = TODAY - timedelta(days=ATTENDANCE_DAYS)
    end = TODAY
    att = []
    while day <= end:
        if day.weekday() < 5:  # Mon-Fri
            for s in students:
                # one student deliberately pushed below 75% so the flag fires
                absent_rate = 0.30 if s.full_name == "Kabir Menon" else 0.06
                roll = random.random()
                status = "absent" if roll < absent_rate else ("late" if roll < absent_rate + 0.05 else "present")
                att.append(Attendance(student_id=s.id, day=day, status=status))
        day += timedelta(days=1)
    db.session.bulk_save_objects(att)

    # --- marks ---
    marks = []
    for s in students:
        for e in exams:
            for sub in subjects:
                scored = round(random.uniform(48, 98), 1)
                grade = ("A+" if scored >= 90 else "A" if scored >= 80 else
                         "B" if scored >= 70 else "C" if scored >= 60 else "D")
                marks.append(Mark(student_id=s.id, subject_id=sub.id, exam_id=e.id,
                                  scored=scored, max_marks=100, grade=grade))
    db.session.bulk_save_objects(marks)

    # --- fees ---
    invoices = []
    term1_due = date(AY_START_YEAR, 6, 15)          # already past -> some are overdue
    term2_due = date(AY_START_YEAR, 11, 15)         # still upcoming -> "pending"
    for s in students:
        invoices.append(FeeInvoice(student_id=s.id, term=f"Term 1 {AY}", head="Tuition",
                                   amount=45000, paid=random.choice([45000, 45000, 22500]),
                                   due_date=term1_due))
        invoices.append(FeeInvoice(student_id=s.id, term=f"Term 2 {AY}", head="Tuition",
                                   amount=45000, paid=random.choice([0, 0, 22500, 45000]),
                                   due_date=term2_due))
        invoices.append(FeeInvoice(student_id=s.id, term=f"Term 2 {AY}", head="Transport",
                                   amount=8500, paid=random.choice([0, 8500]),
                                   due_date=term2_due))
    db.session.bulk_save_objects(invoices)

    # --- homework (recent, so the default 7-day window finds it) ---
    hw = []
    today = TODAY
    for sec in sections:
        for n, sub in enumerate(subjects[:4]):
            hw.append(Homework(
                class_section_id=sec.id, subject_id=sub.id,
                assigned_on=today - timedelta(days=n),
                due_on=today + timedelta(days=2 + n),
                details=f"{sub.name}: complete exercise set {n + 3} and revise the chapter summary.",
            ))
    db.session.bulk_save_objects(hw)

    # --- timetable ---
    slots = []
    times = ["09:00", "09:45", "10:30", "11:30", "12:15", "13:30", "14:15"]
    for sec in sections:
        for weekday in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
            for p, t in enumerate(times, start=1):
                slots.append(TimetableSlot(
                    class_section_id=sec.id, weekday=weekday, period=p, starts_at=t,
                    subject_id=subjects[(p + len(weekday)) % len(subjects)].id,
                ))
    db.session.bulk_save_objects(slots)

    db.session.commit()
    return {
        "branches": len(branches),
        "classes": len(sections),
        "students": len(students),
        "attendance_rows": len(att),
        "marks": len(marks),
        "invoices": len(invoices),
        "homework": len(hw),
        "timetable_slots": len(slots),
    }
