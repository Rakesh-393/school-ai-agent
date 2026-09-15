"""
Database schema for the school.

Deliberately small but realistic: the agent's tools map 1:1 onto these tables.
When you add a table, you add a tool -- that is the whole extension story.
"""
from datetime import date, datetime

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import UnicodeText
from sqlalchemy.dialects import mssql

db = SQLAlchemy()

# Long free text: an essay-length homework note, a full LLM answer.
#
# The variant is not decoration. SQLAlchemy's UnicodeText becomes NTEXT on SQL
# Server, and NTEXT has been deprecated since 2005 and is announced for removal.
# Worse, it is not a comparable type: an NTEXT column cannot appear in =, GROUP
# BY, DISTINCT or an index, so the first query that filters on a chat log fails
# with a type error rather than a missing feature. NVARCHAR(max) is the modern
# replacement and behaves like an ordinary string column.
# Everywhere else -- SQLite, MySQL, Postgres -- the variant is ignored.
LONG_TEXT = UnicodeText().with_variant(mssql.NVARCHAR(None), "mssql")

# Columns are declared Unicode/UnicodeText rather than String/Text on purpose.
#
# On SQLite, MySQL and Postgres the two are the same thing. On SQL Server they
# are not: String becomes VARCHAR and Unicode becomes NVARCHAR. VARCHAR stores
# one byte per character in the database's collation codepage, so anything
# outside it is silently replaced with "?" on write -- and this app stores
# exactly the text that breaks: LLM answers full of curly quotes and en dashes,
# and Telugu/Devanagari student and branch names. There is no error and no way
# back once it is written.


class Branch(db.Model):
    """
    One campus of the school group.

    `aliases` is the interesting column: a pipe-separated list of every other
    way a human might name this branch -- the locality, the informal short name,
    the old name. The fuzzy resolver scores the query against ALL of them and
    keeps the best, so "jyothinagar", "explorica" and "cambridge branch" all
    land on the same row. See resolver.resolve(alias_attr=...).
    """
    __tablename__ = "branches"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.Unicode(120), nullable=False, unique=True, index=True)
    code = db.Column(db.Unicode(15), unique=True)          # "PHS", "PHER"
    locality = db.Column(db.Unicode(80))                   # "Mankammathota"
    city = db.Column(db.Unicode(60), default="Karimnagar")
    state = db.Column(db.Unicode(60), default="Telangana")
    board = db.Column(db.Unicode(40))                      # State (SSC) | CBSE | Cambridge (CIE)
    grades = db.Column(db.Unicode(60))                     # "Pre-K to Grade 10"
    address = db.Column(db.Unicode(240))
    phone = db.Column(db.Unicode(60))
    email = db.Column(db.Unicode(120))
    website = db.Column(db.Unicode(120))
    principal = db.Column(db.Unicode(120))
    established = db.Column(db.Unicode(30))
    aliases = db.Column(db.Unicode(300), default="")       # "explorica|jyothinagar|cambridge"

    sections = db.relationship("ClassSection", back_populates="branch")

    @property
    def alias_list(self) -> list[str]:
        """Every string a user might type for this branch, including the real name."""
        parts = [self.name, self.code, self.locality]
        parts += [a.strip() for a in (self.aliases or "").split("|")]
        return [p for p in parts if p]


class ClassSection(db.Model):
    """e.g. Grade 7 - B, at one branch."""
    __tablename__ = "class_sections"
    id = db.Column(db.Integer, primary_key=True)
    branch_id = db.Column(db.Integer, db.ForeignKey("branches.id"), nullable=False)
    grade = db.Column(db.Unicode(20), nullable=False)      # "7"
    section = db.Column(db.Unicode(10), nullable=False)    # "B"
    class_teacher = db.Column(db.Unicode(80))
    room = db.Column(db.Unicode(20))

    branch = db.relationship("Branch", back_populates="sections")
    students = db.relationship("Student", back_populates="klass")

    @property
    def label(self) -> str:
        return f"Grade {self.grade}-{self.section}"

    @property
    def full_label(self) -> str:
        return f"Grade {self.grade}-{self.section} ({self.branch.name})" if self.branch else self.label


class Student(db.Model):
    __tablename__ = "students"
    id = db.Column(db.Integer, primary_key=True)
    admission_no = db.Column(db.Unicode(20), unique=True, nullable=False)
    full_name = db.Column(db.Unicode(120), nullable=False, index=True)
    roll_no = db.Column(db.Integer)
    date_of_birth = db.Column(db.Date)
    gender = db.Column(db.Unicode(10))
    class_section_id = db.Column(db.Integer, db.ForeignKey("class_sections.id"))
    guardian_name = db.Column(db.Unicode(120))
    guardian_phone = db.Column(db.Unicode(20))
    guardian_email = db.Column(db.Unicode(120))
    bus_route = db.Column(db.Unicode(40))

    klass = db.relationship("ClassSection", back_populates="students")
    attendance = db.relationship("Attendance", back_populates="student")
    marks = db.relationship("Mark", back_populates="student")
    invoices = db.relationship("FeeInvoice", back_populates="student")


class Subject(db.Model):
    __tablename__ = "subjects"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.Unicode(60), nullable=False, index=True)   # "Mathematics"
    code = db.Column(db.Unicode(15), unique=True)                  # "MATH"
    teacher = db.Column(db.Unicode(80))


class Attendance(db.Model):
    __tablename__ = "attendance"
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey("students.id"), nullable=False)
    day = db.Column(db.Date, nullable=False, index=True)
    status = db.Column(db.Unicode(10), nullable=False)   # present | absent | late | leave
    remark = db.Column(db.Unicode(200))

    student = db.relationship("Student", back_populates="attendance")
    __table_args__ = (db.UniqueConstraint("student_id", "day", name="uq_att_student_day"),)


class Exam(db.Model):
    __tablename__ = "exams"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.Unicode(60), nullable=False)     # "Term 1"
    academic_year = db.Column(db.Unicode(15), default="2025-26")
    starts_on = db.Column(db.Date)


class Mark(db.Model):
    __tablename__ = "marks"
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey("students.id"), nullable=False)
    subject_id = db.Column(db.Integer, db.ForeignKey("subjects.id"), nullable=False)
    exam_id = db.Column(db.Integer, db.ForeignKey("exams.id"), nullable=False)
    scored = db.Column(db.Float, nullable=False)
    max_marks = db.Column(db.Float, default=100)
    grade = db.Column(db.Unicode(5))

    student = db.relationship("Student", back_populates="marks")
    subject = db.relationship("Subject")
    exam = db.relationship("Exam")


class FeeInvoice(db.Model):
    __tablename__ = "fee_invoices"
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey("students.id"), nullable=False)
    term = db.Column(db.Unicode(30), nullable=False)      # "Term 1 2025-26"
    head = db.Column(db.Unicode(40), default="Tuition")   # Tuition | Transport | Lab
    amount = db.Column(db.Float, nullable=False)
    paid = db.Column(db.Float, default=0.0)
    due_date = db.Column(db.Date)

    student = db.relationship("Student", back_populates="invoices")

    @property
    def balance(self) -> float:
        return round(self.amount - self.paid, 2)


class Homework(db.Model):
    __tablename__ = "homework"
    id = db.Column(db.Integer, primary_key=True)
    class_section_id = db.Column(db.Integer, db.ForeignKey("class_sections.id"), nullable=False)
    subject_id = db.Column(db.Integer, db.ForeignKey("subjects.id"), nullable=False)
    assigned_on = db.Column(db.Date, default=date.today, index=True)
    due_on = db.Column(db.Date)
    details = db.Column(LONG_TEXT, nullable=False)

    klass = db.relationship("ClassSection")
    subject = db.relationship("Subject")


class TimetableSlot(db.Model):
    __tablename__ = "timetable_slots"
    id = db.Column(db.Integer, primary_key=True)
    class_section_id = db.Column(db.Integer, db.ForeignKey("class_sections.id"), nullable=False)
    weekday = db.Column(db.Unicode(10), nullable=False)   # Monday..Saturday
    period = db.Column(db.Integer, nullable=False)
    subject_id = db.Column(db.Integer, db.ForeignKey("subjects.id"))
    starts_at = db.Column(db.Unicode(10))                 # "09:00"

    klass = db.relationship("ClassSection")
    subject = db.relationship("Subject")


class ChatLog(db.Model):
    """Every turn is stored: your eval set, your audit trail, your debugging tool."""

    # The only table this app writes. It lives on the "logs" bind, which points
    # at the main database when the app owns it and at a local SQLite file when
    # it does not -- see dburi.log_store_uri().
    __bind_key__ = "logs"
    __tablename__ = "chat_logs"
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Unicode(64), index=True)
    role_context = db.Column(db.Unicode(20))    # parent | teacher | admin
    question = db.Column(LONG_TEXT)
    answer = db.Column(LONG_TEXT)
    tools_used = db.Column(db.Unicode(300))
    latency_ms = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
