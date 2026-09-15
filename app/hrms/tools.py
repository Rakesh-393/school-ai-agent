"""
HRMS TOOL REGISTRY -- what the agent can ask the HR database.

Same contract as the school toolkit in app/agent/tools.py: the model never
writes SQL, it picks a function name and fills in typed arguments, and every
query here is hand-written and parameterised.

Two rules specific to this toolkit, because it reads a database the agent does
not own and that holds real people's records:

  1. READ ONLY. Every statement is a SELECT. There is no insert, no update, no
     DDL, and nothing here should ever grow one. Ask for a read-only SQL login
     as well; code discipline is the second line of defence, not the first.

  2. AGGREGATES, NOT PEOPLE. Employee has 64 columns including salary, Aadhaar,
     PAN, date of birth and home address. The tools here count and group; none
     returns an individual's row. Adding a per-person lookup is a real feature
     with a real access-control question attached, not a small extension.

Status = 1 means active in every master table, so every query filters on it.
Counts come back computed in SQL -- never a list for the model to count, which
is the one arithmetic task language models are reliably bad at.
"""
from __future__ import annotations

from sqlalchemy import func, select

from app.hrms import schema as s
from app.hrms.links import RESOURCES, resource_link  # noqa: F401  (re-exported)
from app.models import db


def _rows(stmt) -> list[dict]:
    """Run a select and return plain dicts, ready to hand to the model as JSON."""
    return [dict(r._mapping) for r in db.session.execute(stmt)]


def _active(table):
    return select(table).where(table.c.Status == s.ACTIVE)


# ------------------------------------------------------------------ org / geo
def get_organisation_info() -> dict:
    """Who this HRMS belongs to: name, code, address, city/state/country."""
    stmt = (
        select(
            s.Organisations.c.OrganisationName.label("organisation"),
            s.Organisations.c.OrganisationCode.label("code"),
            s.Organisations.c.OrganisationAddress.label("address"),
            s.Cities.c.CityName.label("city"),
            s.States.c.StateName.label("state"),
            s.Countries.c.CountryName.label("country"),
        )
        .select_from(
            s.Organisations
            .outerjoin(s.Cities, s.Cities.c.CityID == s.Organisations.c.CityID)
            .outerjoin(s.States, s.States.c.StateID == s.Organisations.c.StateID)
            .outerjoin(s.Countries, s.Countries.c.CountryID == s.Organisations.c.CountryID)
        )
        .where(s.Organisations.c.Status == s.ACTIVE)
    )
    rows = _rows(stmt)
    if not rows:
        return {"error": "no_data", "message": "No active organisation is configured."}
    return {"total_organisations": len(rows), "organisations": rows}


def list_units(country: str | None = None, state: str | None = None) -> dict:
    """
    The branches / campuses / units, with where each one is.

    "How many branches do you have" is the single most common question, so the
    count is computed here and returned as a number. See the note in the school
    toolkit's list_branches: never let the model count the list itself.
    """
    stmt = (
        select(
            s.Units.c.UnitID.label("id"),
            s.Units.c.UnitName.label("unit"),
            s.Units.c.UnitCode.label("code"),
            s.Units.c.UnitAddress.label("address"),
            s.Cities.c.CityName.label("city"),
            s.States.c.StateName.label("state"),
            s.Countries.c.CountryName.label("country"),
        )
        .select_from(
            s.Units
            .outerjoin(s.Cities, s.Cities.c.CityID == s.Units.c.CityID)
            .outerjoin(s.States, s.States.c.StateID == s.Units.c.StateID)
            .outerjoin(s.Countries, s.Countries.c.CountryID == s.Units.c.CountryID)
        )
        .where(s.Units.c.Status == s.ACTIVE)
        .order_by(s.Units.c.UnitName)
    )
    if country:
        stmt = stmt.where(func.lower(s.Countries.c.CountryName) == country.strip().lower())
    if state:
        stmt = stmt.where(func.lower(s.States.c.StateName) == state.strip().lower())

    units = _rows(stmt)
    return {
        "total_units": len(units),
        "filtered_by": {"country": country, "state": state},
        "units": units,
    }


def list_sub_units(unit: str | None = None) -> dict:
    """Departments / sub-units, optionally only those under one unit."""
    stmt = (
        select(
            s.SubUnits.c.SubUnitName.label("sub_unit"),
            s.SubUnits.c.SubUnitCode.label("code"),
            s.Units.c.UnitName.label("unit"),
        )
        .select_from(s.SubUnits.outerjoin(s.Units, s.Units.c.UnitID == s.SubUnits.c.UnitID))
        .where(s.SubUnits.c.Status == s.ACTIVE)
        .order_by(s.Units.c.UnitName, s.SubUnits.c.SubUnitName)
    )
    if unit:
        stmt = stmt.where(func.lower(s.Units.c.UnitName).like(f"%{unit.strip().lower()}%"))
    rows = _rows(stmt)
    return {"total_sub_units": len(rows), "filtered_by_unit": unit, "sub_units": rows}


def list_locations() -> dict:
    """Every country, state and city the organisation has a presence in."""
    stmt = (
        select(
            s.Locations.c.LocationName.label("location"),
            s.Cities.c.CityName.label("city"),
            s.States.c.StateName.label("state"),
            s.Countries.c.CountryName.label("country"),
        )
        .select_from(
            s.Locations
            .outerjoin(s.Cities, s.Cities.c.CityID == s.Locations.c.CityID)
            .outerjoin(s.States, s.States.c.StateID == s.Locations.c.StateID)
            .outerjoin(s.Countries, s.Countries.c.CountryID == s.Locations.c.CountryID)
        )
        .where(s.Locations.c.Status == s.ACTIVE)
        .order_by(s.Countries.c.CountryName, s.States.c.StateName)
    )
    rows = _rows(stmt)
    return {
        "total_locations": len(rows),
        "countries": sorted({r["country"] for r in rows if r["country"]}),
        "locations": rows,
    }


# ------------------------------------------------------------------ academics
def list_subjects() -> dict:
    """Subjects configured in the system."""
    stmt = (
        select(s.Subjects.c.SubjectName.label("subject"), s.Subjects.c.SubjectCode.label("code"))
        .where(s.Subjects.c.Status == s.ACTIVE)
        .order_by(s.Subjects.c.SubjectName)
    )
    rows = _rows(stmt)
    return {"total_subjects": len(rows), "subjects": rows}


def list_grades() -> dict:
    """Grades / classes, with the stream and NEP stage each belongs to."""
    stmt = (
        select(
            s.Grades.c.GradeName.label("grade"),
            s.Grades.c.GradeCode.label("code"),
            s.Streams.c.StreamName.label("stream"),
            s.NEPStages.c.StageName.label("nep_stage"),
        )
        .select_from(
            s.Grades
            .outerjoin(s.Streams, s.Streams.c.StreamID == s.Grades.c.StreamID)
            .outerjoin(s.NEPStages, s.NEPStages.c.StageID == s.Grades.c.NEPStageID)
        )
        .where(s.Grades.c.Status == s.ACTIVE)
        .order_by(s.Grades.c.GradeID)
    )
    rows = _rows(stmt)
    return {"total_grades": len(rows), "grades": rows}


def list_academic_years() -> dict:
    """Academic years and their start and end dates."""
    stmt = (
        select(
            s.AcademicYears.c.AcademicYearName.label("academic_year"),
            s.AcademicYears.c.StartDate.label("starts_on"),
            s.AcademicYears.c.EndDate.label("ends_on"),
        )
        .where(s.AcademicYears.c.Status == s.ACTIVE)
        .order_by(s.AcademicYears.c.StartDate)
    )
    rows = _rows(stmt)
    for r in rows:
        for k in ("starts_on", "ends_on"):
            r[k] = str(r[k].date()) if r[k] else None
    return {"total_academic_years": len(rows), "academic_years": rows}


# ------------------------------------------------------------------ org design
def list_designations(category: str | None = None, teaching_only: bool | None = None) -> dict:
    """
    Job titles, with the function and staff category each sits in.

    `teaching_only` is a separate argument rather than a category name on
    purpose. "Teaching" looks like a category but is not one: the categories are
    things like 'Grade 06 & Grade 08' and 'House Keeping', and whether they are
    academic is the IsTeaching flag on Categories. Asking for category="teaching"
    matched nothing at all and returned a confident empty list.
    """
    stmt = (
        select(
            s.Designations.c.DesignationName.label("designation"),
            s.Designations.c.DesignationCode.label("code"),
            s.Functionalities.c.FunctionName.label("function"),
            s.Categories.c.CategoryName.label("category"),
        )
        .select_from(
            s.Designations
            .outerjoin(s.Functionalities,
                       s.Functionalities.c.FunctionID == s.Designations.c.FunctionID)
            .outerjoin(s.Categories, s.Categories.c.CategoryID == s.Designations.c.CategoryID)
        )
        .where(s.Designations.c.Status == s.ACTIVE)
        .order_by(s.Designations.c.DesignationName)
    )
    if teaching_only is not None:
        stmt = stmt.where(s.Categories.c.IsTeaching == (1 if teaching_only else 0))
    if category:
        stmt = stmt.where(func.lower(s.Categories.c.CategoryName).like(f"%{category.strip().lower()}%"))

    rows = _rows(stmt)
    out = {
        "total_designations": len(rows),
        "filtered_by": {"category": category, "teaching_only": teaching_only},
        "designations": rows,
    }
    if category and not rows:
        # An empty list reads as "there are none", which is a different claim
        # from "that is not one of the categories". Say which, and show the real
        # ones so the agent can offer them instead of guessing again.
        known = _rows(
            select(s.Categories.c.CategoryName.label("category"))
            .where(s.Categories.c.Status == s.ACTIVE)
            .order_by(s.Categories.c.CategoryName)
        )
        out["message"] = f"No staff category matches {category!r}."
        out["available_categories"] = [k["category"] for k in known]
        out["hint"] = "For academic roles use teaching_only=true, not a category name."
    return out


def list_departments() -> dict:
    """Functions and staff categories -- the top level of the org structure."""
    funcs = _rows(
        select(s.Functionalities.c.FunctionName.label("function"),
               s.Functionalities.c.Description.label("description"))
        .where(s.Functionalities.c.Status == s.ACTIVE)
        .order_by(s.Functionalities.c.FunctionName)
    )
    cats = _rows(
        select(s.Categories.c.CategoryName.label("category"),
               s.Categories.c.IsTeaching.label("is_teaching"))
        .where(s.Categories.c.Status == s.ACTIVE)
        .order_by(s.Categories.c.CategoryName)
    )
    for c in cats:
        c["is_teaching"] = bool(c["is_teaching"])
    return {"functions": funcs, "staff_categories": cats}


# ------------------------------------------------------------------ headcount
def get_headcount(group_by: str | None = None) -> dict:
    """
    How many active employees, optionally broken down.

    Counts only. This deliberately cannot return who they are; see the rule at
    the top of this module.
    """
    total = db.session.execute(
        select(func.count()).select_from(s.Employee).where(s.Employee.c.Status == s.ACTIVE)
    ).scalar_one()

    out: dict = {"total_employees": total}

    dimensions = {
        "unit": (s.Units.c.UnitName, s.Units, s.Units.c.UnitID == s.Employee.c.UnitID),
        "designation": (s.Designations.c.DesignationName, s.Designations,
                        s.Designations.c.DesignationID == s.Employee.c.DesignationID),
        "category": (s.Categories.c.CategoryName, s.Categories,
                     s.Categories.c.CategoryID == s.Employee.c.CategoryID),
    }

    key = (group_by or "").strip().lower()
    if key in dimensions:
        label, table, on = dimensions[key]
        stmt = (
            select(label.label(key), func.count().label("employees"))
            .select_from(s.Employee.outerjoin(table, on))
            .where(s.Employee.c.Status == s.ACTIVE)
            .group_by(label)
            .order_by(func.count().desc())
        )
        out["grouped_by"] = key
        rows = _rows(stmt)
        # A null group is a real bucket, not a missing row: employees whose
        # UnitID/DesignationID is null or points at an inactive record. Left as
        # None it renders as a blank line in a table and reads like a glitch,
        # and the numbers underneath stop adding up to the total.
        unassigned = 0
        for r in rows:
            if r[key] is None:
                r[key] = "Unassigned"
                unassigned = r["employees"]
        out["breakdown"] = rows
        if unassigned:
            out["unassigned_note"] = (
                f"{unassigned} of {total} active employees have no {key} on record."
            )
    elif key in ("teaching", "is_teaching"):
        stmt = (
            select(s.Employee.c.IsTeaching.label("is_teaching"), func.count().label("employees"))
            .where(s.Employee.c.Status == s.ACTIVE)
            .group_by(s.Employee.c.IsTeaching)
        )
        out["grouped_by"] = "teaching"
        out["breakdown"] = [
            {"staff": "Teaching" if r["is_teaching"] else "Non-teaching",
             "employees": r["employees"]}
            for r in _rows(stmt)
        ]
    elif key:
        out["note"] = (
            f"Cannot group by {group_by!r}. Supported: unit, designation, category, teaching."
        )
    return out


# ------------------------------------------------------------------ recruiting
def list_vacancies(unit: str | None = None) -> dict:
    """Open vacancies, with the designation and unit each is for."""
    stmt = (
        select(
            s.Vacancies.c.VacancyName.label("vacancy"),
            s.Vacancies.c.VacancyCode.label("code"),
            s.Vacancies.c.Description.label("description"),
            s.Designations.c.DesignationName.label("designation"),
            s.Units.c.UnitName.label("unit"),
            s.AcademicYears.c.AcademicYearName.label("academic_year"),
        )
        .select_from(
            s.Vacancies
            .outerjoin(s.Designations,
                       s.Designations.c.DesignationID == s.Vacancies.c.DesignationID)
            .outerjoin(s.Units, s.Units.c.UnitID == s.Vacancies.c.UnitID)
            .outerjoin(s.AcademicYears,
                       s.AcademicYears.c.AcademicYearID == s.Vacancies.c.AcademicYearID)
        )
        .where(s.Vacancies.c.Status == s.ACTIVE)
        .order_by(s.Vacancies.c.VacancyName)
    )
    if unit:
        stmt = stmt.where(func.lower(s.Units.c.UnitName).like(f"%{unit.strip().lower()}%"))
    rows = _rows(stmt)
    return {
        "total_vacancies": len(rows),
        "filtered_by_unit": unit,
        "vacancies": rows,
        # A vacancy list is useless without the way to apply, and the model must
        # not invent a URL. Hand it the real one alongside the data.
        "how_to_apply": RESOURCES.get("application_form"),
    }


def list_policies(category: str | None = None) -> dict:
    """
    Policy documents on record: name, category, effective date.

    Returns the LIST, not the text. Policies.Content holds the full document and
    quoting it accurately is a retrieval problem, not a lookup one -- that is
    what the RAG side is for.
    """
    stmt = (
        select(
            s.Policies.c.PolicyName.label("policy"),
            s.PolicyCategories.c.CategoryName.label("category"),
            s.Policies.c.Effected_Date.label("effective_from"),
            s.Units.c.UnitName.label("unit"),
        )
        .select_from(
            s.Policies
            .outerjoin(s.PolicyCategories,
                       s.PolicyCategories.c.CategoryID == s.Policies.c.CategoryID)
            .outerjoin(s.Units, s.Units.c.UnitID == s.Policies.c.UnitID)
        )
        .where(s.Policies.c.Status == s.ACTIVE)
        .order_by(s.Policies.c.PolicyName)
    )
    if category:
        stmt = stmt.where(
            func.lower(s.PolicyCategories.c.CategoryName).like(f"%{category.strip().lower()}%")
        )
    rows = _rows(stmt)
    for r in rows:
        r["effective_from"] = str(r["effective_from"].date()) if r["effective_from"] else None
    return {"total_policies": len(rows), "filtered_by_category": category, "policies": rows}
