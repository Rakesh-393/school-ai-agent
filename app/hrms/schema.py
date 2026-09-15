"""
The HRMS tables, described for reading only.

WHY THESE ARE NOT db.Model CLASSES
Every db.Model lands in db.metadata, and db.create_all() builds everything in
db.metadata. Describing someone else's tables that way would put them on the
list of tables this app tries to create on boot. They live on their own
MetaData instead, so there is no path from "the agent can read this table" to
"the agent tried to create this table". dburi.owns_schema() is the second lock;
this is the first.

WHY DECLARED AND NOT REFLECTED
reflect() needs a live connection at import time, so the app would not start
offline and the columns would be invisible in code review. Declaring them costs
a few lines and makes every column the agent can see explicit and greppable.
Only the columns the tools actually read are listed. Employee in particular has
64 columns of personal data -- date of birth, parents' names and mobiles,
spouse, home address -- and none of it is described here, because a column that
does not exist in this file cannot be selected by a tool or leaked by a model.

CONVENTIONS IN THIS DATABASE
  * Every master table has Status; 1 means active. Tools filter on it.
  * Primary keys are named after the table: UnitID, SubjectID, DesignationID.
  * Names are <Thing>Name, codes are <Thing>Code.
"""
from sqlalchemy import Column, DateTime, Integer, MetaData, String, Table

# Deliberately NOT db.metadata. See the module docstring.
hrms_metadata = MetaData()


def _table(name: str, *columns: Column) -> Table:
    """Every table here carries Status; add it once rather than in each call."""
    return Table(name, hrms_metadata, *columns, Column("Status", Integer))


Organisations = _table(
    "Organisations",
    Column("OrganisationID", Integer, primary_key=True),
    Column("OrganisationName", String),
    Column("OrganisationCode", String),
    Column("OrganisationAddress", String),
    Column("CityID", Integer),
    Column("StateID", Integer),
    Column("CountryID", Integer),
)

Units = _table(
    "Units",
    Column("UnitID", Integer, primary_key=True),
    Column("UnitName", String),
    Column("UnitCode", String),
    Column("UnitAddress", String),
    Column("OrganisationID", Integer),
    Column("LocationID", Integer),
    Column("CityID", Integer),
    Column("StateID", Integer),
    Column("CountryID", Integer),
)

SubUnits = _table(
    "SubUnits",
    Column("SubUnitID", Integer, primary_key=True),
    Column("SubUnitName", String),
    Column("SubUnitCode", String),
    Column("UnitID", Integer),
)

Locations = _table(
    "Locations",
    Column("LocationID", Integer, primary_key=True),
    Column("LocationName", String),
    Column("LocationCode", String),
    Column("CityID", Integer),
    Column("StateID", Integer),
    Column("CountryID", Integer),
)

Countries = _table(
    "Countries",
    Column("CountryID", Integer, primary_key=True),
    Column("CountryName", String),
    Column("CountryCode", String),
)

States = _table(
    "States",
    Column("StateID", Integer, primary_key=True),
    Column("CountryID", Integer),
    Column("StateName", String),
    Column("StateCode", String),
)

Cities = _table(
    "Cities",
    Column("CityID", Integer, primary_key=True),
    Column("StateID", Integer),
    Column("CountryID", Integer),
    Column("CityName", String),
    Column("CityCode", String),
)

Subjects = _table(
    "Subjects",
    Column("SubjectID", Integer, primary_key=True),
    Column("SubjectName", String),
    Column("SubjectCode", String),
    Column("OrganisationID", Integer),
)

Designations = _table(
    "Designations",
    Column("DesignationID", Integer, primary_key=True),
    Column("DesignationName", String),
    Column("DesignationCode", String),
    Column("Description", String),
    Column("FunctionID", Integer),
    Column("CategoryID", Integer),
    Column("LevelID", Integer),
)

Grades = _table(
    "Grades",
    Column("GradeID", Integer, primary_key=True),
    Column("GradeName", String),
    Column("GradeCode", String),
    Column("NEPStageID", Integer),
    Column("StreamID", Integer),
)

Streams = _table(
    "Streams",
    Column("StreamID", Integer, primary_key=True),
    Column("StreamName", String),
    Column("StreamCode", String),
)

NEPStages = _table(
    "NEPStages",
    Column("StageID", Integer, primary_key=True),
    Column("StageName", String),
    Column("StageCode", String),
    Column("Description", String),
)

Categories = _table(
    "Categories",
    Column("CategoryID", Integer, primary_key=True),
    Column("CategoryName", String),
    Column("CategoryCode", String),
    Column("Description", String),
    Column("FunctionID", Integer),
    Column("IsTeaching", Integer),
)

Functionalities = _table(
    "Functionalities",
    Column("FunctionID", Integer, primary_key=True),
    Column("FunctionName", String),
    Column("FunctionCode", String),
    Column("Description", String),
)

AcademicYears = _table(
    "AcademicYears",
    Column("AcademicYearID", Integer, primary_key=True),
    Column("AcademicYearName", String),
    Column("StartDate", DateTime),
    Column("EndDate", DateTime),
)

Vacancies = _table(
    "Vacancies",
    Column("VacancyID", Integer, primary_key=True),
    Column("VacancyName", String),
    Column("VacancyCode", String),
    Column("Description", String),
    Column("DesignationID", Integer),
    Column("FunctionID", Integer),
    Column("CategoryID", Integer),
    Column("UnitID", Integer),
    Column("AcademicYearID", Integer),
)

Policies = _table(
    "Policies",
    Column("PolicyID", Integer, primary_key=True),
    Column("PolicyName", String),
    Column("CategoryID", Integer),
    Column("UnitID", Integer),
    Column("Effected_Date", DateTime),
)

PolicyCategories = _table(
    "Policy_Categories",
    Column("CategoryID", Integer, primary_key=True),
    Column("CategoryName", String),
)

# Only what headcount groups by. The real table has 64 columns, and the ones
# left out are left out on purpose: FirstName, Mobile, Email, DOB_Actual,
# FatherName, PresentHouseNo, Salary, AadhaarNo, PAN. An aggregate never needs
# them, and a column absent here cannot be selected by a tool or surfaced by a
# model. Add one only alongside a rule about who may see it.
Employee = _table(
    "Employee",
    Column("EmployeeID", Integer, primary_key=True),
    Column("Gender", String),
    Column("IsTeaching", Integer),
    Column("UnitID", Integer),
    Column("DesignationID", Integer),
    Column("CategoryID", Integer),
    Column("FunctionID", Integer),
    Column("DateOfJoining", DateTime),
)

ACTIVE = 1
