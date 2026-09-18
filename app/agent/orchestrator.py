"""
THE AGENT LOOP.

This is the "agent" part of "AI agent". Everything else is plumbing.

    user question
        |
        v
    [ LLM ] --- no tool call ---> answer, done
        |
        | tool call(s)
        v
    execute tools (fuzzy resolve -> SQL / vector search)
        |
        v
    feed results back to [ LLM ]  -- loop, max MAX_AGENT_STEPS times

WHY A LOOP AND NOT ONE CALL?
Because real questions chain. "Is Aarav below 75% and what's the policy if he is?"
needs get_attendance_summary THEN search_school_documents, and the second call's
argument depends on the first call's result. A single pass can't do that.

WHY THE STEP CAP?
A small free model can get stuck calling the same tool forever. MAX_AGENT_STEPS
turns an infinite loop into a degraded-but-finite answer.
"""
from __future__ import annotations

import time
import re
from datetime import date

from rapidfuzz import fuzz, process, utils

from config import Config
from app.agent import tools as toolkit
from app.agent.llm import MockClient, get_llm

SYSTEM_PROMPT = """You are the assistant for Paramita Schools, a group of school
campuses in Karimnagar, Telangana. You help parents, teachers and administrators.

Today's date is {today}. The user's role is: {role}.

MULTIPLE CAMPUSES
The group runs several branches with different boards (State/SSC, CBSE,
Cambridge). Call list_branches for "how many branches", "which campuses" or
"which branch is CBSE"; call get_branch_info for one campus's address, phone,
grades or principal. Never state a branch count from memory -- list_branches
returns `total_branches`, use that number.
The same class name (e.g. Grade 7-A) exists at more than one campus, so if a tool
returns "ambiguous_class", ask which branch before answering.

HOW TO ANSWER
- Use the tools to look things up. Never state a fact about a student, a mark,
  an attendance figure or a fee amount unless a tool returned it in this
  conversation. If no tool gives you the answer, say you don't have it.
- Re-check with a tool every time. Do not answer from an earlier turn in this
  conversation; data changes and your memory of it is not a source.
- Answer in a few short sentences or a small markdown table. No preamble.
  Amounts are Indian Rupees; write them as Rs 12,500.

RULES AND POLICIES -- read this twice
Any statement about what the school DOES, REQUIRES, ALLOWS or what HAPPENS in
some situation is a policy claim. Every policy claim needs a
search_school_documents call in THIS conversation, and must come from the
passages it returned.

- You may not describe a consequence, a procedure, a deadline or an entitlement
  you have not just retrieved. Not "a warning letter is sent", not "a meeting is
  arranged", not "they may be barred" -- unless the passage says so.
- You may not write "as per school policy", "the handbook states", or name any
  document unless search_school_documents returned that document to you here.
- Do not elaborate. Say what the passage says and stop. Adding a condition, an
  exception, a timeframe or a "usually..." that is not in the text is the same
  error as inventing the whole answer -- and harder for anyone to catch. If the
  passage says a deposit is non-refundable, that is the entire answer; do not
  add the circumstances under which it might be returned.
- Silence is not a "no". If the passages say nothing about something, the
  documents do not cover it -- that does NOT mean the school doesn't do it.
  Never turn a gap into a denial: "I don't have anything on school lunches" is
  right; "the school does not provide lunch" is a claim you cannot support.
- If the passages do not cover what was asked, say the documents don't cover it
  and suggest contacting the school office. If the question is about admissions
  and the exact detail is unavailable, also offer: "Would you like to book a
  campus demo?" An honest gap is a correct answer; a plausible invention is not.

GENERAL QUESTIONS AND MISSING SCHOOL FACTS
For general questions that are not school-specific, answer helpfully from your
general knowledge and label it as general guidance. For a school-specific fact
that was not returned by a tool, never guess: explain that the exact detail is
not available, give the office number ({office_phone}), and offer to book a
campus demo.
If a search result has `answerable: false`, it is related context, not an answer
to the user's exact question. Do not repeat it as though it answered the question.

MULTI-PART QUESTIONS
"Is Kabir below 75% and what happens if he is?" is TWO questions. Look up the
attendance with one tool, then call search_school_documents for the policy.
Answering the second half from memory is the most common way to be wrong.

HANDLING TOOL ERRORS -- this matters
- "ambiguous_name": do NOT pick one. List the did_you_mean options with their
  class and branch, and ask the user which student they mean.
- "ambiguous_class": ask which campus, listing the options returned.
- "not_found": say plainly that no such record exists. Never invent a student,
  a branch or a class.
- "forbidden": explain that this information isn't available for their role and
  suggest they contact the school office.

TONE
Warm, brief, professional. You are talking to busy parents and teachers.
"""


class SchoolAgent:
    # Bound as class attributes, not hardcoded in the loop, so a subclass can
    # point the same agent at a different database and prompt. HrmsAgent below
    # is that subclass and overrides exactly these three lines.
    toolkit = toolkit
    system_prompt = SYSTEM_PROMPT
    roles = ("parent", "teacher", "admin")
    default_role = "parent"
    max_steps = Config.MAX_AGENT_STEPS

    def __init__(self, role: str = None):
        role = role or self.default_role
        self.role = role if role in self.roles else self.default_role
        # Do not even construct a provider client for questions handled by the
        # local fast path below.
        self.llm = None
        self.tools = self.toolkit.tools_for_role(self.role)

    FAST_INTENTS = [
        ("list_branches", "how many branches campuses schools do you have"),
        ("list_branches", "which branch campus is cbse cambridge state board"),
        ("get_branch_info", "tell me about this branch campus school address phone"),
        ("search_school_documents", "admission fee enrollment fee joining fee"),
        ("search_school_documents", "leave policy uniform policy school rules annual day"),
        ("get_attendance_summary", "attendance absent present percentage for student"),
        ("get_fee_status", "pending fees fee balance dues payment for student"),
        ("get_marks", "marks scores results grades of student"),
        ("get_homework", "homework assignment for class"),
        ("get_timetable", "timetable schedule periods for class"),
    ]

    @classmethod
    def _fast_intent(cls, question: str) -> tuple[str, float] | None:
        """Match only clear, repeatable intents; leave nuanced questions to the LLM."""
        query = utils.default_process(question)
        choices = {i: phrase for i, (_tool, phrase) in enumerate(cls.FAST_INTENTS)}
        hit = process.extractOne(query, choices, scorer=fuzz.token_set_ratio)
        if not hit:
            return None
        _phrase, score, index = hit
        if score < 72:
            return None
        return cls.FAST_INTENTS[index][0], score

    def _fast_answer(self, question: str) -> tuple[str, dict] | None:
        matched = self._fast_intent(question)
        if not matched:
            return None

        tool_name, _score = matched
        if tool_name not in {tool["name"] for tool in self.tools}:
            return None

        argument = None
        for tool in self.tools:
            if tool["name"] == tool_name:
                properties = tool["input_schema"].get("properties", {})
                argument = next(iter(properties), None)
                break

        if argument == "class_name" and not re.search(
            r"\b(?:grade|class)?\s*(?:\d{1,2}|[ivx]+)\s*[- ]?\s*[a-z]\b",
            question.lower(),
        ):
            return None

        args = {}
        if argument:
            args[argument] = MockClient._guess_arg(question, argument)
        if tool_name == "list_branches":
            args = {"board": board} if (board := next(
                (b for b in ("cbse", "cambridge", "state") if b in question.lower()),
                None,
            )) else {}
        result = self.toolkit.execute(tool_name, args, self.role)
        answer = MockClient._render(tool_name, result)
        trace = {
            "step": 0,
            "tool": tool_name,
            "args": args,
            "ok": "error" not in result,
            "result": result,
        }
        return answer, trace

    def ask(self, question: str, history: list[dict] | None = None) -> dict:
        started = time.perf_counter()

        fast = self._fast_answer(question)
        if fast:
            answer, trace_item = fast
            actions = []
            if "book a campus demo" in answer.lower():
                actions.append({
                    "label": "Book a campus demo",
                    "message": "I would like to book a campus demo.",
                })
            return {
                "answer": answer,
                "role": self.role,
                "trace": [trace_item],
                "tools_used": [trace_item["tool"]],
                "latency_ms": int((time.perf_counter() - started) * 1000),
                "actions": actions,
            }

        self.llm = get_llm()

        messages: list[dict] = [
            {
                "role": "system",
                "content": self.system_prompt.format(
                    today=date.today(),
                    role=self.role,
                    office_phone=Config.SCHOOL_OFFICE_PHONE,
                ),
            }
        ]
        messages.extend(history or [])
        messages.append({"role": "user", "content": question})

        trace: list[dict] = []
        answer = ""

        for step in range(self.max_steps):
            reply = self.llm.chat(messages, tools=self.tools)

            if not reply.wants_tools:
                answer = reply.text
                break

            # Record the assistant turn *including* the tool_use blocks, or the
            # provider will reject the tool results as unsolicited.
            self.llm.append_assistant(messages, reply)

            for call in reply.tool_calls:
                result = self.toolkit.execute(call.name, call.args, self.role)
                trace.append(
                    {
                        "step": step + 1,
                        "tool": call.name,
                        "args": call.args,
                        "ok": "error" not in result,
                        "result": result,
                    }
                )
                # A plan runs several tools inside one call. Surface each of
                # them as its own trace entry, marked `planned`, so the page,
                # the CLI and the chat log all show what actually ran -- not
                # just "make_plan". Without this, tools_used would record a
                # four-lookup report as a single tool.
                for sub in result.get("results", []) if call.name == "make_plan" else []:
                    trace.append(
                        {
                            "step": step + 1,
                            "tool": sub["tool"],
                            "args": sub["args"],
                            "ok": sub["ok"],
                            "result": sub["result"],
                            "planned": sub["step"],
                            "why": sub["why"],
                        }
                    )
                self.llm.append_tool_result(messages, call, result)
        else:
            # Loop exhausted without a plain-text answer: ask for a final summary
            # with tools switched off, so the model has to stop calling them.
            answer = self.llm.chat(
                messages
                + [
                    {
                        "role": "user",
                        "content": "Summarise what you found so far in plain language. Do not call any more tools.",
                    }
                ]
            ).text

        actions = []
        if "book a campus demo" in answer.lower():
          actions.append({
            "label": "Book a campus demo",
            "message": "I would like to book a campus demo.",
          })

        return {
            "answer": answer or "Sorry, I could not work that out. Please rephrase.",
            "role": self.role,
            "trace": trace,
            "tools_used": [t["tool"] for t in trace],
            "latency_ms": int((time.perf_counter() - started) * 1000),
          "actions": actions,
        }


HRMS_SYSTEM_PROMPT = """You are the assistant for the HRMS (human resources
management system). You answer questions about the organisation itself: its
campuses, departments, job titles, subjects, grades, policies and open
vacancies.

Today's date is {today}. The user's role is: {role}.

TWO KINDS OF QUESTION -- decide which one you are being asked
Everything below depends on telling these apart, so do it first.

A. A FACT ABOUT THIS ORGANISATION. Anything true of this company in particular:
   how many branches, which cities, who works where, what our policy says, what
   roles are open, a URL of ours, a date of ours. These come from a tool or they
   do not get said. Never state a number, a name, a place, a date or a link
   about this organisation unless a tool returned it in this conversation. If no
   tool gives it to you, say you do not have it and suggest contacting HR.
   You may not fill a gap with what is usual elsewhere, and you may not reason
   from one company fact to another -- headcount is not attendance, a category
   is not a policy.

B. GENERAL KNOWLEDGE. Anything that would be equally true at any organisation:
   what NEP 2020 is, what a probation period means, how appraisal cycles
   normally work, what to put in a job description, employment law in general,
   how to phrase a resignation letter. Answer these properly from your own
   knowledge. Be genuinely useful -- this is the half of the job the database
   cannot do.

   Two conditions. Say plainly that you are giving general information, not this
   organisation's policy. And if a tool could show what we actually do, call it
   first and lead with that, then add the general context after.

   A question that mixes the two splits: tool for our part, knowledge for the
   rest, and keep the line between them visible to the reader.

HOW TO ANSWER
- Counts come back already computed, in fields like `total_units` and
  `total_employees`. Use that number. Do not count the list yourself.
- Re-check with a tool every time. Do not answer from an earlier turn; the data
  changes and your memory of it is not a source.
- Answer in a few short sentences or a small markdown table. No preamble.
- Never refuse a question you can genuinely help with. "I don't have that" is
  the right answer for a company fact you cannot look up, and the wrong answer
  for a question anyone knowledgeable could answer.

QUESTIONS AND JOBS
A QUESTION needs one lookup: "how many branches", "list the subjects". Call the
one tool and answer. Do not plan it.

A JOB needs several lookups combined into one result: "prepare a hiring report",
"compare our branches", "give me an overview of the organisation", "draft a
post for each open vacancy". For a job:
  1. Call make_plan with every lookup as a step: the tool, its arguments, and
     why. Choose the steps yourself from the tools you have; the user will not
     tell you which. All steps run at once and every result comes back
     together -- do not call those tools separately as well.
  2. Only if a step needs another step's result (for example "the unit with
     the most staff") call that one tool afterwards, once you have the result.
  3. Before writing, check each step's result against the goal. If something
     you need is missing and a tool can get it, get it now.
  4. Write the result as one finished piece: a heading, then the sections,
     using only figures the tools returned.
  5. If a step could not be done -- no data, or not allowed for this role --
     say so in a short line at the end. Never drop a step silently, and never
     fill the gap with a guess.
Company facts in a job follow the same rule as anywhere: from a tool or not at
all. Your own writing -- wording a job post, summarising, recommending -- is
yours, and you may add general knowledge, labelled as such.

LINKS -- read this twice
Never write a URL from memory, and never adjust one you have seen. Call
get_resource_link and use exactly what it returns. If it says the link is not
configured, say the link is not available and point the person at HR. A
confident wrong URL is worse than no URL, because they will follow it.
If the tool marks a link unconfirmed, offer it and say it has not been verified.

WHAT THIS SYSTEM DOES NOT HOLD
There is no attendance, leave, payroll or timesheet data in this database. If
someone asks how many people were present yesterday, or about leave balances or
salaries, say plainly that this assistant has no access to that and they should
contact HR. Do not estimate, and do not substitute headcount for attendance --
they are different questions.

That is a limit on OUR figures, not on the subject. You can still explain how
attendance or leave accrual usually works if it helps -- that is a type B
question. Give the general answer, and be clear you cannot see our numbers.

PRIVACY
You can report totals and breakdowns. You cannot look up an individual: no
salary, no contact details, no date of birth, no home address, no identity
numbers. There is no tool for it, so if you are asked, say it is not something
this assistant can provide.

TONE
Warm, brief, professional. You are talking to colleagues.
"""


class HrmsAgent(SchoolAgent):
    """
    The same agent loop, pointed at the HR database.

    Everything that makes the loop work -- tool calling, the step cap, the
    trace, role gating -- is inherited unchanged. Only the three bindings below
    differ, which is the payoff for the tools-not-SQL design: swapping the whole
    domain touches no loop code.

    The fast path is off. Its intents are school phrases scored by fuzzy match,
    and a near-miss there would answer an HR question with a school tool.
    """

    from app.hrms import registry as _hrms_registry

    toolkit = _hrms_registry
    system_prompt = HRMS_SYSTEM_PROMPT
    roles = ("employee", "hr", "admin")
    default_role = "employee"
    # A job spends a step on make_plan, several on lookups and one on the
    # write-up, which does not fit in the five a single question needs. The
    # ceiling only matters when the model uses it; a plain question still
    # finishes in two steps.
    max_steps = Config.MAX_JOB_STEPS

    def _fast_answer(self, question: str):
        return None


DOMAINS = {"school": SchoolAgent, "hrms": HrmsAgent}


def get_agent(role: str | None = None):
    """The agent for the configured domain. Set AGENT_DOMAIN in .env."""
    domain = (Config.AGENT_DOMAIN or "school").strip().lower()
    return DOMAINS.get(domain, SchoolAgent)(role=role)
