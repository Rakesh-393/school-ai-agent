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
from datetime import date

from config import Config
from app.agent import tools as toolkit
from app.agent.llm import get_llm

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
- For questions about rules, policies, procedures, holidays or events, call
  search_school_documents and answer only from the passages it returns,
  naming the source document.
- Answer in a few short sentences or a small markdown table. No preamble.
  Amounts are Indian Rupees; write them as Rs 12,500.

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
    def __init__(self, role: str = "parent"):
        self.role = role if role in ("parent", "teacher", "admin") else "parent"
        self.llm = get_llm()
        self.tools = toolkit.tools_for_role(self.role)

    def ask(self, question: str, history: list[dict] | None = None) -> dict:
        started = time.perf_counter()

        messages: list[dict] = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(today=date.today(), role=self.role),
            }
        ]
        messages.extend(history or [])
        messages.append({"role": "user", "content": question})

        trace: list[dict] = []
        answer = ""

        for step in range(Config.MAX_AGENT_STEPS):
            reply = self.llm.chat(messages, tools=self.tools)

            if not reply.wants_tools:
                answer = reply.text
                break

            # Record the assistant turn *including* the tool_use blocks, or the
            # provider will reject the tool results as unsolicited.
            self.llm.append_assistant(messages, reply)

            for call in reply.tool_calls:
                result = toolkit.execute(call.name, call.args, self.role)
                trace.append(
                    {
                        "step": step + 1,
                        "tool": call.name,
                        "args": call.args,
                        "ok": "error" not in result,
                        "result": result,
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

        return {
            "answer": answer or "Sorry, I could not work that out. Please rephrase.",
            "role": self.role,
            "trace": trace,
            "tools_used": [t["tool"] for t in trace],
            "latency_ms": int((time.perf_counter() - started) * 1000),
        }
