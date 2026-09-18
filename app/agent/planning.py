"""
make_plan -- the tool that turns the chatbot into an agent.

The loop in orchestrator.ask() could always run several steps: the model calls
a tool, reads the result, and decides whether to call another. It almost never
did. Across 90 real conversations, 75 used exactly one tool and 4 used more
than one, because people ask small questions and small questions need one
lookup. The agent machinery was there and idle.

This tool gives it something to do. For a JOB -- "prepare a hiring report",
"compare our branches", "draft a post for each open vacancy" -- the model writes
a plan: which tools to call, with which arguments, and why. This module runs
every step, then hands all the results back at once. The model checks them
against the goal and writes the result. The model decides the steps; nobody
tells it which tables to read.

This is the plan-and-execute pattern. Two reasons it runs the steps here rather
than asking the model to call them itself:

  1. THE MODEL WOULD NOT BATCH. The first version only recorded the plan and
     told the model to "call the independent lookups together". It made them
     one per round trip anyway, and still did after being told twice. Every
     round trip resends the system prompt and tool list -- about 2,000 tokens --
     and Groq's free tier allows 8,000 tokens a minute for this model. A
     four-step report used its budget by the fourth call, Groq answered 429,
     and the OpenAI client silently waited 13 and 21 seconds per retry: a
     64-second report, of which the database took a third of a second.
     Running the plan here makes it two model calls whatever the model's habits.

  2. THE PLAN BECOMES VISIBLE AND CHECKABLE. Each step lands in the trace with
     its reason and its result, so the chat page shows what the model decided
     to do and what happened. That is the difference between "it is an agent"
     as a claim and as something you can watch.

Nothing here widens access. Every step goes through the toolkit's own execute(),
so role gating applies step by step: an employee who asks for a hiring report
gets a plan whose vacancy step comes back "forbidden", and the model is told to
report that rather than skip it.
"""
from __future__ import annotations

from typing import Callable

MIN_STEPS = 2
MAX_STEPS = 6
PLAN_TOOL = "make_plan"


def make_plan(
    goal: str,
    steps: list[dict],
    *,
    execute: Callable[[str, dict, str], dict],
    role: str,
) -> dict:
    """
    Run a multi-step plan and return every step's result.

    `execute` and `role` are supplied by the toolkit, never by the model -- see
    the `orchestrates` flag in the registry. That is what keeps role gating on
    every step.
    """
    goal = (goal or "").strip()
    steps = [s for s in (steps or []) if isinstance(s, dict) and s.get("tool")]

    if not goal:
        return {"error": "missing_goal", "message": "State the job you are planning for."}

    if len(steps) < MIN_STEPS:
        # One step is a question, not a job. Planning it costs a round trip
        # and makes a simple answer slower for nothing.
        return {
            "error": "not_a_job",
            "message": ("This needs only one lookup, so it is a question, not a job. "
                        "Skip the plan: call that tool directly and answer."),
        }

    if len(steps) > MAX_STEPS:
        # A long plan usually means the write-up has been split into steps.
        return {
            "error": "plan_too_long",
            "message": (f"{len(steps)} steps is too many. Keep only lookups that need a "
                        f"tool, at most {MAX_STEPS}. Writing the result is not a step."),
        }

    results = []
    for i, step in enumerate(steps, 1):
        tool = str(step.get("tool", "")).strip()
        args = step.get("args") or {}
        if not isinstance(args, dict):
            args = {}

        if tool == PLAN_TOOL:
            # A plan that plans is a loop with extra steps.
            result = {"error": "nested_plan", "message": "A plan step cannot be another plan."}
        else:
            result = execute(tool, args, role)

        results.append({
            "step": i,
            "why": str(step.get("why", "")).strip(),
            "tool": tool,
            "args": args,
            "ok": "error" not in result,
            "result": result,
        })

    failed = [r["step"] for r in results if not r["ok"]]
    return {
        "goal": goal,
        "steps_run": len(results),
        "steps_failed": failed,
        "results": results,
        "next": (
            "Every step has run; the results are above. Check each one against the "
            "goal. If something you need is still missing and a tool can get it, call "
            "that tool now. Then write the finished result using only figures from "
            "these results. For any failed step, say in one line what could not be "
            "done and why -- never drop it silently and never fill the gap with a guess."
        ),
    }


# Shared schema so any toolkit can offer planning without restating it.
MAKE_PLAN_TOOL = {
    "name": PLAN_TOOL,
    "description": (
        "For a JOB that needs several lookups combined into one result -- a report, a "
        "summary, a comparison, a set of drafts. List every lookup as a step: the tool "
        "to call, its arguments, and why. All steps run at once and you get every "
        "result back together. Do NOT use this for a simple question that one tool "
        "answers; call that tool directly."
    ),
    "fn": make_plan,
    # The registry passes execute() and the user's role to tools with this flag,
    # instead of letting the model supply them.
    "orchestrates": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": "The job in one sentence, e.g. 'Hiring report for HR'.",
            },
            "steps": {
                "type": "array",
                "description": f"{MIN_STEPS} to {MAX_STEPS} lookups, in order.",
                "items": {
                    "type": "object",
                    "properties": {
                        "why": {"type": "string",
                                "description": "What this step is for, in a few words."},
                        "tool": {"type": "string",
                                 "description": "Name of one of your other tools."},
                        "args": {"type": "object",
                                 "description": "That tool's arguments. {} if it takes none."},
                    },
                    "required": ["why", "tool", "args"],
                },
            },
        },
        "required": ["goal", "steps"],
        "additionalProperties": False,
    },
}
