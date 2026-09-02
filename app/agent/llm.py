"""
LLM abstraction.

Everything upstream (the orchestrator) speaks ONE shape:

    client.chat(messages, tools) -> LLMReply(text, tool_calls, raw)

...so swapping a free model for a paid one is a single .env edit, and none of
your agent logic changes. That is the whole point of this file.

TWO WIRE FORMATS EXIST IN THE WORLD:

  1. OpenAI-compatible  ->  Groq, Google Gemini, Ollama, OpenRouter, Together.
     tool calls come back on  message.tool_calls[]  and results are fed back as
     a message with  role="tool".

  2. Anthropic          ->  Claude.
     tool calls are  content blocks of type "tool_use"  and results go back as
     content blocks of type "tool_result" inside a *user* message.

We normalise both to ToolCall(id, name, args).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from config import Config


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class LLMReply:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: object = None

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


# ---------------------------------------------------------------- OpenAI-compatible
class OpenAICompatClient:
    """Groq / Gemini / Ollama / OpenRouter -- same wire format, different base_url."""

    def __init__(self, api_key: str, base_url: str, model: str, extra: dict | None = None):
        from openai import OpenAI

        self.model = model
        # Provider-specific request params (e.g. Groq's reasoning_effort).
        # Kept out of chat() so the call site stays provider-agnostic.
        self.extra = extra or {}
        self._c = OpenAI(api_key=api_key or "not-needed", base_url=base_url)

    @staticmethod
    def to_wire_tools(tools: list[dict]) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                },
            }
            for t in tools
        ]

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMReply:
        kwargs = {"model": self.model, "messages": messages, "temperature": 0.1, **self.extra}
        if tools:
            kwargs["tools"] = self.to_wire_tools(tools)
            kwargs["tool_choice"] = "auto"

        resp = self._c.chat.completions.create(**kwargs)
        msg = resp.choices[0].message

        calls = []
        for tc in (msg.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append(ToolCall(id=tc.id, name=tc.function.name, args=args))

        return LLMReply(text=msg.content or "", tool_calls=calls, raw=msg)

    # --- history helpers (format-specific, so they live on the client) ---
    def append_assistant(self, messages: list[dict], reply: LLMReply) -> None:
        messages.append(reply.raw.model_dump(exclude_none=True))

    def append_tool_result(self, messages: list[dict], call: ToolCall, result: dict) -> None:
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call.id,
                "name": call.name,
                "content": json.dumps(result, default=str),
            }
        )


# ---------------------------------------------------------------- Anthropic
class AnthropicClient:
    """The upgrade path. Same .chat() signature, different wire format."""

    def __init__(self, api_key: str, model: str):
        import anthropic

        self.model = model
        self._c = anthropic.Anthropic(api_key=api_key)

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMReply:
        # Anthropic keeps the system prompt out of `messages`.
        system = ""
        convo = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                convo.append(m)

        kwargs = {
            "model": self.model,
            "max_tokens": 4096,
            "system": system,
            "messages": convo,
            "thinking": {"type": "adaptive"},   # current API; budget_tokens is removed
        }
        if tools:
            kwargs["tools"] = tools             # already {name, description, input_schema}

        resp = self._c.messages.create(**kwargs)

        text = "".join(b.text for b in resp.content if b.type == "text")
        calls = [
            ToolCall(id=b.id, name=b.name, args=b.input)
            for b in resp.content
            if b.type == "tool_use"
        ]
        return LLMReply(text=text, tool_calls=calls, raw=resp)

    def append_assistant(self, messages: list[dict], reply: LLMReply) -> None:
        messages.append({"role": "assistant", "content": reply.raw.content})

    def append_tool_result(self, messages: list[dict], call: ToolCall, result: dict) -> None:
        # Claude expects tool_result blocks inside a *user* message. Batch them:
        # if the last message is already a tool-result user turn, extend it.
        block = {
            "type": "tool_result",
            "tool_use_id": call.id,
            "content": json.dumps(result, default=str),
        }
        if (
            messages
            and messages[-1]["role"] == "user"
            and isinstance(messages[-1]["content"], list)
            and messages[-1]["content"]
            and messages[-1]["content"][0].get("type") == "tool_result"
        ):
            messages[-1]["content"].append(block)
        else:
            messages.append({"role": "user", "content": [block]})


# ---------------------------------------------------------------- mock
class MockClient:
    """
    A scripted LLM. No network, no API key, deterministic.

    Set LLM_PROVIDER=mock. Use it to test the orchestrator, the routes and the
    UI before you have a key -- and in CI, where you do not want your tests to
    depend on a rate-limited free tier.

    It keyword-matches the question to one tool call, then produces a text
    answer from whatever the tool returned.
    """

    RULES = [
        # Order matters -- first match wins, so put the specific rules first.
        (("how many branch", "how many campus", "branches", "campuses",
          "list your school", "which branch", "which campus"), "list_branches", None),
        (("branch", "campus"), "get_branch_info", "branch_name"),
        (("attendance", "absent", "present"), "get_attendance_summary", "student_name"),
        (("fee", "fees", "due", "payment"), "get_fee_status", "student_name"),
        (("mark", "marks", "score", "result", "grade"), "get_marks", "student_name"),
        (("homework", "assignment"), "get_homework", "class_name"),
        (("timetable", "schedule", "period"), "get_timetable", "class_name"),
        (("policy", "rule", "leave", "uniform", "holiday", "annual day"),
         "search_school_documents", "query"),
    ]

    def __init__(self, *_a, **_kw):
        self.model = "mock"

    # Words that are never part of a student's name. A real LLM knows this
    # implicitly; the mock needs it spelled out.
    STOP = {
        "what", "whats", "what's", "is", "are", "the", "a", "an", "any", "for",
        "of", "in", "on", "my", "me", "show", "tell", "give", "get", "how",
        "much", "many", "does", "do", "did", "has", "have", "was", "were",
        "please", "can", "you", "child", "son", "daughter", "student", "pending",
        "attendance", "absent", "present", "fee", "fees", "due", "payment",
        "mark", "marks", "score", "result", "results", "grade", "grades",
        "homework", "assignment", "timetable", "schedule", "policy", "rule",
        "january", "february", "march", "april", "may", "june", "july",
        "august", "september", "october", "november", "december", "month",
        "term", "this", "last",
    }

    @classmethod
    def _guess_arg(cls, question: str, kind: str) -> str:
        import re

        if kind == "query":
            return question
        if kind == "branch_name":
            drop = {"branch", "branches", "campus", "campuses", "school", "the",
                    "what", "where", "is", "tell", "me", "about", "address",
                    "of", "your", "a", "an"}
            words = [w.strip("?.,!'\"") for w in question.split()]
            keep = [w for w in words if w and w.lower() not in drop]
            return " ".join(keep[:3]) if keep else question
        if kind == "class_name":
            m = re.search(r"\b(\d{1,2}\s*-?\s*[A-Za-z])\b", question)
            return m.group(1) if m else "7A"

        # student_name: drop stopwords, then prefer capitalised leftovers.
        words = [w.strip("?.,!'\"") for w in question.split()]
        keep = [w for w in words if w and w.lower() not in cls.STOP]
        caps = [w for w in keep if w[:1].isupper()]
        picked = caps or keep
        return " ".join(picked[:2]) if picked else question

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMReply:
        if not tools:  # final-summary pass
            return LLMReply(text="[mock] Summary of the findings above.")

        last_user = next(
            (m for m in reversed(messages)
             if m["role"] == "user" and isinstance(m.get("content"), str)),
            {"content": ""},
        )
        q = last_user["content"].lower()

        # Already ran a tool this turn? Then answer instead of looping.
        if any(m.get("role") == "tool" for m in messages):
            tool_msg = messages[-1]
            payload = json.loads(tool_msg["content"])
            return LLMReply(text=self._render(tool_msg.get("name", ""), payload))

        available = {t["name"] for t in tools}
        for keywords, tool_name, arg in self.RULES:
            if tool_name in available and any(k in q for k in keywords):
                if arg is None:
                    args = {}
                    # list_branches takes an optional board filter
                    for b in ("cbse", "cambridge", "state"):
                        if b in q:
                            args = {"board": b}
                            break
                else:
                    args = {arg: self._guess_arg(last_user["content"], arg)}
                return LLMReply(
                    tool_calls=[ToolCall(id="mock_1", name=tool_name, args=args)],
                    raw=type("M", (), {
                        "model_dump": lambda self, **k: {
                            "role": "assistant", "content": None,
                            "tool_calls": [{
                                "id": "mock_1", "type": "function",
                                "function": {"name": tool_name, "arguments": "{}"},
                            }],
                        }
                    })(),
                )
        return LLMReply(text="[mock] No rule matched. Try asking about attendance, fees or policy.")

    # ---------------------------------------------------------------- rendering
    # A real LLM turns a tool's JSON into prose. The mock has to do it by hand.
    # Keeping this here (rather than in the route or the template) means the
    # orchestrator's contract is identical for every provider: it always gets
    # finished text back, never a payload it has to format itself.

    @staticmethod
    def _money(n) -> str:
        """12500 -> 'Rs 12,500'  (Indian grouping is 12,500 / 1,12,500)."""
        try:
            n = float(n)
        except (TypeError, ValueError):
            return str(n)
        whole = f"{int(round(n)):,}"
        return f"Rs {whole}"

    @staticmethod
    def _table(headers: list[str], rows: list[list]) -> str:
        head = "| " + " | ".join(headers) + " |"
        rule = "| " + " | ".join("---" for _ in headers) + " |"
        body = ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
        return "\n".join([head, rule, *body])

    @classmethod
    def _render(cls, tool: str, p: dict) -> str:
        if "error" in p:
            return cls._render_error(p)

        if tool == "list_branches":
            n = p["total_branches"]
            filt = p.get("filtered_by_board")
            lead = (f"We have **{n} campuses** on the {filt} board:" if filt
                    else f"Paramita Schools has **{n} campuses**, all in Karimnagar:")
            rows = [[b["name"], b.get("board", "-"), b.get("locality", "-"), b.get("students", "-")]
                    for b in p["branches"]]
            return lead + "\n\n" + cls._table(["Campus", "Board", "Locality", "Students"], rows)

        if tool == "get_branch_info":
            bits = [f"**{p['name']}** ({p.get('code', '')}) — {p.get('board', '')}"]
            for label, key in [("Grades", "grades"), ("Address", "address"),
                               ("Phone", "phone"), ("Email", "email"),
                               ("Website", "website"), ("Principal", "principal"),
                               ("Established", "established")]:
                if p.get(key):
                    bits.append(f"- {label}: {p[key]}")
            if p.get("students") is not None:
                bits.append(f"- Students on roll: {p['students']} across {p.get('classes', 0)} classes")
            return "\n".join(bits)

        if tool == "get_attendance_summary":
            if p.get("message"):
                return f"{p['message']} ({p['student']}, {p.get('period', '')})"
            flag = ("\n\n**Below the 75% requirement** — the class teacher issues a written "
                    "notice at this level." if p["flag"] == "BELOW_75_PERCENT" else "")
            absent = p.get("absent_dates") or []
            tail = f"\n\nAbsent on: {', '.join(absent)}" if absent else ""
            return (f"{p['student']} ({p.get('class', '')}) was present for "
                    f"**{p['attendance_percent']}%** of {p['school_days']} school days "
                    f"in {p['period']}.\n\nBreakdown: "
                    + ", ".join(f"{k} {v}" for k, v in p["breakdown"].items())
                    + tail + flag)

        if tool == "get_fee_status":
            if p.get("message"):
                return f"{p['student']}: {p['message']}"
            rows = [[i["term"], i["head"], cls._money(i["amount"]), cls._money(i["paid"]),
                     cls._money(i["balance"]), i["status"]] for i in p["invoices"]]
            total = p["total_outstanding"]
            lead = (f"{p['student']} has **{cls._money(total)} outstanding**."
                    if total > 0 else f"{p['student']} has **no dues**. All invoices are settled.")
            return (lead + "\n\n"
                    + cls._table(["Term", "Head", "Amount", "Paid", "Balance", "Status"], rows))

        if tool == "get_marks":
            if p.get("message"):
                return f"{p['student']}: {p['message']}"
            rows = [[r["exam"], r["subject"], f"{r['scored']}/{r['out_of']}",
                     f"{r['percent']}%", r.get("grade", "-")] for r in p["results"]]
            overall = (f"\n\nOverall: **{p['overall_percent']}%**"
                       if p.get("overall_percent") is not None else "")
            return (f"Marks for {p['student']} ({p.get('class', '')}):\n\n"
                    + cls._table(["Exam", "Subject", "Score", "%", "Grade"], rows) + overall)

        if tool == "get_homework":
            hw = p.get("homework")
            if isinstance(hw, str):
                return f"{p['class']} at {p.get('branch', '')}: {hw}"
            rows = [[h["subject"], h["assigned_on"], h.get("due_on", "-"), h["details"]] for h in hw]
            return (f"Homework for **{p['class']}** at {p.get('branch', '')} "
                    f"(last {p['window_days']} days):\n\n"
                    + cls._table(["Subject", "Assigned", "Due", "Details"], rows))

        if tool == "get_timetable":
            tt = p.get("timetable")
            if isinstance(tt, str):
                return f"{p['class']}: {tt}"
            out = [f"Timetable for **{p['class']}** at {p.get('branch', '')} "
                   f"(room {p.get('room', '-')}):"]
            for day, slots in tt.items():
                periods = ", ".join(f"P{s['period']} {s['subject']}" for s in slots)
                out.append(f"- **{day}**: {periods}")
            return "\n".join(out)

        if tool == "get_class_attendance_report":
            rows = [[s["student"], s["school_days"], s["present"], f"{s['percent']}%"]
                    for s in p["students"]]
            low = p.get("below_75_percent") or []
            tail = (f"\n\n**Below 75%:** {', '.join(low)}" if low
                    else "\n\nEvery student is above 75%.")
            return (f"{p['class']} at {p.get('branch', '')}, {p['period']}:\n\n"
                    + cls._table(["Student", "Days", "Present", "%"], rows) + tail)

        if tool == "get_student_profile":
            lines = [f"**{p['name']}** ({p['admission_no']})"]
            for label, key in [("Class", "class"), ("Campus", "branch"),
                               ("Class teacher", "class_teacher"), ("Roll no", "roll_no"),
                               ("Date of birth", "date_of_birth"), ("Guardian", "guardian"),
                               ("Guardian phone", "guardian_phone"), ("Bus route", "bus_route")]:
                if p.get(key):
                    lines.append(f"- {label}: {p[key]}")
            return "\n".join(lines)

        if tool == "search_school_documents":
            top = p["passages"][0]
            crumb, _, body = top["text"].partition("\n\n")
            return f"{body.strip()}\n\n_Source: {top['source']} — {crumb.strip()}_"

        return json.dumps(p, indent=2, default=str)[:600]

    @staticmethod
    def _render_error(p: dict) -> str:
        """The clarifying-question path -- the most important behaviour to demo."""
        kind = p["error"]

        if kind == "ambiguous_name":
            opts = "\n".join(
                f"- **{c['name']}** — {c.get('class', '')}, {c.get('branch', '')}"
                for c in p["did_you_mean"]
            )
            return (f"More than one student matches \"{p['query']}\". "
                    f"Which one do you mean?\n\n{opts}")

        if kind == "ambiguous_class":
            opts = "\n".join(f"- {c['branch']}" for c in p["did_you_mean"])
            return (f"{p['query']} exists at several campuses. Which one?\n\n{opts}")

        if kind == "not_found":
            extra = p.get("available_branches") or p.get("available_classes")
            tail = f"\n\nAvailable: {', '.join(extra)}" if extra else ""
            return f"I have no record matching \"{p['query']}\".{tail}"

        if kind == "unknown_board":
            return (f"No campus uses the \"{p['query']}\" board. "
                    f"We offer: {', '.join(p['available_boards'])}.")

        if kind == "forbidden":
            return ("That information isn't available for your role. "
                    "Please contact the school office.")

        return p.get("message", kind)

    def append_assistant(self, messages: list[dict], reply: LLMReply) -> None:
        messages.append(reply.raw.model_dump())

    def append_tool_result(self, messages: list[dict], call: ToolCall, result: dict) -> None:
        messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name,
                         "content": json.dumps(result, default=str)})


# ---------------------------------------------------------------- factory
def get_llm():
    p = Config.LLM_PROVIDER
    if p == "mock":
        return MockClient()
    if p == "groq":
        # reasoning_effort is only understood by the gpt-oss family; sending it
        # to another model is a 400, so gate on the model name.
        extra = ({"reasoning_effort": Config.GROQ_REASONING_EFFORT}
                 if "gpt-oss" in Config.GROQ_MODEL else {})
        return OpenAICompatClient(
            Config.GROQ_API_KEY, "https://api.groq.com/openai/v1",
            Config.GROQ_MODEL, extra,
        )
    if p == "gemini":
        return OpenAICompatClient(
            Config.GEMINI_API_KEY,
            "https://generativelanguage.googleapis.com/v1beta/openai/",
            Config.GEMINI_MODEL,
        )
    if p == "ollama":
        return OpenAICompatClient("ollama", Config.OLLAMA_BASE_URL, Config.OLLAMA_MODEL)
    if p == "anthropic":
        return AnthropicClient(Config.ANTHROPIC_API_KEY, Config.ANTHROPIC_MODEL)
    raise ValueError(f"Unknown LLM_PROVIDER: {p!r}")
