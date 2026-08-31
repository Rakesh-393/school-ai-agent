# School AI Agent — Flask + RAG + Tool Calling

A working AI assistant for a **multi-campus school group**. Parents, teachers and
admins ask questions in plain English; the agent looks up real answers in a
database and in school policy documents, and refuses to invent anything.

Demo data models Paramita Schools, Karimnagar — **five campuses** across three
boards (State/SSC, CBSE, Cambridge).

Built to be **read and understood**, not just run. Every non-obvious decision is
explained in a comment where the code lives.

---

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate                # Windows.  macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

copy .env.example .env                # macOS/Linux: cp .env.example .env
#  -> put your free Groq key in .env   (https://console.groq.com/keys)

set FLASK_APP=run.py                  # macOS/Linux: export FLASK_APP=run.py
flask seed-db --reset                 # 5 branches, 12 classes, 40 students, 4 months of data
flask ingest-docs                     # build the vector index (first run downloads a ~80MB model)

python run.py                         # -> http://127.0.0.1:5000
```

**No API key yet?** `.env` ships with `LLM_PROVIDER=mock`, so the app runs
immediately. Everything except the language model is real — real database, real
fuzzy matching, real vector search, real formatted answers. Only the "which tool
should I call" decision is keyword-matched instead of reasoned.

Switch to a real model when you have a key: put it in `GROQ_API_KEY` and set
`LLM_PROVIDER=groq`. Mock's limits show up the moment a question is phrased
unusually or needs two tools chained — that gap is exactly what the LLM buys you.

### If chat replies with an error

Check `http://127.0.0.1:5000/api/health` first — it reports whether the LLM is
actually usable, not just whether Flask is up:

| `status` | Meaning | Fix |
|---|---|---|
| `ok` | provider configured | — |
| `llm_not_configured` | key missing or still the `.env.example` placeholder | paste a real key into `.env`, restart |

The startup banner says the same thing. Common runtime failures are translated
into plain English by `_explain()` in [app/routes/chat.py](app/routes/chat.py):
a rejected key, a free-tier rate limit (HTTP 429), an unreachable host, a retired
model name. Anything unrecognised is passed through verbatim so a real bug is
never hidden behind a friendly message.

Ask from the terminal instead of the browser:

```bash
flask ask "How many branches do you have?"
flask ask "Which branch is CBSE?"
flask ask "Tell me about the Jyothinagar campus"
flask ask "What is Kabir Menon's attendance this month?"
flask ask "Which students in 7A at Heritage are below 75%?" --role teacher
flask ask "How do I apply for leave?"
```

---

## What it does

| Question | Path taken |
|---|---|
| "How many branches do you have?" | `list_branches` → SQL count (**never** the model's own count) |
| "Which branch is CBSE?" | `list_branches(board="CBSE")` → 2 campuses |
| "Tell me about Jyothinagar" | alias match → `get_branch_info` → Explorica Premium School |
| "Homework for 7A?" | 5 campuses have Grade 7-A → agent asks **which branch** |
| "Kabir's attendance in July?" | fuzzy-resolve name → `get_attendance_summary` → SQL |
| "Any pending fees for Diya?" | fuzzy-resolve name → `get_fee_status` → SQL |
| "Which students in 7A are below 75%?" | parse class → `get_class_attendance_report` → SQL (**staff only**) |
| "How do I apply for leave?" | `search_school_documents` → vector search → cite the source |
| "What's Aarav's attendance?" | **two** students match → agent asks *which* Aarav |

---

## Architecture

```
Browser  ──POST /api/chat──►  Flask (routes/chat.py)
                                  │
                                  ▼
                        SchoolAgent (agent/orchestrator.py)
                                  │  the agent loop
                    ┌─────────────┴──────────────┐
                    ▼                            ▼
              LLM (agent/llm.py)          Tools (agent/tools.py)
        Groq / Gemini / Ollama /                 │
        Anthropic / mock                ┌────────┴────────┐
                                        ▼                 ▼
                          Resolver (agent/resolver.py)   RAG (agent/rag.py)
                            rapidfuzz entity matching     Chroma + MiniLM
                                        │                 │
                                        ▼                 ▼
                                  SQLAlchemy / SQLite   data/docs/*.md
```

### The technology choices

| Concern | Choice | Why this one |
|---|---|---|
| Web | Flask + blueprints + app factory | Small, explicit, testable without a server |
| ORM / DB | SQLAlchemy + SQLite | Zero setup; change `DATABASE_URL` for MySQL/Postgres |
| LLM (free) | **Groq `llama-3.3-70b-versatile`** | Free tier, very fast, and it does **native tool calling** — non-negotiable for an agent |
| LLM (offline) | Ollama `qwen2.5:7b` | Runs on your laptop, no key, no network |
| LLM (upgrade) | Anthropic `claude-opus-5` | Adapter already written — flip one env var |
| Embeddings | `all-MiniLM-L6-v2` via ONNX | Free, local, 384-dim, and avoids a ~2 GB PyTorch install |
| Vector store | ChromaDB (persistent) | File-based, no server, good enough to millions of chunks |
| Fuzzy matching | **RapidFuzz** | C++ fast; `fuzzywuzzy` is the slow, abandoned ancestor |

---

## The three ideas worth learning here

### 1. Tools, not text-to-SQL

The popular tutorial approach is "let the LLM write SQL". Don't — not with student
records. One bad generation and you have leaked another family's fee history, or
dropped a table.

Instead the LLM chooses from a **whitelist of Python functions** with typed
arguments ([app/agent/tools.py](app/agent/tools.py)). Every query is hand-written and
parameterised. That gives you:

- zero SQL-injection surface
- **role-based access control** — `tools_for_role("parent")` hides
  `get_class_attendance_report` entirely, and `execute()` re-checks it in case
  the model hallucinates the name
- tools you can unit-test with **no LLM at all** (`scripts/test_tools.py`)

### 2. Fuzzy matching as a three-state answer

[app/agent/resolver.py](app/agent/resolver.py). The LLM hands you a messy string;
your job is to turn it into an id — *or admit you can't*.

```
"Aarav Sharma"   → matched     (exact match short-circuits, before any scoring)
"kabirr menonn"  → matched     (92, and nothing else is close)
"maths"          → matched     (80, and it's the only candidate → accept)
"aarav"          → ambiguous   (Aarav Sharma 100 vs Aaravi Sharma 90 — too close to call)
"Zaphod"         → not_found   (nothing clears the floor)
```

The tuning knobs live in [config.py](config.py):

| Constant | Meaning |
|---|---|
| `FUZZY_ACCEPT = 88` | score needed to be considered a confident match |
| `FUZZY_MARGIN = 12` | ...**and** it must beat runner-up by this much |
| `FUZZY_SOLO = 75` | a lone candidate this good is accepted without a margin test |
| `FUZZY_SUGGEST = 65` | floor for appearing in "did you mean" |

The margin rule is the important one. `"aarav"` scores 100 against *Aarav Sharma*
and 90 against *Aaravi Sharma* — a single character apart, a different child.
Without the margin test the agent silently answers about the wrong student. With
it, the tool returns `ambiguous_name` **as data**, and the system prompt tells the
model to ask which one. Refusing to guess is a feature.

**One row, many names.** A person has one name; an *institution* has many —
the official name, the locality people actually say, the board, the short code.
`Branch.aliases` holds them all, and `resolve(..., alias_attr="alias_list")`
scores the query against every alias and keeps each row's best. That is why all
of these land on the right campus:

```
explorica          → Explorica Premium School
jyothinagar        → Explorica Premium School     (locality, not in the name)
the cambridge one  → Explorica Premium School     (board, not in the name)
iris               → Paramita World School        (former name)
alugunoor          → Paramita World School
Hogwarts           → not_found
```

Note it does **not** use `process.extract` for this. That maps one *choice string*
to one result, so two aliases of the same branch would return as two separate
candidates and the margin test would see a row tie with itself. Score per row,
take the max.

When a real user's phrasing fails, add an alias — that is far safer than
loosening `FUZZY_ACCEPT`.

**Which scorer?** RapidFuzz offers several and the choice matters:

| Scorer | `"sharma aarav"` vs `"Aarav Sharma"` | Use for |
|---|---|---|
| `ratio` | 45 | short exact-ish strings |
| `partial_ratio` | 100 | substring / prefix input |
| `token_sort_ratio` | 100 | word order differs |
| `token_set_ratio` | 100 | word order **and** extra words differ |
| `WRatio` | 96 | general-purpose default |

We take `max(WRatio, token_set_ratio)` because they fail on different inputs.

**And know when *not* to use it.** Class labels are structured, so `_find_class()`
parses `"7B"` / `"VII B"` / `"grade 7-b"` with a regex first. `WRatio("7A", "7B")`
is 50 — fuzzy matching on short codes is actively dangerous. *Structure first,
fuzz second.*

**Uniqueness is scoped.** Once there are five campuses, `"Grade 7-A"` is no longer
a key — it exists at all of them. `_find_class` returns `ambiguous_class` listing
every campus, and the agent asks which one. A field that was unique yesterday
stops being unique the moment you add a tenant dimension; go re-check every
lookup when you do.

### 3. RAG quality is mostly chunking

[app/agent/rag.py](app/agent/rag.py). The first version used plain 800-character
chunks. Asking *"how do I apply for leave"* retrieved the paragraph about
**lunch-break attendance marking** — from the right document, the wrong section.

Two fixes took the top-hit similarity from 0.32 to 0.49, and made it the correct
section:

1. **Split on markdown headings first**, size-split only oversized sections.
   Headings are authored semantic boundaries — respect them.
2. **Prepend a breadcrumb** to every chunk:
   `Attendance Policy > Applying for leave`. Now the heading words are inside the
   embedded vector, so a question phrased like the heading actually matches it —
   and the model can cite the section.

Before you reach for a bigger embedding model, fix your chunking.

---

## Project layout

```
config.py                  all tunables, read from .env
run.py                     entry point
app/
  __init__.py              app factory + flask CLI commands
  models.py                SQLAlchemy schema
  seed.py                  demo data: BRANCHES, SECTIONS_BY_BRANCH; anchored to TODAY
  routes/
    web.py                 serves the chat page
    chat.py                POST /api/chat, /api/reset, /api/health
  agent/
    orchestrator.py        THE AGENT LOOP + system prompt
    llm.py                 provider abstraction (Groq/Gemini/Ollama/Anthropic/mock)
    tools.py               tool registry, schemas, RBAC
    resolver.py            fuzzy entity resolution
    rag.py                 chunking, embedding, vector search
  templates/chat.html      single-file UI, shows the tool trace
data/docs/                 the RAG corpus — drop .md/.txt/.pdf here
  branches_and_campuses.md   the five campuses, in prose, for RAG
scripts/test_tools.py      exercise tools + resolver with NO LLM
```

---

## Extending it

**Add a capability** — one entry in `TOOLS`:

```python
{
    "name": "get_bus_route_info",
    "description": "Pickup point and timing for a student's bus route.",
    "fn": get_bus_route_info,
    "allowed_for": {"parent", "admin"},
    "input_schema": {
        "type": "object",
        "properties": {"student_name": {"type": "string"}},
        "required": ["student_name"],
        "additionalProperties": False,
    },
}
```

Write the function, test it in `scripts/test_tools.py`, done. The orchestrator
and the UI need no changes.

**Tool descriptions are prompt engineering.** The `description` field is how the
model decides whether to call your tool. Vague description, wrong tool chosen.

**Add a branch** — one dict in `seed.BRANCHES` (give it generous `aliases`), plus
its sections in `SECTIONS_BY_BRANCH`. Re-run `flask seed-db --reset`. Also add it
to `data/docs/branches_and_campuses.md` so RAG-style questions about campuses work.

**Add documents** — drop files in `data/docs/`, run `flask ingest-docs`.

**Switch model** — edit `LLM_PROVIDER` in `.env`. Nothing else.

---

## Where to go next

1. **Real auth, scoped to a branch.** Right now `role` comes from a dropdown the
   user controls. Add Flask-Login and derive the role, the parent's own children,
   *and their branch* from the session — then filter `_find_student` to that
   parent's children and default `branch` to the one they belong to. That single
   change removes most of the ambiguity the agent currently has to ask about, and
   it is the most important production fix on this list.
2. **Streaming.** Responses arrive all at once. Server-Sent Events would make it
   feel far faster.
3. **Evaluation.** `chat_logs` records every question, answer, tool and latency.
   Turn 50 of those rows into a fixed test set and measure before/after whenever
   you change a prompt or a threshold. Vibes are not a metric.
4. **Rate limiting** on `/api/chat` — free tiers run out.
5. **Postgres + Alembic** when the schema starts changing.
6. **A reranker** (`bge-reranker-base`) over the top-20 Chroma hits if retrieval
   quality stops improving with better chunking.

---

## Costs

Groq's free tier and Ollama cost nothing. If you move to `claude-opus-5`, a
typical question here is ~2–4k input tokens and a few hundred output — set
`ANTHROPIC_MODEL=claude-haiku-4-5` first if cost matters more than answer
quality, and measure the difference against your eval set before deciding.
