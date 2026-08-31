"""
ENTITY RESOLUTION  --  the safety layer between the LLM and the database.

WHY THIS EXISTS
---------------
An LLM extracts entities from messy human text. A parent types:

    "wats aravs attendnce in july"

The LLM will confidently hand your tool  name="arav"  -- which matches nothing
in `students.full_name`, because the real row says "Aarav Sharma".

If you just do  WHERE full_name = 'arav'  you get zero rows and the agent
hallucinates an answer. If you do  WHERE full_name LIKE '%arav%'  you get
"Aaravi Menon" too and leak another child's data to the wrong parent.

So: never let the LLM pick a primary key. It gives us a *string*; this module
turns that string into an *id* (or an honest "I'm not sure which one").

HOW RAPIDFUZZ SCORES
--------------------
rapidfuzz gives several scorers. Which one you pick matters a lot:

  ratio           plain Levenshtein similarity. "aarav sharma" vs "sharma aarav" -> 45. Bad for names.
  partial_ratio   best matching substring. "sharma" vs "Aarav Sharma" -> 100. Good for partial input.
  token_sort_ratio  sorts words first, then compares. Fixes word order. -> 100.
  token_set_ratio   set intersection of words. Ignores order AND extra words. Best for names.
  WRatio          a weighted blend of all of the above. Best general-purpose default.

We score with BOTH `WRatio` and `token_set_ratio` and keep the higher one,
because they fail in different ways.

THREE-WAY OUTCOME (this is the important design idea)
-----------------------------------------------------
Fuzzy matching is not "found / not found". It is three states:

  score >= FUZZY_ACCEPT (88)   -> confident. Use it.
  score >= FUZZY_SUGGEST (65)  -> ambiguous. Ask the user "did you mean X or Y?"
  below                        -> nothing. Say so, don't invent.

Returning the ambiguous case to the LLM *as data* is what makes the agent feel
smart -- it asks a clarifying question instead of guessing.
"""
from dataclasses import dataclass, field
from typing import Any, Sequence

from rapidfuzz import fuzz, process, utils

from config import Config


@dataclass
class Candidate:
    id: int
    label: str
    score: float
    extra: dict = field(default_factory=dict)


@dataclass
class Resolution:
    """status is one of: matched | ambiguous | not_found"""
    status: str
    best: Candidate | None = None
    candidates: list[Candidate] = field(default_factory=list)
    query: str = ""

    def as_tool_payload(self) -> dict:
        """Shape handed back to the LLM when resolution fails."""
        if self.status == "matched":
            return {"resolved": self.best.label, "id": self.best.id}
        if self.status == "ambiguous":
            return {
                "error": "ambiguous_name",
                "query": self.query,
                "message": "Multiple records look similar. Ask the user which one they mean.",
                "did_you_mean": [
                    {"name": c.label, "score": round(c.score), **c.extra}
                    for c in self.candidates
                ],
            }
        return {
            "error": "not_found",
            "query": self.query,
            "message": f"No record resembling '{self.query}' exists. Do not guess.",
        }


def _score(query: str, choice: str) -> float:
    """Take the more forgiving of two scorers -- they fail on different inputs."""
    q = utils.default_process(query)     # lowercase, strip punctuation, collapse space
    c = utils.default_process(choice)
    return max(fuzz.WRatio(q, c), fuzz.token_set_ratio(q, c))


def _resolve_with_aliases(
    query: str,
    rows: Sequence[Any],
    label_attr: str,
    alias_attr: str,
    extra_fn,
) -> Resolution:
    """
    One row, many names: score the query against every alias, keep the row's best.

    Note we do NOT use process.extract here. That maps one choice string to one
    result, so two aliases of the same branch would come back as two separate
    "candidates" and the margin test would see a tie between a row and itself.
    Scoring per row and taking the max collapses them correctly.
    """
    cands: list[Candidate] = []
    for row in rows:
        aliases = getattr(row, alias_attr) or []
        best_alias, best = "", 0.0
        for alias in aliases:
            s = _score(query, alias)
            if s > best:
                best_alias, best = alias, s

        if best < Config.FUZZY_SUGGEST:
            continue

        extra = extra_fn(row) if extra_fn else {}
        # Surface which alias matched -- invaluable when debugging a bad match.
        if best_alias and best_alias.lower() != getattr(row, label_attr).lower():
            extra = {**extra, "matched_on": best_alias}
        cands.append(Candidate(row.id, getattr(row, label_attr), best, extra))

    cands.sort(key=lambda c: c.score, reverse=True)
    return _decide(cands, query)


def _decide(cands: list[Candidate], query: str) -> Resolution:
    """Shared accept / ask / reject logic. See the three-way outcome note above."""
    if not cands:
        return Resolution(status="not_found", query=query)

    best = cands[0]

    # Only ONE plausible candidate: no one to confuse it with.
    if len(cands) == 1:
        if best.score >= Config.FUZZY_SOLO:
            return Resolution(status="matched", best=best, candidates=cands, query=query)
        return Resolution(status="ambiguous", candidates=cands, query=query)

    if best.score >= Config.FUZZY_ACCEPT and (best.score - cands[1].score) >= Config.FUZZY_MARGIN:
        return Resolution(status="matched", best=best, candidates=cands, query=query)

    return Resolution(status="ambiguous", candidates=cands[:3], query=query)


def resolve(
    query: str,
    rows: Sequence[Any],
    label_attr: str = "full_name",
    extra_fn=None,
    limit: int = 5,
    alias_attr: str | None = None,
) -> Resolution:
    """
    Match a free-text `query` against `rows` (any objects with `.id` and a label attr).

    `extra_fn(row) -> dict` adds disambiguating context to suggestions, e.g. the
    class name -- so the user sees "Aarav Sharma (Grade 7-B)" not two identical names.

    `alias_attr` names a property returning a LIST of strings the row can also be
    called. Each row is then scored against every alias and keeps its best. This
    is what lets "jyothinagar", "explorica" and "cambridge branch" all resolve to
    Explorica Premium School, none of which the official name contains.

    A person has one name; an organisation has many. Reach for aliases whenever
    the thing being matched is a place, a product or an institution.
    """
    query = (query or "").strip()
    if not query or not rows:
        return Resolution(status="not_found", query=query)

    if alias_attr:
        return _resolve_with_aliases(query, rows, label_attr, alias_attr, extra_fn)

    # --- Step 0: exact match wins outright ---------------------------------
    # Without this, "Aarav Sharma" scores 100 against itself and 96 against
    # "Aaravi Sharma", the margin rule below rejects it, and the agent asks a
    # pointless clarifying question about a name the user typed perfectly.
    norm = utils.default_process(query)
    exact = [r for r in rows if utils.default_process(getattr(r, label_attr)) == norm]
    if len(exact) == 1:
        row = exact[0]
        return Resolution(
            status="matched",
            best=Candidate(row.id, getattr(row, label_attr), 100.0,
                           extra_fn(row) if extra_fn else {}),
            query=query,
        )
    # len(exact) > 1 means genuine duplicates -- fall through to the ambiguous path.

    choices = {i: getattr(r, label_attr) for i, r in enumerate(rows)}

    # process.extract does the N-way comparison in C++ -- fast even on 10k students.
    hits = process.extract(
        query,
        choices,
        scorer=fuzz.WRatio,
        processor=utils.default_process,
        limit=limit,
        score_cutoff=Config.FUZZY_SUGGEST,
    )

    cands: list[Candidate] = []
    for label, _wratio, idx in hits:
        row = rows[idx]
        cands.append(
            Candidate(
                id=row.id,
                label=label,
                score=_score(query, label),          # re-score with the blended scorer
                extra=(extra_fn(row) if extra_fn else {}),
            )
        )
    cands.sort(key=lambda c: c.score, reverse=True)
    return _decide(cands, query)
