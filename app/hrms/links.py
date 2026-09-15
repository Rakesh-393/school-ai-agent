"""
Static links and contacts the agent is allowed to give out.

WHY A FILE AND NOT A PROMPT LINE
A URL in the system prompt is a suggestion; a language model will cheerfully
produce a plausible neighbouring one -- /careers/apply instead of /apply,
a stale subdomain, a path that has never existed. A wrong link is worse than no
link, because the person follows it and blames the system rather than asking
again. Here a URL is either present and correct, or the tool says it does not
have one. There is no third outcome where the model fills the gap.

TO ADD OR CHANGE A LINK
Edit this dict. Give each entry the aliases people actually type, so the lookup
matches "where do I apply" and "job application" and "careers page" to the same
answer. Nothing else in the codebase needs to change.

The URLs below are placeholders until you confirm the real ones -- every entry
is marked `confirmed: False`, and the tool passes that flag through so the agent
can hedge instead of stating a guess as fact.
"""

RESOURCES: dict[str, dict] = {
    "application_form": {
        "title": "Job application form",
        "url": "https://mcb.keka.com/careers/",
        "description": "Browse open roles and apply.",
        "aliases": [
            "application form", "apply", "how to apply", "job application",
            "careers", "career page", "job openings", "vacancy form",
            "recruitment form", "where do i apply",
        ],
        "confirmed": False,
    },
    "hrms_portal": {
        "title": "HRMS portal",
        "url": "",
        "description": "Sign in to the HR system.",
        "aliases": [
            "hrms", "portal", "hrms login", "employee portal", "self service",
            "ess", "login link",
        ],
        "confirmed": False,
    },
    "hr_helpdesk": {
        "title": "HR helpdesk",
        "url": "",
        "description": "Contact HR for anything this assistant cannot answer.",
        "aliases": [
            "hr contact", "hr helpdesk", "hr email", "contact hr", "hr support",
            "raise a ticket", "who do i ask",
        ],
        "confirmed": False,
    },
}


def _score(query: str, entry_key: str, entry: dict) -> int:
    """Crude containment match. Good enough: the alias lists are short and hand-written."""
    q = query.strip().lower()
    if not q:
        return 0
    candidates = [entry_key.replace("_", " "), entry["title"].lower(), *entry["aliases"]]
    best = 0
    for cand in candidates:
        c = cand.lower()
        if q == c:
            best = max(best, 100)
        elif c in q or q in c:
            # Longer overlaps win, so "hr helpdesk" beats a bare "hr".
            best = max(best, 50 + min(len(c), 40))
    return best


def resource_link(topic: str | None = None) -> dict:
    """
    The link for a topic, or the whole list when no topic is given.

    An entry with an empty `url` is reported as missing rather than guessed at,
    and `confirmed` tells the agent whether a human has verified the address.
    """
    available = [
        {"key": k, "title": v["title"], "url": v["url"],
         "description": v["description"], "confirmed": v["confirmed"]}
        for k, v in RESOURCES.items()
    ]

    if not topic:
        return {"links": [a for a in available if a["url"]],
                "unavailable": [a["key"] for a in available if not a["url"]]}

    ranked = sorted(
        ((_score(topic, k, v), k, v) for k, v in RESOURCES.items()),
        key=lambda t: t[0], reverse=True,
    )
    score, key, entry = ranked[0]
    if score == 0:
        return {
            "error": "unknown_topic",
            "query": topic,
            "message": "No link on record for that. Do not invent one.",
            "available_topics": [a["key"] for a in available],
        }
    if not entry["url"]:
        return {
            "error": "link_not_configured",
            "topic": key,
            "title": entry["title"],
            "message": (f"{entry['title']} is a known topic but no URL is on record. "
                        "Say the link is not available and suggest contacting HR. "
                        "Do not guess a URL."),
        }
    return {
        "topic": key,
        "title": entry["title"],
        "url": entry["url"],
        "description": entry["description"],
        "confirmed": entry["confirmed"],
        "note": None if entry["confirmed"] else
                "This URL has not been verified yet -- offer it, but say so.",
    }
