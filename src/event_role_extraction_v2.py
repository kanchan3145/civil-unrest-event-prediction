
import re


ENTITY_PATTERNS = {
    "farmers": r"\bfarmers?\b",
    "students": r"\bstudents?\b",
    "workers": r"\bworkers?\b",
    "protesters": r"\bprotesters?\b",
    "police": r"\bpolice\b",
    "security forces": r"\bsecurity forces?\b",
    "government": r"\bgovernment\b|\bgovt\b",
    "administration": r"\badministration\b",
    "farm laws": r"\bfarm laws?\b"
}


def extract_event_roles(text):
    text = str(text)

    actors = set()
    targets = set()
    responders = set()
    mentioned = set()

    for entity, pattern in ENTITY_PATTERNS.items():
        if re.search(pattern, text, re.I):
            mentioned.add(entity)

    # Actor: explicitly participates in an event.
    actor_pattern = (
        r"\b(farmers?|students?|workers?|protesters?)"
        r"\s+(?:have\s+|had\s+|are\s+|were\s+)?"
        r"(?:protested|protest|marched|rallied|"
        r"demonstrated|went on strike)\b"
    )

    for match in re.finditer(actor_pattern, text, re.I):
        actor = match.group(1).lower()

        if actor.endswith("s"):
            normalized = actor
        else:
            normalized = actor + "s"

        actors.add(normalized)

    # Targets must have an explicit opposition relation.
    target_pattern = (
        r"\b(?:against|opposing|demanding repeal of)\s+"
        r"(?:the\s+)?"
        r"(government|govt|police|administration|farm laws)\b"
    )

    for match in re.finditer(target_pattern, text, re.I):
        target = match.group(1).lower()

        if target == "govt":
            target = "government"

        targets.add(target)

    # Responders: explicitly described as taking action.
    responder_pattern = (
        r"\b(police|security forces)\s+"
        r"(?:have\s+|had\s+|are\s+|were\s+)?"
        r"(?:deployed|detained|arrested|dispersed|"
        r"fired|used tear gas|lathi charged)\b"
    )

    for match in re.finditer(
        responder_pattern,
        text,
        re.I
    ):
        responders.add(
            match.group(1).lower()
        )

    unresolved = (
        mentioned
        - actors
        - targets
        - responders
    )

    return {
        "event_actors": "|".join(sorted(actors)),
        "event_targets": "|".join(sorted(targets)),
        "responding_entities": "|".join(
            sorted(responders)
        ),
        "unresolved_entities": "|".join(
            sorted(unresolved)
        )
    }


if __name__ == "__main__":

    example = (
        "Farmers protested against the government "
        "while police deployed security personnel."
    )

    print(extract_event_roles(example))
