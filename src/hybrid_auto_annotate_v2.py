
from pathlib import Path
import re
import pandas as pd
import numpy as np

from extract_event_evidence_v3 import analyze_post, clean_text

BASE = Path(__file__).resolve().parent.parent / "data" / "annotation"

INPUT_FILE = BASE / "additional_auto_labels.csv"
OUTPUT_FILE = BASE / "additional_hybrid_labels_v2.csv"

PATTERNS = {
    "incident": [
        r"\b(?:clashes?|clashed)\s+(?:with|between|during|at)\b",
        r"\b(?:injured|wounded|killed|beaten|attacked)\b",
        r"\b(?:arrested|detained|lathi[- ]charged)\b",
        r"\b(?:police|cops)\s+(?:fired|used)\s+tear\s*gas\b",
        r"\b(?:pelting|pelted)\s+stones?\b",
        r"\b(?:blocking|blocked)\s+(?:roads?|highways?|traffic)\b",
        r"\b(?:climbed|climbing)\s+on(?:to)?\s+"
        r"(?:a\s+|the\s+)?(?:police|water cannon)",
    ],
    "ongoing_protest": [
        r"\b(?:have|has)\s+been\s+protesting\b",
        r"\b(?:are|were)\s+protesting\b",
        r"\bprotests?\s+(?:happening|underway|ongoing)\b",
        r"\b(?:farmers|protesters|workers|students)\s+"
        r"(?:marched|marching|gathered|protested)\b",
        r"\b(?:thousands?|hundreds?)\s+"
        r"(?:attend|attended|joined|gathered)\b",
        r"\b(?:held|staged|organized|organised)\s+"
        r"(?:a\s+|the\s+)?(?:rally|march|protest|strike)\b",
        r"\b(?:farmers|protesters)\s+(?:are|were)\s+"
        r"(?:at|on)\s+(?:the\s+)?forefront\s+of\s+protests?\b",
    ],
    "planned": [
        r"\b(?:will|shall)\s+(?:march|protest|hold|stage|organize)\b",
        r"\b(?:planned|planning|scheduled|upcoming|proposed)\s+"
        r"(?:protest|rally|march|strike)\b",
        r"\b(?:march|rally|protest|strike)\s+"
        r"(?:has\s+been|had\s+been|was|is)\s+"
        r"(?:postponed|cancelled|scheduled)\b",
        r"\b(?:will|may|might)\s+continue\s+"
        r"(?:their\s+|the\s+)?(?:agitation|protest)\b",
        r"\b(?:proceed|go\s+ahead)\s+with\s+the\s+march\b",
    ],
    "related": [
        r"\brepeal(?:ed|ing|s)?\s+(?:the\s+)?farm\s+laws?\b",
        r"\bfarm\s+laws?\s+(?:were|have\s+been)\s+repealed\b",
        r"\b(?:government|govt|goi|pm)\b.{0,55}"
        r"\b(?:repeal|repealed|repealing)\b",
        r"\b(?:government|govt|goi)\b.{0,45}"
        r"\b(?:announced|decided|approved)\b",
    ],
    "fact_check": [
        r"\bfact[\s-]*check\b",
        r"\bpassed\s+off\s+as\b",
        r"\b(?:fake|false|misleading)\s+(?:video|claim|news)\b",
        r"\bno\s+evidence\s+(?:of|that)\b",
    ],
    "ambiguous_claim": [
        r"\b(?:allegedly|unconfirmed|rumou?red)\b",
        r"\b(?:accused|accuses|allegations?)\b",
        r"\b(?:missing\s+farmers?|farmer\s+genocide)\b",
        r"\b(?:responsible\s+for|involved\s+in)\s+riots?\b",
        r"\b(?:farmers?\s+deaths?|atrocities)\b",
        r"\b(?:planning|trying)\s+(?:a\s+)?genocide\b",
    ],
    "context": [
        r"\bfarmers?\b",
        r"\bprotest(?:ers?|ing|s)?\b",
        r"\bpolice\b",
        r"\b(?:march|rally|strike|demonstration|agitation)\b",
        r"\b(?:clash|riot|blockade)\b",
    ],
}

COMPILED = {
    key: [re.compile(p, re.I) for p in patterns]
    for key, patterns in PATTERNS.items()
}


def get_hits(text, category):
    hits = []
    for pattern in COMPILED[category]:
        hits.extend(m.group(0) for m in pattern.finditer(text))
    return list(dict.fromkeys(hits))


def classify_v2(row):
    text = clean_text(row.get("text", ""))
    v3 = analyze_post(row)

    hits = {
        category: get_hits(text, category)
        for category in PATTERNS
    }

    incident = hits["incident"]
    ongoing = hits["ongoing_protest"]
    planned = hits["planned"]
    related = hits["related"]
    fact_check = hits["fact_check"]
    ambiguous = hits["ambiguous_claim"]

    context = bool(hits["context"])

    # Categories are provisional machine predictions.
    # Fact-check context takes priority over incident keywords.
    if not text:
        category = "uncertain"
        reason = "Empty text"

    elif fact_check:
        category = "uncertain"
        reason = "Fact-check or possible misattribution"

    elif incident and ambiguous:
        category = "uncertain"
        reason = "Incident expression with ambiguous attribution"

    elif incident and context:
        category = "actual_incident"
        reason = "Explicit incident action detected"

    elif ongoing and context:
        category = "actual_incident"
        reason = "Ongoing or reported protest activity"

    elif planned:
        category = "planned_event"
        reason = "Future, proposed or postponed event"

    elif related:
        category = "related_action"
        reason = "Related government or organizational action"

    elif v3["event_evidence_decision"] == "related_action":
        category = "related_action"
        reason = "V3 related-action evidence"

    elif ambiguous and context:
        category = "uncertain"
        reason = "Ambiguous civil-unrest claim"

    elif v3["event_evidence_decision"] == "event_candidate":
        # V3-only evidence is not always reliable.
        category = "uncertain"
        reason = "V3 candidate without corroborating V2 expression"

    elif v3["event_evidence_decision"] == "planned_or_future":
        category = "uncertain"
        reason = "V3 future cue without corroborating V2 expression"

    else:
        category = "non_event"
        reason = "No sufficiently specific event evidence"

    target_map = {
        "actual_incident": 1,
        "planned_event": 0,
        "related_action": 0,
        "non_event": 0,
        "uncertain": np.nan,
    }

    matched = []
    for key, values in hits.items():
        if key != "context":
            matched.extend(values)

    return pd.Series({
        "hybrid_v2_category": category,
        "hybrid_v2_target": target_map[category],
        "hybrid_v2_reason": reason,
        "hybrid_v2_evidence": "; ".join(
            dict.fromkeys(matched)
        ),
        "hybrid_v2_source": "automatic_rules_v2",
        "hybrid_v2_human_verified": False,
    })


def main():
    print("=" * 65)
    print("STAGE 7.5 - IMPROVED HYBRID ANNOTATION V2")
    print("=" * 65)

    df = pd.read_csv(
        INPUT_FILE,
        dtype={"post_id": "string"}
    )

    if df["post_id"].isna().any():
        raise ValueError("Missing post IDs")

    if not df["post_id"].is_unique:
        raise ValueError("Duplicate post IDs")

    results = df.apply(classify_v2, axis=1)

    output = pd.concat(
        [
            df.reset_index(drop=True),
            results.reset_index(drop=True)
        ],
        axis=1
    )

    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT_FILE}"
        )

    output.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nTOTAL RECORDS:", len(output))

    print("\nV2 CATEGORY DISTRIBUTION:")
    print(
        output["hybrid_v2_category"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nV2 BINARY TARGETS:")
    print(
        output["hybrid_v2_target"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nSAVED:", OUTPUT_FILE)
    print("\nSTAGE 7.5 COMPLETED")


if __name__ == "__main__":
    main()
