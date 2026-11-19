
from pathlib import Path
import re
import html
import pandas as pd

BASE = Path(__file__).resolve().parent.parent / "data" / "annotation"

INPUT_FILE = BASE / "additional_hybrid_labels_v2.csv"
OUTPUT_FILE = BASE / "pseudo_label_selection_v3.csv"


def normalize_text(value):
    if pd.isna(value):
        return ""

    text = html.unescape(str(value)).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\[(?:user|url)\]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def compile_patterns(patterns):
    return [re.compile(p, re.I) for p in patterns]


# Explicit protest actions.
STRONG_ACTIONS = compile_patterns([
    r"\b(?:clash|clashes|clashed|clashing)\s+"
    r"(?:with|between|amid|during)\b",

    r"\b(?:clashes|clash)\s+broke\s+out\b",

    r"\b(?:police|cops|officers?)\b.{0,65}"
    r"\b(?:tear[\s-]*gas|lathi[\s-]*charge|"
    r"arrested|detained|fired)\b",

    r"\b(?:lathi[\s-]*charg(?:e|ed|ing)|"
    r"tear[\s-]*gassed)\b",

    r"\b(?:farmers?|protesters?)\b.{0,55}"
    r"\b(?:arrested|detained|injured|"
    r"attacked|beaten|assaulted)\b",

    r"\b(?:arrested|detained|injured)\b.{0,55}"
    r"\b(?:farmers?|protesters?|police|cops)\b",

    r"\b(?:farmers?|protesters?)\b.{0,65}"
    r"\b(?:broke|removed|crossed|"
    r"knocked\s+down|ran\s+over)\b.{0,35}"
    r"\b(?:barricades?|barriers?)\b",

    r"\b(?:stone[\s-]*pelting|"
    r"pelt(?:ed|ing)\s+stones?)\b",
])


# Actions indicating an actual protest or march.
ONGOING_ACTIONS = compile_patterns([
    r"\b(?:farmers?|protesters?|workers?)\s+"
    r"(?:are|were|have\s+been|had\s+been|"
    r"still)\s+protesting\b",

    r"\b(?:farmers?|protesters?)\s+"
    r"(?:marched|marching|gathered|assembled)\b",

    r"\b(?:farmers?|protesters?)\s+"
    r"(?:held|staged|organized|organised)\b"
    r".{0,40}\b(?:rally|march|protest)\b",

    r"\b(?:thousands?|hundreds?)\s+"
    r"(?:of\s+)?(?:farmers?|protesters?)\b"
    r".{0,45}\b(?:marched|gathered|"
    r"joined|attended|protested)\b",

    r"\b(?:thousands?|hundreds?)\s+"
    r"(?:attend|attended|joined)\b"
    r".{0,50}\b(?:mahapanchayat|rally|"
    r"protest|march)\b",

    r"\b(?:farmers?|protesters?)\b.{0,45}"
    r"\b(?:blocked|blocking|occupied)\b"
    r".{0,30}\b(?:roads?|highways?|"
    r"traffic|borders?)\b",

    r"\b(?:protest|march|rally)\s+"
    r"(?:is|was)\s+(?:underway|ongoing|"
    r"happening|taking\s+place)\b",

    r"\b(?:protesters?|farmers?)\b.{0,45}"
    r"\b(?:participated|took\s+part)\b"
    r".{0,40}\b(?:rally|march|protest)\b",
])


# A separate event participant or setting is needed
# for many strong action matches.
PROTEST_ACTORS = re.compile(
    r"\b(?:farmers?|protesters?|kisan|kisaan|"
    r"agitating\s+farmers?|farm\s+unions?)\b",
    re.I
)

PROTEST_SETTING = re.compile(
    r"\b(?:farm\s+laws?|tractor\s+(?:rally|march)|"
    r"singhu|tikri|ghazipur|mahapanchayat|"
    r"farmers?\s+protest)\b",
    re.I
)


# Exclude obvious unrelated event contexts.
UNRELATED = re.compile(
    r"\b(?:galwan\s+valley|"
    r"china\s+and\s+india\s+clash|"
    r"clashes\s+with\s+islamists|"
    r"pakistan\s+during\s+clashes|"
    r"bjp[\s-]*tmc\s+clash|"
    r"gujarat\s+riots|"
    r"amritsar\s+massacre)\b",
    re.I
)

DISPUTED = re.compile(
    r"\b(?:fact[\s-]*check|"
    r"passed\s+off\s+as|"
    r"fake\s+(?:video|news|claim)|"
    r"false\s+claim|"
    r"unconfirmed|rumou?red)\b",
    re.I
)

QUESTION = re.compile(
    r"\b(?:why\s+are\s+farmers?\s+protesting|"
    r"where\s+are\s+(?:the\s+)?farmers?\s+marching|"
    r"for\s+how\s+long\s+will\s+"
    r"(?:they|farmers?)\s+protest)\b",
    re.I
)

FUTURE = re.compile(
    r"\b(?:will|shall|may|might|"
    r"plan(?:ned|ning)?\s+to)\s+"
    r"(?:march|protest|hold|stage|"
    r"organize|organise|join)\b",
    re.I
)

INVITATION = re.compile(
    r"\b(?:join\s+(?:us|the\s+march|"
    r"citizens['’]?\s+march)|"
    r"come\s+to\s+(?:the\s+)?(?:rally|march)|"
    r"upcoming\s+(?:rally|march|protest))\b",
    re.I
)

# Broad action terms alone should not establish an incident.
WEAK_ONLY = re.compile(
    r"\b(?:tractor\s+rally|tractor\s+march|"
    r"are\s+protesting|injured|attacked|"
    r"killed|clash|protest)\b",
    re.I
)


def get_matches(text, patterns):
    matches = []

    for pattern in patterns:
        for match in pattern.finditer(text):
            matches.append(match.group(0))

    return list(dict.fromkeys(matches))


def select_label(row):
    text = normalize_text(row.get("text", ""))

    def decision(status, reason, evidence=""):
        return pd.Series({
            "selection_status": status,
            "selection_reason": reason,
            "selection_evidence": evidence
        })

    if not text:
        return decision("abstain", "Empty text")

    if UNRELATED.search(text):
        return decision(
            "abstain",
            "Possible unrelated incident"
        )

    if DISPUTED.search(text):
        return decision(
            "abstain",
            "Disputed or unconfirmed report"
        )

    strong = get_matches(text, STRONG_ACTIONS)
    ongoing = get_matches(text, ONGOING_ACTIONS)

    actor = bool(PROTEST_ACTORS.search(text))
    setting = bool(PROTEST_SETTING.search(text))

    future = bool(
        FUTURE.search(text) or INVITATION.search(text)
    )

    question = bool(QUESTION.search(text))

    # A future announcement or rhetorical question
    # may coexist with a real past event.
    # Such mixed cases are not automatically labeled.
    if future or question:
        return decision(
            "abstain",
            "Future, question or mixed temporal context",
            "; ".join(strong + ongoing)
        )

    # Strong action + relevant participants or setting.
    if strong and actor and setting:
        return decision(
            "selected_positive",
            "Strong protest-related incident evidence",
            "; ".join(strong)
        )

    # Ongoing activity + contextual support.
    if ongoing and actor and setting:
        return decision(
            "selected_positive",
            "Explicit ongoing protest activity",
            "; ".join(ongoing)
        )

    if strong or ongoing:
        return decision(
            "abstain",
            "Event evidence lacks sufficient relevance",
            "; ".join(strong + ongoing)
        )

    if WEAK_ONLY.search(text):
        return decision(
            "abstain",
            "Weak or generic event mention"
        )

    return decision(
        "abstain",
        "No sufficiently reliable positive evidence"
    )


def main():
    print("=" * 65)
    print("STAGE 7.13 - EVIDENCE-BASED LABEL FILTER")
    print("=" * 65)

    df = pd.read_csv(
        INPUT_FILE,
        dtype={"post_id": "string"}
    )

    if df["post_id"].isna().any():
        raise ValueError("Missing post IDs")

    if not df["post_id"].is_unique:
        raise ValueError("Duplicate post IDs")

    selections = df.apply(select_label, axis=1)

    result = pd.concat(
        [
            df.reset_index(drop=True),
            selections.reset_index(drop=True)
        ],
        axis=1
    )

    result["selected_binary_target"] = pd.NA

    result.loc[
        result["selection_status"] == "selected_positive",
        "selected_binary_target"
    ] = 1

    result["selected_binary_target"] = pd.to_numeric(
        result["selected_binary_target"],
        errors="coerce"
    )

    result["label_source"] = "automatic_pseudo_label"
    result["human_verified"] = False

    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT_FILE}"
        )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nSELECTION DISTRIBUTION:")
    print(
        result["selection_status"]
        .value_counts()
        .to_string()
    )

    print("\nSELECTION REASONS:")
    print(
        result["selection_reason"]
        .value_counts()
        .to_string()
    )

    print("\nSAVED:", OUTPUT_FILE)
    print("STAGE 7.13 COMPLETED")


if __name__ == "__main__":
    main()
