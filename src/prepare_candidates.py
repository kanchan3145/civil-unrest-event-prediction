
"""
Civil Unrest Project
Stage 2: Prepare event candidate tweets
"""

from pathlib import Path
import re
import pandas as pd


# --------------------------------------------------
# 1. Paths and settings
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_DIR / "data" / "processed" / "tweets_processed.csv"

OUTPUT_DIR = PROJECT_DIR / "data" / "annotation"
OUTPUT_FILE = OUTPUT_DIR / "balanced_candidate_pool.csv"

MAX_PER_CATEGORY = 500
RANDOM_CONTROL_COUNT = 500
RANDOM_SEED = 42


# --------------------------------------------------
# 2. Event-related keyword patterns
# --------------------------------------------------

CATEGORY_PATTERNS = {

    "protest": [
        r"\bprotested\b",
        r"\bprotesting\b",
        r"\bprotesters\b",
        r"\bprotestors\b",
        r"\b(staged|held|organized|organised|joined|led)\s+(?:a\s+)?protest\b",
        r"\bprotests?\s+(?:against|outside|at|near)\b"
    ],

    "strike": [
        r"\bwent on strike\b",
        r"\bcalled for (?:a )?strike\b",
        r"\bstrike observed\b",
        r"\bworkers.{0,60}\bon strike\b",
        r"\bstriking workers\b"
    ],

    "march": [
        r"\bmarch(?:ed|ing|ers)?\b"
    ],

    "clash": [
        r"\bclash(?:es|ed|ing)?\b",
        r"\bclashed with\b"
    ],

    "riot": [
        r"\briot(?:s|ing|ers)?\b"
    ]
}


# Compile patterns once.
COMPILED_PATTERNS = {
    category: [
        re.compile(pattern, re.IGNORECASE)
        for pattern in patterns
    ]
    for category, patterns in CATEGORY_PATTERNS.items()
}


# --------------------------------------------------
# 3. Find matching event category
# --------------------------------------------------

def find_category(text):

    for category, patterns in COMPILED_PATTERNS.items():

        for pattern in patterns:

            if pattern.search(text):
                return category

    return None


# --------------------------------------------------
# 4. Main processing
# --------------------------------------------------

def main():

    print("=" * 60)
    print("CIVIL UNREST PROJECT - STAGE 2")
    print("EVENT CANDIDATE PREPARATION")
    print("=" * 60)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {INPUT_FILE}"
        )

    print("\nLoading processed tweets...")

    df = pd.read_csv(
        INPUT_FILE,
        dtype={"post_id": "string"}
    )

    required_columns = [
        "post_id",
        "date",
        "location",
        "text",
        "language",
        "user_hash"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df = df[required_columns].copy()

    df["text"] = df["text"].fillna("").astype(str)

    df = df.drop_duplicates(subset=["post_id"])

    df = df[df["text"].str.strip() != ""].copy()

    print("Usable unique tweets:", len(df))

    # Identify a possible event category.
    print("\nSearching for event-related language...")

    df["candidate_category"] = df["text"].apply(find_category)

    keyword_rows = []
    selected_ids = set()

    for category in CATEGORY_PATTERNS:

        matches = df[
            (df["candidate_category"] == category)
            & (~df["post_id"].isin(selected_ids))
        ]

        sample_count = min(
            MAX_PER_CATEGORY,
            len(matches)
        )

        if sample_count > 0:

            selected = matches.sample(
                n=sample_count,
                random_state=RANDOM_SEED
            )

            keyword_rows.append(selected)

            selected_ids.update(
                selected["post_id"].tolist()
            )

        print(
            f"{category}: {len(matches)} matches, "
            f"{sample_count} selected"
        )

    # Combine keyword candidates.
    if keyword_rows:
        keyword_candidates = pd.concat(
            keyword_rows,
            ignore_index=True
        )
    else:
        keyword_candidates = df.iloc[0:0].copy()

    keyword_candidates["sampling_group"] = "keyword_candidate"

    # Select random controls from posts not selected above.
    remaining = df[
        ~df["post_id"].isin(selected_ids)
    ]

    control_count = min(
        RANDOM_CONTROL_COUNT,
        len(remaining)
    )

    if control_count > 0:

        random_controls = remaining.sample(
            n=control_count,
            random_state=RANDOM_SEED
        ).copy()

    else:
        random_controls = remaining.copy()

    random_controls["candidate_category"] = "none"
    random_controls["sampling_group"] = "random_control"

    # Combine both groups.
    final_df = pd.concat(
        [keyword_candidates, random_controls],
        ignore_index=True
    )

    final_df = final_df.drop_duplicates(
        subset=["post_id"]
    )

    # Save output.
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    final_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    # Display summary.
    print("\n" + "=" * 60)
    print("CANDIDATE PREPARATION COMPLETED")
    print("=" * 60)

    print("Keyword candidates:", len(keyword_candidates))
    print("Random controls:", len(random_controls))
    print("Total selected tweets:", len(final_df))

    print("\nSampling group distribution:")
    print(final_df["sampling_group"].value_counts().to_string())

    print("\nCandidate category distribution:")
    print(
        final_df["candidate_category"]
        .value_counts()
        .to_string()
    )

    print("\nSaved file:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
