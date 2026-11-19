
from pathlib import Path
import pandas as pd

SEED = 42
BASE = Path("data/annotation")

POOL_FILE = BASE / "balanced_candidate_pool.csv"
ANNOTATED_FILE = BASE / "incident_labels_final_v2.csv"
OUTPUT_FILE = BASE / "additional_annotation_sample.csv"

SAMPLE_COUNTS = {
    "keyword_candidate": 240,
    "random_control": 60
}

print("=" * 60)
print("STAGE 7.1 - ADDITIONAL ANNOTATION SAMPLING")
print("=" * 60)

# 1. Load data while preserving large tweet IDs.
pool = pd.read_csv(
    POOL_FILE,
    dtype={"post_id": "string"}
)

annotated = pd.read_csv(
    ANNOTATED_FILE,
    dtype={"post_id": "string"}
)

# 2. Validate input records.
assert pool["post_id"].notna().all()
assert pool["post_id"].is_unique
assert annotated["post_id"].is_unique

# 3. Exclude previously annotated tweets.
available = pool[
    ~pool["post_id"].isin(annotated["post_id"])
].copy()

available = available[
    available["text"].notna()
    & available["text"].astype(str).str.strip().ne("")
]

print("\nAvailable unannotated posts:", len(available))

# 4. Sample separately from each group.
samples = []

for group, count in SAMPLE_COUNTS.items():

    candidates = available[
        available["sampling_group"] == group
    ]

    if len(candidates) < count:
        raise ValueError(
            f"Not enough candidates in {group}. "
            f"Required={count}, Available={len(candidates)}"
        )

    selected = candidates.sample(
        n=count,
        random_state=SEED
    )

    samples.append(selected)

# 5. Combine and shuffle.
sample = pd.concat(samples, ignore_index=True)

sample = sample.sample(
    frac=1,
    random_state=SEED
).reset_index(drop=True)

# 6. Add annotation fields.
sample["manual_event_category"] = ""
sample["manual_incident_target"] = pd.NA
sample["annotation_notes"] = ""
sample["annotation_status"] = "pending"

# 7. Final validation.
assert len(sample) == 300
assert sample["post_id"].is_unique

assert not sample["post_id"].isin(
    annotated["post_id"]
).any()

# 8. Save without overwriting an existing sample.
if OUTPUT_FILE.exists():
    raise FileExistsError(
        f"{OUTPUT_FILE} already exists. "
        "Existing annotation work was not overwritten."
    )

sample.to_csv(OUTPUT_FILE, index=False)

print("\nSAMPLE DISTRIBUTION:")
print(
    sample["sampling_group"]
    .value_counts()
    .to_string()
)

print("\nTOTAL SELECTED:", len(sample))
print("DUPLICATE POST IDS:", sample["post_id"].duplicated().sum())

print("\nSAVED FILE:", OUTPUT_FILE)
print("\nSTAGE 7.1 COMPLETED")
