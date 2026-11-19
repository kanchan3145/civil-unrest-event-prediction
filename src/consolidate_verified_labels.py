
from pathlib import Path
import pandas as pd

BASE = Path("data/annotation")

OLD_FILE = BASE / "incident_labels_consolidated.csv"
NEW_FILE = BASE / "incident_provisional_reviewed.csv"
OUTPUT_FILE = BASE / "incident_labels_final_v2.csv"

old = pd.read_csv(OLD_FILE, dtype={"post_id": "string"})
new = pd.read_csv(NEW_FILE, dtype={"post_id": "string"})

print("=" * 60)
print("STAGE 6.5 - FINAL LABEL CONSOLIDATION")
print("=" * 60)

# 1. Check unique post IDs.
assert old["post_id"].notna().all()
assert new["post_id"].notna().all()
assert old["post_id"].is_unique
assert new["post_id"].is_unique

# 2. Ensure all new records already exist.
assert set(new["post_id"]).issubset(set(old["post_id"]))

# 3. Verify that new decisions belong to provisional posts.
provisional_ids = set(
    old.loc[
        old["final_label_status"] == "provisional_unverified",
        "post_id"
    ]
)

assert set(new["post_id"]) == provisional_ids, (
    "Provisional IDs do not match. Consolidation stopped."
)

# 4. Validate new categories and targets.
target_map = {
    "actual_incident": 1,
    "non_event": 0,
    "planned_event": 0,
    "related_action": 0,
    "uncertain": None
}

assert new["provisional_verified_category"].isin(target_map).all()

expected_targets = new[
    "provisional_verified_category"
].map(target_map)

provided_targets = pd.to_numeric(
    new["provisional_verified_incident_target"],
    errors="coerce"
)

assert (
    expected_targets.fillna(-1).eq(
        provided_targets.fillna(-1)
    )
).all(), "Verified target mismatch."

# 5. Preserve previous values for traceability.
final = old.copy()

final["previous_final_event_category"] = final[
    "final_event_category"
]

final["previous_final_label_status"] = final[
    "final_label_status"
]

final["previous_final_incident_target"] = final[
    "final_incident_target"
]

final["final_verification_notes"] = final[
    "verification_notes"
]

# 6. Apply the 68 new decisions by post_id.
new_index = new.set_index("post_id")
mask = final["post_id"].isin(new_index.index)

matched_ids = final.loc[mask, "post_id"]

final.loc[mask, "final_event_category"] = (
    matched_ids.map(
        new_index["provisional_verified_category"]
    ).to_numpy()
)

final.loc[mask, "final_incident_target"] = (
    matched_ids.map(
        new_index["provisional_verified_incident_target"]
    ).to_numpy()
)

final.loc[mask, "final_verification_notes"] = (
    matched_ids.map(
        new_index["provisional_verification_notes"]
    ).to_numpy()
)

final.loc[mask, "final_label_status"] = "manually_verified"

uncertain_mask = (
    mask & final["final_event_category"].eq("uncertain")
)

final.loc[
    uncertain_mask, "final_label_status"
] = "verified_uncertain"

# 7. Final integrity checks.
assert len(final) == len(old) == 100
assert final["post_id"].is_unique

assert not final["final_label_status"].eq(
    "provisional_unverified"
).any()

assert final["final_event_category"].notna().all()

assert final.loc[
    final["final_event_category"].eq("uncertain"),
    "final_incident_target"
].isna().all()

assert final.loc[
    final["final_event_category"].eq("actual_incident"),
    "final_incident_target"
].eq(1).all()

assert final.loc[
    final["final_event_category"].isin(
        ["non_event", "planned_event", "related_action"]
    ),
    "final_incident_target"
].eq(0).all()

# 8. Save to a NEW file only.
if OUTPUT_FILE.exists():
    raise FileExistsError(
        f"{OUTPUT_FILE} already exists. "
        "Existing output was not overwritten."
    )

final.to_csv(OUTPUT_FILE, index=False)

print("\nFINAL RECORDS:", len(final))

print("\nFINAL LABEL STATUS:")
print(final["final_label_status"].value_counts(dropna=False))

print("\nFINAL EVENT CATEGORIES:")
print(final["final_event_category"].value_counts(dropna=False))

print("\nFINAL BINARY TARGETS:")
print(final["final_incident_target"].value_counts(dropna=False))

print("\nSAVED:", OUTPUT_FILE)
print("\nSTAGE 6.5 CONSOLIDATION COMPLETED")
