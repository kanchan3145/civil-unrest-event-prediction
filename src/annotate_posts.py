
"""
STAGE 7: MANUAL ANNOTATION OF CIVIL UNREST POSTS

Labels:
e = event
n = not_event
u = uncertain
s = skip
q = quit
"""

from pathlib import Path
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_DIR / "data" / "annotation"
    / "manual_annotation.csv"
)

LABELS = {
    "e": "event",
    "n": "not_event",
    "u": "uncertain"
}


def main():

    print("=" * 65)
    print("CIVIL UNREST PROJECT - STAGE 7")
    print("MANUAL POST ANNOTATION")
    print("=" * 65)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"File not found: {INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        dtype={"post_id": "string"}
    )

    required = ["text", "manual_label", "annotation_reason"]

    for column in required:
        if column not in df.columns:
            raise ValueError(
                f"Missing column: {column}"
            )

    completed = 0

    for index, row in df.iterrows():
        if index < 66:
            continue
        existing_label = row["manual_label"]

        if pd.notna(existing_label) and str(existing_label).strip():
            continue

        print("\n" + "=" * 65)
        print(f"POST {index + 1} OF {len(df)}")
        print("=" * 65)

        print("\nPOST TEXT:\n")
        print(str(row["text"]))

        print("\nChoose the correct label:")
        print("e = event")
        print("n = not_event")
        print("u = uncertain")
        print("s = skip this post")
        print("q = save and quit")

        while True:

            choice = input("\nYour choice (e/n/u/s/q): ").strip().lower()

            if choice in ["e", "n", "u", "s", "q"]:
                break

            print("Invalid choice. Enter e, n, u, s, or q.")

        if choice == "q":
            print("\nProgress saved. You can resume later.")
            break

        if choice == "s":
            continue

        label = LABELS[choice]

        reason = input(
            "Short reason for your decision: "
        ).strip()

        df.at[index, "manual_label"] = label
        df.at[index, "annotation_reason"] = reason

        # Save immediately after each annotation.
        df.to_csv(
            INPUT_FILE,
            index=False,
            encoding="utf-8-sig"
        )

        completed += 1

        print(f"\nSaved label: {label}")

    remaining = df["manual_label"].isna() | (
        df["manual_label"].astype(str).str.strip() == ""
    )

    print("\n" + "=" * 65)
    print("ANNOTATION PROGRESS")
    print("=" * 65)

    print("New annotations in this session:", completed)
    print("Completed annotations:", len(df) - remaining.sum())
    print("Remaining posts:", remaining.sum())

    print("\nFile:", INPUT_FILE)


if __name__ == "__main__":
    main()
