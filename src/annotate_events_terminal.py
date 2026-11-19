
from pathlib import Path
import pandas as pd

# =====================================================
# STAGE 11.4 - TERMINAL-BASED HUMAN ANNOTATION
# =====================================================

ROOT = Path(__file__).resolve().parent.parent

FILE = (
    ROOT / "data" / "extracted"
    / "human_extraction_audit_v1.csv"
)

VALID_LABELS = {
    "e": "event",
    "n": "not_event",
    "u": "uncertain"
}


def save_progress(df):
    """Save annotations safely using a temporary file."""

    temporary_file = FILE.with_suffix(".tmp")

    df.to_csv(
        temporary_file,
        index=False,
        encoding="utf-8-sig"
    )

    temporary_file.replace(FILE)


def main():

    if not FILE.exists():
        print("Annotation file not found.")
        print("Run Stage 11.4 --prepare first.")
        return

    df = pd.read_csv(
        FILE,
        dtype={"post_id": "string"},
        keep_default_na=False
    )

    if "manual_decision" not in df.columns:
        raise ValueError(
            "manual_decision column is missing."
        )

    total = len(df)

    print("=" * 65)
    print("CIVIL UNREST - HUMAN EVENT ANNOTATION")
    print("=" * 65)

    for index, row in df.iterrows():

        existing = str(
            row["manual_decision"]
        ).strip().lower()

        # Already annotated: skip automatically.
        if existing in VALID_LABELS.values():
            continue

        print("\n" + "=" * 65)
        print(f"TWEET {index + 1} OF {total}")
        print("=" * 65)

        print("\nPOST ID:")
        print(row["post_id"])

        print("\nTWEET TEXT:")
        print(row["text"])

        print("\n" + "-" * 65)

        print("Choose the correct label:")
        print("e = event")
        print("n = not_event")
        print("u = uncertain")
        print("s = skip this tweet")
        print("q = save and quit")

        while True:

            choice = input(
                "\nYour choice: "
            ).strip().lower()

            if choice == "q":

                save_progress(df)

                print("\nProgress saved.")
                print("You can resume later.")
                return

            if choice == "s":
                print("Tweet skipped.")
                break

            if choice in VALID_LABELS:

                df.at[
                    index,
                    "manual_decision"
                ] = VALID_LABELS[choice]

                save_progress(df)

                print(
                    "Saved:",
                    VALID_LABELS[choice]
                )

                break

            print(
                "Invalid choice. "
                "Enter e, n, u, s or q."
            )

    completed = (
        df["manual_decision"]
        .isin(VALID_LABELS.values())
        .sum()
    )

    print("\n" + "=" * 65)
    print("ANNOTATION SESSION FINISHED")
    print("=" * 65)

    print("Total tweets:", total)
    print("Annotated tweets:", completed)
    print("Remaining:", total - completed)

    print("\nAll annotations saved to:")
    print(FILE)


if __name__ == "__main__":
    main()
