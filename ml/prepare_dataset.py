import pandas as pd
from pathlib import Path

from config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    SEQUENCE_LENGTH
)


# ============================================================
# FILE PATHS
# ============================================================

TRAIN_INPUT = RAW_DATA_DIR / "train.csv"

TEST_INPUT = RAW_DATA_DIR / "test.csv"

TRAIN_OUTPUT = (
    PROCESSED_DATA_DIR /
    "train_clean.csv"
)

TEST_OUTPUT = (
    PROCESSED_DATA_DIR /
    "test_clean.csv"
)


# ============================================================
# PROCESS FUNCTION
# ============================================================

def process_file(input_path, output_path):

    print("\n========================================")
    print(f"Processing: {input_path.name}")
    print("========================================")

    # ----------------------------------------
    # Read CSV
    # ----------------------------------------

    df = pd.read_csv(input_path)

    print("\nOriginal shape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    # ----------------------------------------
    # Check required columns
    # ----------------------------------------

    required_columns = [
        "chr_id",
        "sequence",
        "label"
    ]

    for column in required_columns:

        if column not in df.columns:

            raise ValueError(
                f"Missing required column: {column}"
            )

    # ----------------------------------------
    # Keep only required columns
    # ----------------------------------------

    df = df[
        [
            "chr_id",
            "sequence",
            "label"
        ]
    ]

    # ----------------------------------------
    # Remove missing values
    # ----------------------------------------

    before = len(df)

    df = df.dropna(
        subset=[
            "sequence",
            "label"
        ]
    )

    removed = before - len(df)

    print(
        f"\nRemoved missing rows: {removed}"
    )

    # ----------------------------------------
    # Clean sequence
    # ----------------------------------------

    df["sequence"] = (
        df["sequence"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # ----------------------------------------
    # Check sequence length
    # ----------------------------------------

    sequence_lengths = (
        df["sequence"]
        .str.len()
    )

    print("\nSequence length distribution:")

    print(
        sequence_lengths
        .value_counts()
        .sort_index()
    )

    invalid_length = (
        sequence_lengths
        != SEQUENCE_LENGTH
    )

    print(
        "\nInvalid-length sequences:",
        invalid_length.sum()
    )

    # Remove incorrect lengths
    df = df[
        ~invalid_length
    ].copy()

    # ----------------------------------------
    # Check DNA characters
    # ----------------------------------------

    valid_bases = set("ACGT")

    def valid_sequence(sequence):

        return all(
            base in valid_bases
            for base in sequence
        )

    valid_mask = (
        df["sequence"]
        .apply(valid_sequence)
    )

    print(
        "Invalid DNA sequences:",
        (~valid_mask).sum()
    )

    df = df[
        valid_mask
    ].copy()

    # ----------------------------------------
    # Convert labels
    # ----------------------------------------

    df["label"] = pd.to_numeric(
        df["label"],
        errors="coerce"
    )

    # Remove invalid labels
    df = df.dropna(
        subset=["label"]
    )

    # Convert to integer
    df["label"] = (
        df["label"]
        .astype(int)
    )

    # ----------------------------------------
    # Make sure labels are 0 / 1
    # ----------------------------------------

    valid_labels = df["label"].isin(
        [0, 1]
    )

    print(
        "Invalid labels:",
        (~valid_labels).sum()
    )

    df = df[
        valid_labels
    ].copy()

    # ----------------------------------------
    # Remove duplicate sequences
    # ----------------------------------------

    before = len(df)

    df = df.drop_duplicates(
        subset=["sequence"]
    )

    duplicates_removed = (
        before - len(df)
    )

    print(
        "Duplicate sequences removed:",
        duplicates_removed
    )

    # ----------------------------------------
    # Reset index
    # ----------------------------------------

    df = df.reset_index(
        drop=True
    )

    # ----------------------------------------
    # Final information
    # ----------------------------------------

    print("\nFinal shape:")
    print(df.shape)

    print("\nClass distribution:")

    print(
        df["label"]
        .value_counts()
        .sort_index()
    )

    print("\nFirst 5 rows:")

    print(
        df.head()
    )

    # ----------------------------------------
    # Save
    # ----------------------------------------

    df.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nSaved to:\n{output_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("========================================")
    print("TFBS DATASET PREPARATION")
    print("========================================")

    # Check files exist

    if not TRAIN_INPUT.exists():

        raise FileNotFoundError(
            f"Training file not found:\n"
            f"{TRAIN_INPUT}"
        )

    if not TEST_INPUT.exists():

        raise FileNotFoundError(
            f"Test file not found:\n"
            f"{TEST_INPUT}"
        )

    # Process training data

    process_file(
        TRAIN_INPUT,
        TRAIN_OUTPUT
    )

    # Process test data

    process_file(
        TEST_INPUT,
        TEST_OUTPUT
    )

    print("\n")
    print("========================================")
    print("DATA PREPARATION COMPLETE")
    print("========================================")


if __name__ == "__main__":
    main()