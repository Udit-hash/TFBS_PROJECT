import pandas as pd
import torch

from preprocessing import one_hot_encode


def main():

    # Load processed training data

    df = pd.read_csv(
        "data/processed/train_clean.csv"
    )

    # Take first DNA sequence

    sequence = df.iloc[0]["sequence"]

    label = df.iloc[0]["label"]

    print("================================")
    print("PREPROCESSING TEST")
    print("================================")

    print("\nOriginal sequence:")

    print(sequence)

    print("\nSequence length:")

    print(len(sequence))

    print("\nLabel:")

    print(label)

    # One-hot encode

    encoded = one_hot_encode(
        sequence
    )

    print("\nEncoded shape:")

    print(encoded.shape)

    print("\nExpected shape:")

    print("(4, 101)")

    # Check

    assert encoded.shape == (
        4,
        101
    )

    # Check each position has
    # exactly one 1

    column_sums = encoded.sum(
        axis=0
    )

    assert (
        column_sums == 1
    ).all()

    print(
        "\n✓ Sequence encoding successful"
    )

    print(
        "✓ Shape is (4, 101)"
    )

    print(
        "✓ Every DNA position has exactly one active channel"
    )


if __name__ == "__main__":

    main()