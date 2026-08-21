import numpy as np

from config import (
    BASE_TO_INDEX,
    SEQUENCE_LENGTH
)


def clean_sequence(sequence: str) -> str:

    sequence = str(sequence)

    sequence = "".join(
        sequence.split()
    )

    sequence = sequence.upper()

    if not sequence:

        raise ValueError(
            "Sequence is empty."
        )

    if len(sequence) != SEQUENCE_LENGTH:

        raise ValueError(
            f"Expected sequence length "
            f"{SEQUENCE_LENGTH}, "
            f"got {len(sequence)}"
        )

    if not all(
        base in BASE_TO_INDEX
        for base in sequence
    ):

        raise ValueError(
            "Invalid DNA sequence. "
            "Only A, C, G and T are allowed."
        )

    return sequence


def one_hot_encode(
    sequence: str
) -> np.ndarray:

    """
    DNA:

    A = [1,0,0,0]
    C = [0,1,0,0]
    G = [0,0,1,0]
    T = [0,0,0,1]

    Output shape:

    (4, 101)
    """

    sequence = clean_sequence(
        sequence
    )

    encoded = np.zeros(
        (4, SEQUENCE_LENGTH),
        dtype=np.float32
    )

    for position, base in enumerate(
        sequence
    ):

        channel = BASE_TO_INDEX[
            base
        ]

        encoded[
            channel,
            position
        ] = 1.0

    return encoded