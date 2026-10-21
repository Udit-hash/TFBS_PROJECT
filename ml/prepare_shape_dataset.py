"""Generate native DNAshapeR features for the existing train/test CSV splits."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from config import PROCESSED_DATA_DIR, SEQUENCE_LENGTH
from dna_shape import DNAshapeRPredictor, FEATURE_NAMES, PREDICTOR_NAME
from preprocessing import clean_sequence


REQUIRED_COLUMNS = ("sequence", "label")
SHAPE_DATA_DIR = PROCESSED_DATA_DIR / "dna_shape"


def _read_split(path: Path) -> tuple[list[str], np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Processed split was not found: {path}")
    frame = pd.read_csv(path)
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"Processed split contains no rows: {path}")

    sequences = [clean_sequence(value) for value in frame["sequence"]]
    if len(set(sequences)) != len(sequences):
        raise ValueError(
            f"{path} contains duplicate sequences; remove duplicates before "
            "creating training and validation subsets."
        )
    labels = pd.to_numeric(frame["label"], errors="raise").to_numpy()
    if not np.isin(labels, [0, 1]).all():
        raise ValueError(f"{path} labels must contain only 0 and 1.")
    return sequences, labels.astype(np.int64, copy=False)


def _predict_split(
    sequences: list[str],
    predictor: DNAshapeRPredictor,
    batch_size: int,
) -> tuple[np.ndarray, str, tuple[int, ...]]:
    if batch_size < 1:
        raise ValueError("batch_size must be greater than zero.")
    batches = []
    version = predictor.version
    native_lengths = None
    for start in range(0, len(sequences), batch_size):
        result = predictor.predict_batch(sequences[start : start + batch_size])
        if result.feature_names != FEATURE_NAMES or result.predictor != PREDICTOR_NAME:
            raise ValueError("DNA-shape predictor returned unexpected metadata.")
        if result.predictor_version != version:
            raise ValueError("DNAshapeR version changed while preparing a split.")
        if native_lengths is None:
            native_lengths = result.native_lengths
        elif native_lengths != result.native_lengths:
            raise ValueError(
                "DNAshapeR returned inconsistent native lengths between batches."
            )
        batches.append(result.values)
    features = np.concatenate(batches, axis=0)
    if features.shape[:2] != (len(sequences), len(FEATURE_NAMES)):
        raise ValueError(
            "DNAshapeR feature rows do not correspond one-to-one with sequences."
        )
    if features.shape[2] == 0:
        raise ValueError("DNAshapeR returned no native positions.")
    if native_lengths is None:
        raise ValueError("No DNA-shape batches were generated.")
    return features, version, native_lengths


def _save_split(
    path: Path,
    sequences: list[str],
    labels: np.ndarray,
    features: np.ndarray,
    native_lengths: tuple[int, ...],
    predictor_version: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, suffix=".npz", delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            np.savez_compressed(
                temporary_file,
                schema_version=np.asarray(1, dtype=np.int64),
                sequences=np.asarray(sequences),
                labels=labels,
                features=features.astype(np.float32, copy=False),
                native_lengths=np.asarray(native_lengths, dtype=np.int64),
                feature_names=np.asarray(FEATURE_NAMES),
                predictor=np.asarray(PREDICTOR_NAME),
                predictor_version=np.asarray(predictor_version),
                sequence_length=np.asarray(SEQUENCE_LENGTH, dtype=np.int64),
            )
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def prepare_shape_splits(
    train_csv: Path,
    test_csv: Path,
    output_dir: Path,
    predictor: DNAshapeRPredictor,
    batch_size: int = 256,
) -> tuple[Path, Path]:
    train_sequences, train_labels = _read_split(train_csv)
    test_sequences, test_labels = _read_split(test_csv)

    overlap = set(train_sequences).intersection(test_sequences)
    if overlap:
        raise ValueError(
            f"Train and test splits share {len(overlap)} DNA sequences. "
            "Resolve this split leakage before preparing shape features."
        )

    train_features, version, train_native_lengths = _predict_split(
        train_sequences, predictor, batch_size
    )
    test_features, test_version, test_native_lengths = _predict_split(
        test_sequences, predictor, batch_size
    )
    if version != test_version:
        raise ValueError("DNAshapeR version changed between train and test.")
    if train_native_lengths != test_native_lengths:
        raise ValueError(
            "DNAshapeR returned different per-feature native position counts "
            "for train and test."
        )

    train_output = output_dir / "train_shape.npz"
    test_output = output_dir / "test_shape.npz"
    _save_split(
        train_output,
        train_sequences,
        train_labels,
        train_features,
        train_native_lengths,
        version,
    )
    _save_split(
        test_output,
        test_sequences,
        test_labels,
        test_features,
        test_native_lengths,
        version,
    )
    return train_output, test_output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create aligned DNAshapeR feature artifacts for the existing splits."
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--cache-dir",
        default=str(PROCESSED_DATA_DIR / "dna_shape_cache"),
        help="Content-addressed feature cache; use an empty string to disable.",
    )
    args = parser.parse_args()
    cache_dir = None if args.cache_dir == "" else Path(args.cache_dir)
    predictor = DNAshapeRPredictor(cache_dir=cache_dir)
    train_path, test_path = prepare_shape_splits(
        PROCESSED_DATA_DIR / "train_clean.csv",
        PROCESSED_DATA_DIR / "test_clean.csv",
        SHAPE_DATA_DIR,
        predictor,
        batch_size=args.batch_size,
    )
    print(f"Saved train features: {train_path}")
    print(f"Saved test features:  {test_path}")


if __name__ == "__main__":
    main()
