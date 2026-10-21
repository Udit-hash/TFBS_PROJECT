import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from dna_shape import DNAShapeBatch, FEATURE_NAMES, PREDICTOR_NAME
from prepare_shape_dataset import prepare_shape_splits
from train_comparison import ShapeDataset, fit_shape_normalizer, load_shape_split
from preprocessing import one_hot_encode


class FakePredictor:
    version = "test-version"

    def predict_batch(self, sequences):
        values = np.empty((len(sequences), 4, 3), dtype=np.float32)
        for row, sequence in enumerate(sequences):
            values[row] = ord(sequence[0]) % 7
        return DNAShapeBatch(
            values=values,
            native_lengths=(3, 3, 3, 3),
            feature_names=FEATURE_NAMES,
            predictor=PREDICTOR_NAME,
            predictor_version=self.version,
        )


class ShapeDatasetTests(unittest.TestCase):
    def _write_csv(self, path: Path, sequences, labels):
        pd.DataFrame(
            {
                "chr_id": [f"chr{index + 1}" for index in range(len(sequences))],
                "sequence": sequences,
                "label": labels,
            }
        ).to_csv(path, index=False)

    def test_feature_rows_and_labels_remain_aligned(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train_sequences = ["A" * 101, "C" * 101]
            test_sequences = ["G" * 101, "T" * 101]
            self._write_csv(root / "train.csv", train_sequences, [1, 0])
            self._write_csv(root / "test.csv", test_sequences, [0, 1])

            train_path, test_path = prepare_shape_splits(
                root / "train.csv",
                root / "test.csv",
                root / "features",
                FakePredictor(),
                batch_size=1,
            )
            train = load_shape_split(train_path)
            test = load_shape_split(test_path)

            self.assertEqual(train["sequences"].tolist(), train_sequences)
            self.assertEqual(train["labels"].tolist(), [1, 0])
            self.assertEqual(
                train["features"][0, 0, 0], ord("A") % 7
            )
            self.assertEqual(
                train["features"][1, 0, 0], ord("C") % 7
            )
            self.assertEqual(test["sequences"].tolist(), test_sequences)
            self.assertEqual(test["labels"].tolist(), [0, 1])
            self.assertEqual(train["features"].shape, (2, 4, 3))

    def test_training_encoding_matches_inference_encoder(self):
        sequences = np.asarray(["ACGT" * 25 + "A"])
        shapes = np.zeros((1, 4, 3), dtype=np.float32)
        dataset = ShapeDataset(sequences, shapes, np.asarray([1]))
        encoded, _, label = dataset[0]
        np.testing.assert_array_equal(encoded.numpy(), one_hot_encode(sequences[0]))
        self.assertEqual(label.item(), 1.0)

    def test_train_test_sequence_overlap_fails_before_prediction(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sequence = "A" * 101
            self._write_csv(root / "train.csv", [sequence], [1])
            self._write_csv(root / "test.csv", [sequence], [0])

            with self.assertRaisesRegex(ValueError, "share 1 DNA sequences"):
                prepare_shape_splits(
                    root / "train.csv",
                    root / "test.csv",
                    root / "features",
                    FakePredictor(),
                )

    def test_normalizer_uses_only_finite_training_values(self):
        features = np.array(
            [
                [[1, 3], [10, np.nan], [2, 2], [4, 8]],
                [[5, 7], [14, 18], [6, 6], [12, 16]],
            ],
            dtype=np.float32,
        )
        means, stds = fit_shape_normalizer(features)
        np.testing.assert_allclose(means, [4, 14, 4, 10])
        self.assertTrue(np.all(stds > 0))


if __name__ == "__main__":
    unittest.main()
