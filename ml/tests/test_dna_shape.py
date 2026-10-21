import tempfile
import unittest
from pathlib import Path
from subprocess import CompletedProcess

import numpy as np

from dna_shape import (
    DNAShapePredictionError,
    DNAshapeRPredictor,
    FEATURE_NAMES,
    R_SCRIPT,
)


class FakeDNAshapeR:
    def __init__(self):
        self.feature_calls = 0

    def __call__(self, command, **kwargs):
        if "-e" in command:
            return CompletedProcess(command, 0, stdout="1.40.0\n", stderr="")

        self.feature_calls += 1
        fasta_path = Path(command[3])
        output_dir = Path(command[4])
        expected_count = int(command[5])
        sequences = [
            line.strip()
            for line in fasta_path.read_text(encoding="ascii").splitlines()
            if line and not line.startswith(">")
        ]
        if len(sequences) != expected_count:
            return CompletedProcess(command, 1, stdout="", stderr="bad FASTA count")

        output_dir.mkdir()
        native_widths = (5, 6, 3, 4)
        for feature_index, (name, width) in enumerate(
            zip(FEATURE_NAMES, native_widths, strict=True)
        ):
            matrix = np.full(
                (expected_count, width), feature_index + 1, dtype=np.float32
            )
            if feature_index == 0:
                matrix[0, 0] = np.nan
            np.savetxt(output_dir / f"{name}.tsv", matrix, delimiter="\t")
        return CompletedProcess(command, 0, stdout="", stderr="")


class DNAshapeTests(unittest.TestCase):
    def test_valid_sequence_preserves_native_dimensions_and_missing_values(self):
        runner = FakeDNAshapeR()
        predictor = DNAshapeRPredictor(
            cache_dir=None, rscript_executable="Rscript-test", runner=runner
        )

        batch = predictor.predict_batch(["ACGT" * 25 + "A"])

        self.assertEqual(batch.values.shape, (1, 4, 6))
        self.assertEqual(batch.native_lengths, (5, 6, 3, 4))
        self.assertEqual(batch.feature_names, FEATURE_NAMES)
        self.assertTrue(np.isnan(batch.values[0, 0, 0]))
        self.assertTrue(
            np.isnan(batch.values[0, 0, 5])
        )
        np.testing.assert_array_equal(
            np.isnan(batch.values[0]).sum(axis=1), [2, 0, 3, 2]
        )

    def test_invalid_and_empty_sequences_are_rejected(self):
        predictor = DNAshapeRPredictor(cache_dir=None, runner=FakeDNAshapeR())
        with self.assertRaisesRegex(ValueError, "At least one DNA sequence"):
            predictor.predict_batch([])
        with self.assertRaisesRegex(ValueError, "Sequence is empty"):
            predictor.predict("")
        with self.assertRaisesRegex(ValueError, "Only A, C, G and T"):
            predictor.predict("N" * 101)
        with self.assertRaisesRegex(ValueError, "Expected sequence length 101"):
            predictor.predict("ACGT")

    def test_cache_preserves_order_and_avoids_recomputation(self):
        runner = FakeDNAshapeR()
        with tempfile.TemporaryDirectory() as temp:
            predictor = DNAshapeRPredictor(
                cache_dir=temp,
                rscript_executable="Rscript-test",
                runner=runner,
            )
            sequence_a = "A" * 101
            sequence_c = "C" * 101

            first = predictor.predict_batch([sequence_a, sequence_c, sequence_a])
            second = predictor.predict_batch([sequence_c, sequence_a])

            self.assertEqual(first.values.shape, (3, 4, 6))
            self.assertTrue(np.array_equal(first.values[0], first.values[2], equal_nan=True))
            self.assertTrue(np.array_equal(first.values[0], second.values[1], equal_nan=True))
            self.assertEqual(first.native_lengths, (5, 6, 3, 4))
            self.assertEqual(runner.feature_calls, 1)

    def test_predictor_errors_are_reported(self):
        def failed_runner(command, **kwargs):
            if "-e" in command:
                return CompletedProcess(command, 0, stdout="1.40.0", stderr="")
            return CompletedProcess(command, 2, stdout="", stderr="predictor failed")

        predictor = DNAshapeRPredictor(
            cache_dir=None, rscript_executable="Rscript-test", runner=failed_runner
        )
        with self.assertRaisesRegex(DNAShapePredictionError, "predictor failed"):
            predictor.predict("A" * 101)

    def test_bridge_script_is_present(self):
        self.assertTrue(R_SCRIPT.is_file())


if __name__ == "__main__":
    unittest.main()
