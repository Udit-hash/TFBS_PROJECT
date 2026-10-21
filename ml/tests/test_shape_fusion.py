import unittest
import tempfile

import numpy as np
import torch

from models.shape_fusion import TFBS_TriBranchShape


class ShapeFusionTests(unittest.TestCase):
    def test_forward_dimensions_with_native_width_and_missing_values(self):
        model = TFBS_TriBranchShape(seq_len=101).eval()
        sequence = torch.zeros(2, 4, 101)
        shape = torch.randn(2, 4, 17)
        shape[0, 0, :2] = float("nan")

        with torch.no_grad():
            logits = model(sequence, shape)
            representation = model.encode(sequence, shape)

        self.assertEqual(logits.shape, (2, 1))
        self.assertEqual(representation.shape, (2, 304))
        self.assertTrue(torch.isfinite(logits).all())

    def test_shape_input_guards_and_checkpoint_round_trip(self):
        model = TFBS_TriBranchShape(seq_len=101, shape_mean=torch.ones(4))
        sequence = torch.zeros(1, 4, 101)
        with self.assertRaisesRegex(ValueError, "length 101"):
            model(torch.zeros(1, 4, 200), torch.zeros(1, 4, 7))
        with self.assertRaisesRegex(ValueError, "native_positions"):
            model(sequence, torch.zeros(1, 4, 0))
        with self.assertRaisesRegex(ValueError, "batch sizes"):
            model(sequence, torch.zeros(2, 4, 7))
        with self.assertRaisesRegex(ValueError, "infinite"):
            model(sequence, torch.full((1, 4, 7), float("inf")))

        restored = TFBS_TriBranchShape(seq_len=101, shape_mean=torch.ones(4))
        with tempfile.TemporaryDirectory() as temp:
            checkpoint_path = f"{temp}/checkpoint.pth"
            torch.save(
                {
                    "model_type": "tri_branch_dna_shape",
                    "sequence_length": 101,
                    "shape_feature_names": ["MGW", "Roll", "ProT", "HelT"],
                    "shape_positions": 7,
                    "model_state_dict": model.state_dict(),
                },
                checkpoint_path,
            )
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
        self.assertEqual(checkpoint["model_type"], "tri_branch_dna_shape")
        restored.load_state_dict(checkpoint["model_state_dict"])
        self.assertTrue(torch.equal(restored.shape_mean, model.shape_mean))

    def test_all_missing_shape_feature_channel_remains_finite(self):
        model = TFBS_TriBranchShape(seq_len=101).eval()
        sequence = torch.zeros(1, 4, 101)
        shape = torch.full((1, 4, 9), float("nan"))
        with torch.no_grad():
            logits = model(sequence, shape)
        self.assertTrue(torch.isfinite(logits).all())


if __name__ == "__main__":
    unittest.main()
