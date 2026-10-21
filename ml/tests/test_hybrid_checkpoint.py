import unittest

import torch

from api import _load_hybrid_checkpoint
from models.hybrid import TFBS_TriBranch


class HybridCheckpointTests(unittest.TestCase):
    def test_loads_checkpoint_with_its_original_hybrid_dimensions(self):
        source = TFBS_TriBranch(
            conv_channels=16,
            lstm_hidden=32,
            classifier_hidden=64,
            classifier_intermediate=False,
        )

        model = _load_hybrid_checkpoint(
            {"model_state_dict": source.state_dict()}
        ).eval()
        with torch.no_grad():
            logits = model(torch.zeros(1, 4, 101))

        self.assertEqual(model.conv8.out_channels, 16)
        self.assertEqual(model.bilstm.hidden_size, 32)
        self.assertEqual(tuple(logits.shape), (1, 1))
        self.assertTrue(torch.isfinite(logits).all())


if __name__ == "__main__":
    unittest.main()
