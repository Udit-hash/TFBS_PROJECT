import asyncio
import unittest
from unittest.mock import patch

import numpy as np
import torch

import api
from dna_shape import (
    DNAShapeBatch,
    DNAShapePrediction,
    FEATURE_NAMES,
    PREDICTOR_NAME,
)


class FakeShapeModel(torch.nn.Module):
    def forward(self, sequence, shape):
        sequence_score = sequence[:, 0, :].sum(dim=1, keepdim=True) * 0.001
        shape_score = torch.nan_to_num(shape).mean(dim=(1, 2)).unsqueeze(1) * 0.001
        return sequence_score + shape_score


class FakeSequenceModel(torch.nn.Module):
    def forward(self, sequence):
        return sequence[:, 0, :].sum(dim=1, keepdim=True) * 0.001


class FakePredictor:
    def predict_batch(self, sequences):
        values = np.zeros((len(sequences), 4, 5), dtype=np.float32)
        return DNAShapeBatch(
            values=values,
            native_lengths=(5, 5, 5, 5),
            feature_names=FEATURE_NAMES,
            predictor=PREDICTOR_NAME,
            predictor_version="test-version",
        )


class FakeSingleShapePredictor:
    def predict(self, sequence):
        values = np.arange(20, dtype=np.float32).reshape(4, 5)
        values[0, 0] = np.nan
        return DNAShapePrediction(
            values=values,
            native_lengths=(4, 5, 3, 5),
            feature_names=FEATURE_NAMES,
            predictor=PREDICTOR_NAME,
            predictor_version="test-version",
        )


class ShapeAPIIntegrationTests(unittest.TestCase):
    def test_dna_shape_endpoint_returns_native_feature_matrix(self):
        with patch.object(api, "shape_matrix_predictor", FakeSingleShapePredictor()):
            response = asyncio.run(
                api.predict_dna_shape(api.SequenceInput(sequence="A" * 101))
            )

        self.assertEqual(response["feature_names"], list(FEATURE_NAMES))
        self.assertEqual(response["native_lengths"], [4, 5, 3, 5])
        self.assertEqual(len(response["rows"]), 5)
        self.assertIsNone(response["rows"][0]["MGW"])
        self.assertEqual(response["rows"][1]["MGW"], 1.0)
        self.assertIsNone(response["rows"][4]["MGW"])
        self.assertIsNone(response["rows"][4]["ProT"])

    def test_dna_shape_endpoint_rejects_invalid_sequence(self):
        with self.assertRaises(api.HTTPException) as error:
            asyncio.run(
                api.predict_dna_shape(api.SequenceInput(sequence="N" * 101))
            )
        self.assertEqual(error.exception.status_code, 400)

    def test_predict_endpoint_uses_shape_features_and_preserves_response(self):
        sequence = "ACGT" * 25 + "A"
        with (
            patch.object(api, "model", FakeShapeModel()),
            patch.object(api, "shape_predictor", FakePredictor()),
            patch.object(api, "SHAPE_MODEL_ACTIVE", True),
            patch.object(api, "SHAPE_POSITIONS", 5),
            patch.object(api, "SHAPE_NATIVE_LENGTHS", (5, 5, 5, 5)),
            patch.object(api, "SHAPE_NATIVE_LENGTHS", (5, 5, 5, 5)),
        ):
            response = asyncio.run(
                api.predict_tfbs(api.SequenceInput(sequence=sequence))
            )

        self.assertEqual(response["status"], "success")
        self.assertEqual(response["input_length"], 101)
        self.assertIn("binding_probability", response)
        self.assertEqual(response["dna_shape"]["feature_names"], list(FEATURE_NAMES))
        self.assertEqual(response["dna_shape"]["native_position_count"], 5)
        self.assertEqual(len(response["visualizations"]["saliency_map"]), 101)

    def test_sequence_only_prediction_keeps_legacy_response_contract(self):
        with (
            patch.object(api, "model", FakeSequenceModel()),
            patch.object(api, "shape_predictor", None),
            patch.object(api, "SHAPE_MODEL_ACTIVE", False),
        ):
            response = asyncio.run(
                api.predict_tfbs(api.SequenceInput(sequence="A" * 101))
            )

        self.assertEqual(response["status"], "success")
        self.assertEqual(response["input_length"], 101)
        self.assertIn("binding_probability", response)
        self.assertIn("prediction", response)
        self.assertIn("visualizations", response)
        self.assertNotIn("dna_shape", response)


if __name__ == "__main__":
    unittest.main()
