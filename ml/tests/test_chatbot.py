"""Tests for the DNA tutor (no PyTorch or network needed)."""
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.pop("LLM_API_KEY", None)  # force offline mode

import chatbot  # noqa: E402

SEQ = "CAGGCTTGGGTACGGAGCTTAGTGGGTTCCTCACCCCAATACGGCATACCGTGACCCCCGGATCTGGGGGTCGCGCCCTTATACAATAGACGATTTTTTCA"


def _prediction():
    saliency = [0.05] * 101
    saliency[37] = 1.0
    effect = [0.0] * 101
    effect[37] = -0.45
    return {
        "binding_probability": 0.93,
        "prediction_label": "Potential TF Binding Site",
        "core_sequence": "CCAATA",
        "window_start": 35,
        "sequence": SEQ,
        "saliency": saliency,
        "ism_position_effect": effect,
        "top_mutations": [{"position": 37, "mutation": "A→C", "delta_p": -0.45}],
    }


class TutorTests(unittest.TestCase):
    def test_general_question_uses_knowledge_base(self):
        reply = chatbot.answer("What is a transcription factor?")
        self.assertEqual(reply["source"], "offline")
        self.assertIn("What is a transcription factor?", reply["references"])

    def test_position_question_uses_prediction_numbers(self):
        reply = chatbot.answer("Why is position 37 important?", prediction=_prediction())
        self.assertIn("Position 37", reply["answer"])
        self.assertIn("#1 out of 101", reply["answer"])
        self.assertIn("45 points", reply["answer"])

    def test_flat_positions_rank_low(self):
        reply = chatbot.answer("what about position 80?", prediction=_prediction())
        self.assertIn("#101 out of 101", reply["answer"])

    def test_off_topic_question_is_redirected(self):
        reply = chatbot.answer("what is the capital of France")
        self.assertIn("not sure", reply["answer"])

    def test_advanced_mode_is_technical(self):
        reply = chatbot.answer("how do base pairs bond?", mode="advanced")
        self.assertIn("Watson", reply["answer"])


if __name__ == "__main__":
    unittest.main()
