"""Run TFBS inference directly from a trained checkpoint, without the API."""

import argparse

import torch

from config import DEVICE, MODEL_DIR, SEQUENCE_LENGTH
from models.hybrid import TFBS_TriBranch
from preprocessing import clean_sequence, one_hot_encode


def load_model():
    hybrid_path = MODEL_DIR / "best_hybrid.pth"
    cnn_path = MODEL_DIR / "best_cnn.pth"

    if hybrid_path.is_file():
        checkpoint = torch.load(hybrid_path, map_location=DEVICE)
        state = checkpoint["model_state_dict"]

        conv_channels = state["conv8.weight"].shape[0]
        lstm_gates, lstm_input = state["bilstm.weight_ih_l0"].shape
        classifier_hidden, classifier_input = state["classifier.0.weight"].shape
        classifier_output = state["classifier.4.weight"].shape[0]
        has_intermediate = classifier_output != 1

        if (
            lstm_gates % 4
            or lstm_input != conv_channels * 3
            or classifier_input != conv_channels * 6 + (lstm_gates // 4) * 2
        ):
            raise RuntimeError(f"Unsupported hybrid checkpoint architecture: {hybrid_path}")

        model = TFBS_TriBranch(
            seq_len=SEQUENCE_LENGTH,
            conv_channels=conv_channels,
            lstm_hidden=lstm_gates // 4,
            classifier_hidden=classifier_hidden,
            classifier_intermediate=has_intermediate,
            classifier_output=classifier_output,
        )
        model.load_state_dict(state)
        model_name = "CNN + BiLSTM + Transformer"
        checkpoint_path = hybrid_path
    else:
        from models.cnn import TFBS_CNN

        if not cnn_path.is_file():
            raise FileNotFoundError(
                f"No trained checkpoint found. Expected {hybrid_path} or {cnn_path}."
            )
        checkpoint = torch.load(cnn_path, map_location=DEVICE)
        model = TFBS_CNN()
        model.load_state_dict(checkpoint["model_state_dict"])
        model_name = "CNN (Baseline)"
        checkpoint_path = cnn_path

    model = model.to(DEVICE)
    model.eval()
    return model, model_name, checkpoint_path


def main():
    parser = argparse.ArgumentParser(
        description="Predict TFBS binding probability using a saved model checkpoint."
    )
    parser.add_argument(
        "sequence",
        help=f"A DNA sequence of exactly {SEQUENCE_LENGTH} bases (A/C/G/T).",
    )
    args = parser.parse_args()

    sequence = clean_sequence(args.sequence)
    model, model_name, checkpoint_path = load_model()
    encoded = torch.tensor(
        one_hot_encode(sequence),
        dtype=torch.float32,
        device=DEVICE,
    ).unsqueeze(0)

    with torch.no_grad():
        probability = torch.sigmoid(model(encoded)).item()

    verdict = (
        "Potential TF Binding Site"
        if probability >= 0.5
        else "Low TF Binding Potential"
    )
    print(f"Model: {model_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {DEVICE}")
    print(f"Binding probability: {probability:.4f} ({probability * 100:.2f}%)")
    print(f"Prediction: {verdict}")


if __name__ == "__main__":
    main()