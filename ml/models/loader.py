"""
One place that knows how to rebuild any saved model.

Checkpoints written by train_all.py contain "model_type" ("cnn", "cnn_bilstm" or "tri_branch")
and, for the exported final models, the tuned "threshold". Older checkpoints (the original
best_hybrid.pth / best_cnn.pth) have only "model_state_dict"; for those the tri-branch layout
is read off the saved weight shapes.
"""

from __future__ import annotations

from pathlib import Path

import torch

from models.cnn import TFBS_CNN
from models.cnn_bilstm import TFBS_CNN_BiLSTM
from models.hybrid import TFBS_TriBranch

ARCH_NAMES = {
    "cnn": "CNN (Baseline)",
    "cnn_bilstm": "CNN + BiLSTM",
    "tri_branch": "CNN + BiLSTM + Transformer",
}


def build_hybrid_from_state(state: dict, seq_len: int = 101) -> TFBS_TriBranch:
    """Rebuild a tri-branch model whose layer sizes are inferred from its saved weights."""
    model = TFBS_TriBranch(seq_len=seq_len)
    try:
        model.load_state_dict(state)
        return model
    except RuntimeError as current_layout_error:
        try:
            conv_channels = state["conv8.weight"].shape[0]
            for name, kernel_size in (("conv8", 8), ("conv16", 16), ("conv24", 24)):
                if tuple(state[f"{name}.weight"].shape) != (conv_channels, 4, kernel_size):
                    raise ValueError("Inconsistent multi-scale convolution dimensions.")
            lstm_gates, lstm_input = state["bilstm.weight_ih_l0"].shape
            if lstm_gates % 4 or lstm_input != conv_channels * 3:
                raise ValueError("Unsupported BiLSTM dimensions in checkpoint.")
            lstm_hidden = lstm_gates // 4
            classifier_hidden, classifier_input = state["classifier.0.weight"].shape
            if classifier_input != conv_channels * 6 + lstm_hidden * 2:
                raise ValueError("Checkpoint classifier does not match its branches.")
            classifier_output = state["classifier.4.weight"].shape[0]
            has_intermediate = classifier_output != 1
            if has_intermediate and tuple(state["classifier.7.weight"].shape) != (1, classifier_output):
                raise ValueError("Unsupported hybrid classifier layout.")
            model = TFBS_TriBranch(
                seq_len=seq_len,
                conv_channels=conv_channels,
                lstm_hidden=lstm_hidden,
                classifier_hidden=classifier_hidden,
                classifier_intermediate=has_intermediate,
                classifier_output=classifier_output,
            )
            model.load_state_dict(state)
            return model
        except (IndexError, KeyError, RuntimeError, ValueError) as compatibility_error:
            raise RuntimeError(
                f"The hybrid checkpoint is incompatible with both supported layouts: {compatibility_error}"
            ) from current_layout_error


def load_model(path: str | Path, device="cpu", seq_len: int = 101):
    """Return (model in eval mode, info dict with arch name, model_type and threshold)."""
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint["model_state_dict"]
    model_type = checkpoint.get("model_type")
    if model_type is None:  # older files: guess from the weight names
        model_type = "tri_branch" if "conv8.weight" in state else (
            "cnn_bilstm" if "bilstm.weight_ih_l0" in state else "cnn")

    if model_type == "tri_branch":
        model = build_hybrid_from_state(state, seq_len)
    elif model_type == "cnn_bilstm":
        model = TFBS_CNN_BiLSTM()
        model.load_state_dict(state)
    elif model_type == "cnn":
        model = TFBS_CNN()
        model.load_state_dict(state)
    else:
        raise ValueError(f"Unknown model_type {model_type!r} in {path}")

    info = {
        "model_type": model_type,
        "arch": ARCH_NAMES[model_type],
        "threshold": float(checkpoint.get("threshold", 0.5)),
        "validation": checkpoint.get("validation"),
        "path": str(path),
    }
    return model.to(device).eval(), info


def first_layer_convs(model) -> list[torch.nn.Conv1d]:
    """The convolutions that read the one-hot DNA directly (their filters are the learned motifs)."""
    return [m for m in model.modules() if isinstance(m, torch.nn.Conv1d) and m.in_channels == 4]
