"""
Attribution (explanation) methods, shared by the website (api.py) and the evaluation (interpret.py).

Every function takes a batch of one-hot sequences x (N, 4, L) and returns a per-position
importance (N, L) for the model's binding logit:

  gradient            - plain saliency map: |d logit / d input|, max over the 4 letters
  gradient_x_input    - gradient times input (only the letter that is actually present counts)
  shiftsmooth         - gradients averaged over small shifts of the sequence (paper Section IV-C)
  integrated_gradients- gradients averaged along a path from a "neutral" sequence (0.25 per letter)
"""

from __future__ import annotations

import contextlib

import numpy as np
import torch


@contextlib.contextmanager
def _grad_mode(model):
    # cuDNN's LSTM can only run backward() in training mode; turning cuDNN off avoids that
    # while keeping dropout/batch-norm in evaluation mode.
    model.eval()
    with torch.backends.cudnn.flags(enabled=False), torch.enable_grad():
        yield


def _input_grad(model, x: torch.Tensor) -> torch.Tensor:
    x = x.detach().clone().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    model(x).sum().backward()
    return x.grad.detach()


def _to_tensor(x, device):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x.float().to(device)


def gradient(model, x, device="cpu") -> np.ndarray:
    x = _to_tensor(x, device)
    with _grad_mode(model):
        g = _input_grad(model, x)
    return g.abs().amax(dim=1).cpu().numpy()


def gradient_x_input(model, x, device="cpu") -> np.ndarray:
    x = _to_tensor(x, device)
    with _grad_mode(model):
        g = _input_grad(model, x)
    return (g * x).sum(dim=1).abs().cpu().numpy()


def shiftsmooth(model, x, device="cpu", max_shift: int = 2) -> np.ndarray:
    """Average |gradient| over cyclic shifts of -max_shift..+max_shift, re-aligned to the original."""
    x = _to_tensor(x, device)
    total = torch.zeros(x.shape[0], x.shape[2], device=device)
    shifts = range(-max_shift, max_shift + 1)
    with _grad_mode(model):
        for s in shifts:
            g = _input_grad(model, torch.roll(x, shifts=s, dims=2))
            total += torch.roll(g.abs().amax(dim=1), shifts=-s, dims=1)
    return (total / len(shifts)).cpu().numpy()


def integrated_gradients(model, x, device="cpu", steps: int = 32) -> np.ndarray:
    x = _to_tensor(x, device)
    baseline = torch.full_like(x, 0.25)
    total = torch.zeros_like(x)
    with _grad_mode(model):
        for alpha in (np.arange(steps) + 0.5) / steps:
            total += _input_grad(model, baseline + float(alpha) * (x - baseline))
    attributions = (x - baseline) * total / steps
    return attributions.sum(dim=1).abs().cpu().numpy()


def integrated_gradients_shuffled(model, x, device="cpu", steps: int = 16, n_refs: int = 8, seed: int = 0) -> np.ndarray:
    """Integrated Gradients averaged over several dinucleotide-shuffled copies of each sequence as
    references (the usual choice in genomics, as in DeepLIFT/DeepSHAP), instead of a flat 0.25 baseline."""
    import random
    from data_utils import dinucleotide_shuffle, one_hot_batch

    x_np = x if isinstance(x, np.ndarray) else x.detach().cpu().numpy()
    seqs = ["".join("ACGT"[i] for i in row.argmax(axis=0)) for row in x_np]
    rng = random.Random(seed)
    xt = _to_tensor(x_np, device)
    out = torch.zeros(xt.shape[0], xt.shape[2], device=device)
    with _grad_mode(model):
        for _ in range(n_refs):
            ref = _to_tensor(one_hot_batch([dinucleotide_shuffle(s, rng) for s in seqs]), device)
            total = torch.zeros_like(xt)
            for alpha in (np.arange(steps) + 0.5) / steps:
                total += _input_grad(model, ref + float(alpha) * (xt - ref))
            out += ((xt - ref) * total / steps).sum(dim=1).abs()
    return (out / n_refs).cpu().numpy()


def random_importance(model, x, device="cpu", seed: int = 0) -> np.ndarray:
    """Chance level for the comparison: random numbers instead of an explanation."""
    x_np = x if isinstance(x, np.ndarray) else x.detach().cpu().numpy()
    return np.random.default_rng(seed + len(x_np)).random((x_np.shape[0], x_np.shape[2]))


METHODS = {
    "Gradient (saliency)": gradient,
    "Gradient × input": gradient_x_input,
    "ShiftSmooth": shiftsmooth,
    "Integrated Gradients (flat 0.25 reference)": integrated_gradients,
    "Integrated Gradients (shuffled references)": integrated_gradients_shuffled,
    "Random (chance level)": random_importance,
}


def normalise(v: np.ndarray) -> np.ndarray:
    lo, hi = float(v.min()), float(v.max())
    return (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)


@torch.no_grad()
def mc_dropout_interval(model, x, device="cpu", passes: int = 30, seed: int = 0):
    """Binding probability with a 95% interval from Monte-Carlo dropout (paper Section V:
    'binding probability scores and confidence intervals'). Batch-norm stays in eval mode."""
    x = _to_tensor(x, device)
    torch.manual_seed(seed)
    model.eval()
    dropouts = [m for m in model.modules() if isinstance(m, torch.nn.Dropout)]
    for m in dropouts:
        m.train()
    try:
        with torch.backends.cudnn.flags(enabled=False):
            probs = torch.stack([torch.sigmoid(model(x)).squeeze(-1) for _ in range(passes)]).cpu().numpy()
    finally:
        for m in dropouts:
            m.eval()
    lo, hi = np.percentile(probs, [2.5, 97.5], axis=0)
    return probs.mean(axis=0), lo, hi
