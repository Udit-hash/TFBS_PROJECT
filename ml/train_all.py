"""
Train and compare every model on the real dataset with 5-fold cross-validation.

What it does (in plain words):
  * The official test set (test_clean.csv) is kept aside and never used for training.
  * The training set (train_clean.csv) is cut into 5 equal parts ("folds").
    Each model is trained 5 times: each time 4 parts are used for learning and
    the 5th part is used to check progress (validation). So every model gets
    5 independent scores, and we report the average ± spread. That shows the
    result is not a lucky split.
  * The cut-off that turns a probability into "binds / doesn't bind" is chosen
    on the validation part (instead of always 0.5), then applied to the test set.
  * Results are saved after every finished run, so if Colab disconnects you can
    rerun the same command and it continues where it stopped.

Models:
  cnn          - CNN baseline (TFBS_CNN)
  cnn_bilstm   - CNN + BiLSTM (TFBS_CNN_BiLSTM)
  tri_branch   - CNN + BiLSTM + Transformer (TFBS_TriBranch), the main model
  shape        - tri_branch + DNA shape (TFBS_TriBranchShape). Optional: needs the
                 DNAshapeR features made by prepare_shape_dataset.py (needs R).

Usage (from the ml/ folder):
  python train_all.py --quick                 # 2-minute smoke test
  python train_all.py                         # full run, 3 sequence models
  python train_all.py --with-shape            # also the DNA-shape model
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

from config import DEVICE, MODEL_DIR, PROCESSED_DATA_DIR, RANDOM_SEED, SEQUENCE_LENGTH
from models.cnn import TFBS_CNN
from models.cnn_bilstm import TFBS_CNN_BiLSTM
from models.hybrid import TFBS_TriBranch
from preprocessing import one_hot_encode
from train_comparison import _train_one, fit_shape_normalizer

SEQUENCE_MODELS = {
    "cnn": ("CNN", TFBS_CNN),
    "cnn_bilstm": ("CNN + BiLSTM", TFBS_CNN_BiLSTM),
    "tri_branch": ("CNN + BiLSTM + Transformer", TFBS_TriBranch),
}
SHAPE_KEY = "shape"
SHAPE_NAME = "CNN + BiLSTM + Transformer + DNA shape"
REPORT_METRICS = ("roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1")


# ------------------------------------------------------------------ data

def load_split(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    for column in ("sequence", "label"):
        if column not in frame.columns:
            raise ValueError(f"{path} is missing the '{column}' column.")
    sequences = frame["sequence"].astype(str).str.strip().str.upper().to_numpy()
    labels = frame["label"].astype(int).to_numpy()
    return sequences, labels


def encode_all(sequences: np.ndarray) -> torch.Tensor:
    """One-hot encode once up front (much faster than encoding every epoch)."""
    return torch.from_numpy(np.stack([one_hot_encode(s) for s in sequences]).astype(np.float32))


def load_shape_features(npz_path: Path, sequences: np.ndarray) -> tuple[np.ndarray, dict]:
    """Return DNA-shape features in the same row order as `sequences`."""
    with np.load(npz_path, allow_pickle=False) as artifact:
        saved_sequences = artifact["sequences"].astype(str)
        features = artifact["features"].astype(np.float32)
        meta = {
            "shape_feature_names": tuple(str(x) for x in artifact["feature_names"]),
            "shape_native_lengths": tuple(int(x) for x in artifact["native_lengths"]),
            "shape_predictor": str(artifact["predictor"]),
            "shape_predictor_version": str(artifact["predictor_version"]),
        }
    index = {s: i for i, s in enumerate(saved_sequences)}
    missing = [s for s in sequences if s not in index]
    if missing:
        raise ValueError(
            f"{len(missing)} sequences have no DNA-shape features in {npz_path}. "
            "Run prepare_shape_dataset.py on the same CSV files first."
        )
    return features[[index[s] for s in sequences]], meta


def make_loader(x, shape, y, idx, batch_size, shuffle):
    dataset = TensorDataset(x[idx], shape[idx], torch.from_numpy(y[idx].astype(np.float32)))
    # drop_last on training batches: a final batch of 1 would break BatchNorm layers
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle)


# --------------------------------------------------------------- scoring

@torch.no_grad()
def predict_probabilities(model, loader, use_shape: bool) -> np.ndarray:
    model.eval()
    out = []
    for sequence, shape, _ in loader:
        sequence = sequence.to(DEVICE)
        logits = model(sequence, shape.to(DEVICE)) if use_shape else model(sequence)
        out.append(torch.sigmoid(logits).squeeze(1).cpu().numpy())
    return np.concatenate(out)


def best_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Cut-off between 0.05 and 0.95 that gives the highest F1 on validation data."""
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    scores = [f1_score(labels, (probabilities >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def score(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "threshold": threshold,
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
    }


# ------------------------------------------------------------- reporting

def summarise(runs: dict) -> dict:
    summary = {}
    for key, folds in runs.items():
        done = [f for f in folds.values() if "test_tuned" in f]
        if not done:
            continue
        row = {"model": done[0]["model"], "folds_completed": len(done)}
        for metric in REPORT_METRICS:
            values = np.array([f["test_tuned"][metric] for f in done])
            row[metric] = {"mean": float(values.mean()), "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0}
        row["threshold"] = {"mean": float(np.mean([f["threshold"] for f in done]))}
        summary[key] = row
    return summary


def markdown_table(summary: dict) -> str:
    header = "| Model | Folds | AUROC | PR-AUC | Accuracy | Precision | Recall | F1 |\n|---|---|---|---|---|---|---|---|\n"
    lines = []
    for row in summary.values():
        cells = [f"{row[m]['mean']:.3f} ± {row[m]['std']:.3f}" for m in REPORT_METRICS]
        lines.append(f"| {row['model']} | {row['folds_completed']} | " + " | ".join(cells) + " |")
    return header + "\n".join(lines) + "\n"


def save_progress(out_dir: Path, runs: dict, config: dict) -> dict:
    summary = summarise(runs)
    payload = {"config": config, "runs": runs, "summary": summary}
    (out_dir / "cv_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "results_table.md").write_text(markdown_table(summary), encoding="utf-8")
    return summary


# ------------------------------------------------------------------ main

def run(args) -> dict:
    out_dir = Path(args.out)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    train_seq, train_y = load_split(data_dir / "train_clean.csv")
    test_seq, test_y = load_split(data_dir / "test_clean.csv")
    if set(train_seq) & set(test_seq):
        raise ValueError("Some sequences appear in both train and test. Fix the data first.")

    rng = np.random.default_rng(args.seed)
    if args.quick:
        train_keep = rng.choice(len(train_seq), size=min(2000, len(train_seq)), replace=False)
        test_keep = rng.choice(len(test_seq), size=min(500, len(test_seq)), replace=False)
        train_seq, train_y = train_seq[train_keep], train_y[train_keep]
        test_seq, test_y = test_seq[test_keep], test_y[test_keep]

    print(f"Device: {DEVICE} | train {len(train_y)} (bound {int(train_y.sum())}) | test {len(test_y)} (bound {int(test_y.sum())})")
    x_train, x_test = encode_all(train_seq), encode_all(test_seq)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.with_shape and SHAPE_KEY not in models:
        models.append(SHAPE_KEY)
    unknown = [m for m in models if m not in SEQUENCE_MODELS and m != SHAPE_KEY]
    if unknown:
        raise ValueError(f"Unknown model(s): {unknown}. Choose from {list(SEQUENCE_MODELS) + [SHAPE_KEY]}")

    # Sequence-only models get a tiny placeholder instead of shape data (it is ignored).
    shape_train = torch.zeros(len(train_y), 4, 1)
    shape_test = torch.zeros(len(test_y), 4, 1)
    shape_meta = {}
    if SHAPE_KEY in models:
        shape_dir = data_dir / "dna_shape"
        f_train, shape_meta = load_shape_features(shape_dir / "train_shape.npz", train_seq)
        f_test, _ = load_shape_features(shape_dir / "test_shape.npz", test_seq)
        real_shape_train, real_shape_test = torch.from_numpy(f_train), torch.from_numpy(f_test)

    config = {
        "folds": args.folds, "epochs": args.epochs, "patience": args.patience,
        "batch_size": args.batch_size, "seed": args.seed, "quick": args.quick,
        "train_size": int(len(train_y)), "test_size": int(len(test_y)),
        "sequence_length": SEQUENCE_LENGTH, "models": models,
    }
    results_path = out_dir / "cv_results.json"
    runs = {}
    if results_path.exists() and not args.restart:
        previous = json.loads(results_path.read_text(encoding="utf-8"))
        if previous.get("config", {}).get("quick") == args.quick:
            runs = previous.get("runs", {})
            print("Resuming: finished runs found in", results_path)

    folds = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    splits = list(folds.split(np.zeros(len(train_y)), train_y))
    test_idx = np.arange(len(test_y))

    for key in models:
        use_shape = key == SHAPE_KEY
        name = SHAPE_NAME if use_shape else SEQUENCE_MODELS[key][0]
        runs.setdefault(key, {})
        for fold, (fit_idx, val_idx) in enumerate(splits, start=1):
            if "test_tuned" in runs[key].get(str(fold), {}):
                print(f"{name} fold {fold}: already done, skipping.")
                continue
            seed = args.seed + fold
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            s_train = real_shape_train if use_shape else shape_train
            s_test = real_shape_test if use_shape else shape_test
            fit_loader = make_loader(x_train, s_train, train_y, fit_idx, args.batch_size, True)
            val_loader = make_loader(x_train, s_train, train_y, val_idx, args.batch_size, False)
            test_loader = make_loader(x_test, s_test, test_y, test_idx, args.batch_size, False)

            metadata = {"sequence_length": SEQUENCE_LENGTH, "fold": fold, "random_seed": seed}
            if use_shape:
                from models.shape_fusion import TFBS_TriBranchShape

                mean, std = fit_shape_normalizer(s_train[fit_idx].numpy())
                model = TFBS_TriBranchShape(seq_len=SEQUENCE_LENGTH, shape_mean=torch.from_numpy(mean), shape_std=torch.from_numpy(std))
                metadata.update(shape_meta)
                metadata.update({
                    "shape_mean": mean.tolist(), "shape_std": std.tolist(),
                    "shape_positions": int(s_train.shape[2]),
                })
            else:
                model = SEQUENCE_MODELS[key][1]()

            print(f"\n=== {name} | fold {fold}/{args.folds} ===")
            started = time.time()
            # Friend's training loop: weighted loss, AdamW, LR schedule, early stopping on validation F1.
            _train_one(
                "tri_branch_dna_shape" if use_shape else key, name, model,
                fit_loader, val_loader, test_loader, train_y[fit_idx],
                use_shape, args.epochs, args.patience, metadata,
            )
            # _train_one leaves the best weights loaded in `model` and saves a checkpoint.
            written = MODEL_DIR / ("best_shape_fusion.pth" if use_shape else f"comparison_{key}.pth")
            fold_ckpt = out_dir / "checkpoints" / f"{key}_fold{fold}.pth"
            shutil.move(str(written), fold_ckpt)  # keeps the website's model folder unchanged

            val_prob = predict_probabilities(model, val_loader, use_shape)
            test_prob = predict_probabilities(model, test_loader, use_shape)
            threshold = best_threshold(train_y[val_idx], val_prob)
            runs[key][str(fold)] = {
                "model": name,
                "fold": fold,
                "minutes": round((time.time() - started) / 60, 2),
                "threshold": threshold,
                "validation": score(train_y[val_idx], val_prob, threshold),
                "test_at_0.5": score(test_y, test_prob, 0.5),
                "test_tuned": score(test_y, test_prob, threshold),
                "checkpoint": str(fold_ckpt),
            }
            save_progress(out_dir, runs, config)
            t = runs[key][str(fold)]["test_tuned"]
            print(f"{name} fold {fold}: test AUROC {t['roc_auc']:.3f}, F1 {t['f1']:.3f} (cut-off {threshold})")

    summary = save_progress(out_dir, runs, config)
    export_final_models(out_dir, runs)
    print("\n" + markdown_table(summary))
    return summary


def export_final_models(out_dir: Path, runs: dict) -> None:
    """For each model keep the fold with the best *validation* AUROC (never pick by test score)."""
    final_dir = out_dir / "final_models"
    final_dir.mkdir(exist_ok=True)
    chosen = {}
    for key, folds in runs.items():
        done = [f for f in folds.values() if "validation" in f and Path(f["checkpoint"]).exists()]
        if not done:
            continue
        best = max(done, key=lambda f: f["validation"]["roc_auc"])
        target = final_dir / f"{key}.pth"
        shutil.copy(best["checkpoint"], target)
        chosen[key] = {"fold": best["fold"], "threshold": best["threshold"], "file": target.name}
    (final_dir / "chosen_folds.json").write_text(json.dumps(chosen, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=str(PROCESSED_DATA_DIR), help="folder with train_clean.csv and test_clean.csv")
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent / "results"), help="where results and checkpoints go")
    parser.add_argument("--models", default="cnn,cnn_bilstm,tri_branch")
    parser.add_argument("--with-shape", action="store_true", help="also train the DNA-shape model (needs DNAshapeR features)")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--quick", action="store_true", help="tiny run (2000 rows, 2 folds, 1 epoch) to check everything works")
    parser.add_argument("--restart", action="store_true", help="ignore saved progress and start again")
    args = parser.parse_args()
    if args.quick:
        args.folds, args.epochs = 2, 1
        args.out = str(Path(args.out) / "quick_test")
    run(args)


if __name__ == "__main__":
    main()
