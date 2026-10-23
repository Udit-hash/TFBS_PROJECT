"""
Train and compare all three models on the real dataset with 5-fold cross-validation.

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
  pwm          - JASPAR motif scan (classic method, no training): best match of the TF's
                 known binding pattern on either strand
  kmer_lr      - k-mer counts + logistic regression (classic machine-learning baseline)
  cnn          - CNN baseline (models/cnn.py)
  cnn_bilstm   - CNN + BiLSTM (models/cnn_bilstm.py)
  tri_branch   - CNN + BiLSTM + Transformer (models/hybrid.py), the main model

Usage (from the ml/ folder):
  python train_all.py --quick      # 2-minute check that everything works
  python train_all.py              # full run
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from config import DEVICE, LEARNING_RATE, PROCESSED_DATA_DIR, RANDOM_SEED, SEQUENCE_LENGTH
from models.cnn import TFBS_CNN
from models.cnn_bilstm import TFBS_CNN_BiLSTM
from models.hybrid import TFBS_TriBranch
from preprocessing import one_hot_encode
from baselines import KmerLogReg, PWMScanner
from motifs import JASPAR_DB, find_by_tf, load_motif_db

MODELS = {
    "cnn": ("CNN", TFBS_CNN),
    "cnn_bilstm": ("CNN + BiLSTM", TFBS_CNN_BiLSTM),
    "tri_branch": ("CNN + BiLSTM + Transformer", TFBS_TriBranch),
}
# Classic (non-deep-learning) baselines: trained/evaluated on exactly the same folds
CLASSIC = {
    "pwm": "JASPAR PWM scan",
    "kmer_lr": "k-mer logistic regression",
}
REPORT_METRICS = ("roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1", "mcc")


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


def make_loader(x: torch.Tensor, y: np.ndarray, idx: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(x[idx], torch.from_numpy(y[idx].astype(np.float32)))
    # drop_last on training batches: a final batch of 1 would break BatchNorm layers
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle)


# --------------------------------------------------------------- scoring

@torch.no_grad()
def predict_probabilities(model: nn.Module, loader: DataLoader) -> np.ndarray:
    model.eval()
    out = []
    for sequence, _ in loader:
        out.append(torch.sigmoid(model(sequence.to(DEVICE))).squeeze(1).cpu().numpy())
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
        "mcc": float(matthews_corrcoef(labels, predictions)),
    }


# -------------------------------------------------------------- training

def train_one(model: nn.Module, name: str, fit_loader: DataLoader, val_loader: DataLoader,
              val_labels: np.ndarray, fit_labels: np.ndarray, epochs: int, patience: int) -> nn.Module:
    """Train with early stopping; keep the weights from the epoch with the best validation F1."""
    model = model.to(DEVICE)
    positives, negatives = int(fit_labels.sum()), int(len(fit_labels) - fit_labels.sum())
    # Weight the loss so both classes count equally even if the data is a bit unbalanced
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negatives / max(positives, 1)], device=DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6)

    best_f1, best_state, stale = -1.0, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, seen = 0.0, 0
        for sequence, labels in fit_loader:
            sequence, labels = sequence.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(sequence), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            seen += len(labels)

        val_prob = predict_probabilities(model, val_loader)
        val_f1 = f1_score(val_labels, (val_prob >= 0.5).astype(int), zero_division=0)
        val_auc = roc_auc_score(val_labels, val_prob)
        scheduler.step(val_f1)
        print(f"  {name} epoch {epoch}/{epochs}: train loss {total_loss / seen:.4f} | val F1 {val_f1:.4f} | val AUROC {val_auc:.4f}")

        if val_f1 > best_f1:
            best_f1, stale = val_f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                print(f"  {name}: stopped early after {epoch} epochs (no improvement for {patience}).")
                break

    model.load_state_dict(best_state)
    return model


# ------------------------------------------------------------- reporting

def summarise(runs: dict) -> dict:
    summary = {}
    for key, folds in runs.items():
        done = [f for f in folds.values() if "test_tuned" in f]
        if not done:
            continue
        row = {"model": done[0]["model"], "folds_completed": len(done)}
        for metric in REPORT_METRICS:
            values = np.array([f["test_tuned"].get(metric, np.nan) for f in done])
            row[metric] = {"mean": float(values.mean()), "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0}
        row["threshold"] = {"mean": float(np.mean([f["threshold"] for f in done]))}
        summary[key] = row
    return summary


def markdown_table(summary: dict) -> str:
    header = ("| Model | Folds | AUROC | PR-AUC | Accuracy | Precision | Recall | F1 | MCC |\n"
              "|---|---|---|---|---|---|---|---|---|\n")
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


def export_final_models(out_dir: Path, runs: dict) -> None:
    """For each model keep the fold with the best *validation* AUROC (never pick by test score)."""
    final_dir = out_dir / "final_models"
    final_dir.mkdir(exist_ok=True)
    chosen = {}
    for key, folds in runs.items():
        done = [f for f in folds.values() if "validation" in f and f.get("checkpoint") and Path(f["checkpoint"]).exists()]
        if not done:
            continue
        best = max(done, key=lambda f: f["validation"]["roc_auc"])
        target = final_dir / f"{key}.pth"
        # store the tuned cut-off inside the model file so the website can use it
        ckpt = torch.load(best["checkpoint"], map_location="cpu")
        ckpt["threshold"] = best["threshold"]
        ckpt["validation"] = best["validation"]
        torch.save(ckpt, target)
        chosen[key] = {"fold": best["fold"], "threshold": best["threshold"], "file": target.name}
    (final_dir / "chosen_folds.json").write_text(json.dumps(chosen, indent=2), encoding="utf-8")


def plot_roc_curves(out_dir: Path, runs: dict, test_y: np.ndarray) -> None:
    """One ROC curve per model (averaged over folds), saved as roc_curves.png for the paper."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = np.linspace(0, 1, 201)
    fig, ax = plt.subplots(figsize=(5, 5))
    for key, folds in runs.items():
        curves, aucs = [], []
        for f in folds.values():
            path = out_dir / "probs" / f"{key}_fold{f['fold']}.npy"
            if "test_tuned" in f and path.exists():
                fpr, tpr, _ = roc_curve(test_y, np.load(path))
                curves.append(np.interp(grid, fpr, tpr))
                aucs.append(f["test_tuned"]["roc_auc"])
        if curves:
            label = next(iter(folds.values()))["model"]
            ax.plot(grid, np.mean(curves, axis=0), lw=2, label=f"{label} (AUROC {np.mean(aucs):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC on the held-out test set (mean of 5 folds)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curves.png", dpi=200)
    plt.close(fig)


def classic_model(key: str, args):
    """Build a classic baseline, or return None (with a reason) if it can't run."""
    if key == "kmer_lr":
        return KmerLogReg(k=args.kmer), None
    db = load_motif_db(JASPAR_DB)
    if not db:
        return None, f"{JASPAR_DB} not found (run: python -c 'import motifs; motifs.fetch_jaspar_core()')"
    hits = find_by_tf(db, args.tf)
    if not hits:
        return None, f"no JASPAR matrix named {args.tf}"
    return PWMScanner(hits[0]["pfm"], hits[0]["id"]), None


# ------------------------------------------------------------------ main

def run_classic(key, args, splits, train_seq, train_y, test_seq, test_y, x_train, x_test, runs, out_dir, config):
    """Same folds, same cut-off rule and same test set as the deep models."""
    name = CLASSIC[key]
    runs.setdefault(key, {})
    for fold, (fit_idx, val_idx) in enumerate(splits, start=1):
        if "test_tuned" in runs[key].get(str(fold), {}):
            print(f"{name} fold {fold}: already done, skipping.")
            continue
        model, problem = classic_model(key, args)
        if model is None:
            print(f"Skipping {name}: {problem}")
            runs.pop(key, None)
            return
        started = time.time()
        if key == "pwm":  # scores one-hot input, nothing to fit
            val_prob = model.predict_proba(x_train[val_idx].numpy())
            test_prob = model.predict_proba(x_test.numpy())
        else:
            model.fit(list(train_seq[fit_idx]), train_y[fit_idx])
            val_prob = model.predict_proba(list(train_seq[val_idx]))
            test_prob = model.predict_proba(list(test_seq))
        np.save(out_dir / "probs" / f"{key}_fold{fold}.npy", test_prob)
        threshold = best_threshold(train_y[val_idx], val_prob)
        runs[key][str(fold)] = {
            "model": name + (f" ({model.motif_id})" if key == "pwm" else ""),
            "fold": fold,
            "minutes": round((time.time() - started) / 60, 2),
            "threshold": threshold,
            "validation": score(train_y[val_idx], val_prob, threshold),
            "test_at_0.5": score(test_y, test_prob, 0.5),
            "test_tuned": score(test_y, test_prob, threshold),
        }
        save_progress(out_dir, runs, config)
        t = runs[key][str(fold)]["test_tuned"]
        print(f"{name} fold {fold}: test AUROC {t['roc_auc']:.3f}, F1 {t['f1']:.3f} (cut-off {threshold})")


def run(args) -> dict:
    out_dir = Path(args.out)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "probs").mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    train_seq, train_y = load_split(data_dir / "train_clean.csv")
    test_seq, test_y = load_split(data_dir / "test_clean.csv")
    if set(train_seq) & set(test_seq):
        raise ValueError("Some sequences appear in both train and test. Fix the data first.")

    if args.quick:
        rng = np.random.default_rng(args.seed)
        keep_train = rng.choice(len(train_seq), size=min(2000, len(train_seq)), replace=False)
        keep_test = rng.choice(len(test_seq), size=min(500, len(test_seq)), replace=False)
        train_seq, train_y = train_seq[keep_train], train_y[keep_train]
        test_seq, test_y = test_seq[keep_test], test_y[keep_test]

    print(f"Device: {DEVICE} | train {len(train_y)} (bound {int(train_y.sum())}) | test {len(test_y)} (bound {int(test_y.sum())})")
    x_train, x_test = encode_all(train_seq), encode_all(test_seq)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in models if m not in MODELS and m not in CLASSIC]
    if unknown:
        raise ValueError(f"Unknown model(s): {unknown}. Choose from {list(CLASSIC) + list(MODELS)}")

    config = {
        "folds": args.folds, "epochs": args.epochs, "patience": args.patience,
        "batch_size": args.batch_size, "seed": args.seed, "quick": args.quick,
        "train_size": int(len(train_y)), "test_size": int(len(test_y)),
        "sequence_length": SEQUENCE_LENGTH, "models": models, "tf": args.tf,
    }
    results_path = out_dir / "cv_results.json"
    runs = {}
    if results_path.exists() and not args.restart:
        previous = json.loads(results_path.read_text(encoding="utf-8"))
        if previous.get("config", {}).get("quick") == args.quick:
            runs = previous.get("runs", {})
            print("Resuming: finished runs found in", results_path)

    splits = list(StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
                  .split(np.zeros(len(train_y)), train_y))
    test_loader = make_loader(x_test, test_y, np.arange(len(test_y)), args.batch_size, False)
    np.save(out_dir / "probs" / "test_labels.npy", test_y)

    for key in models:
        if key in CLASSIC:
            run_classic(key, args, splits, train_seq, train_y, test_seq, test_y, x_train, x_test, runs, out_dir, config)
            continue
        name, model_class = MODELS[key]
        runs.setdefault(key, {})
        for fold, (fit_idx, val_idx) in enumerate(splits, start=1):
            if "test_tuned" in runs[key].get(str(fold), {}):
                print(f"{name} fold {fold}: already done, skipping.")
                continue
            seed = args.seed + fold
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            fit_loader = make_loader(x_train, train_y, fit_idx, args.batch_size, True)
            val_loader = make_loader(x_train, train_y, val_idx, args.batch_size, False)

            print(f"\n=== {name} | fold {fold}/{args.folds} ===")
            started = time.time()
            model = train_one(model_class(), name, fit_loader, val_loader,
                              train_y[val_idx], train_y[fit_idx], args.epochs, args.patience)

            checkpoint = out_dir / "checkpoints" / f"{key}_fold{fold}.pth"
            torch.save({"model_type": key, "model_state_dict": model.state_dict(),
                        "sequence_length": SEQUENCE_LENGTH, "fold": fold, "random_seed": seed}, checkpoint)

            val_prob = predict_probabilities(model, val_loader)
            test_prob = predict_probabilities(model, test_loader)
            np.save(out_dir / "probs" / f"{key}_fold{fold}.npy", test_prob)
            threshold = best_threshold(train_y[val_idx], val_prob)
            runs[key][str(fold)] = {
                "model": name,
                "fold": fold,
                "minutes": round((time.time() - started) / 60, 2),
                "threshold": threshold,
                "validation": score(train_y[val_idx], val_prob, threshold),
                "test_at_0.5": score(test_y, test_prob, 0.5),
                "test_tuned": score(test_y, test_prob, threshold),
                "checkpoint": str(checkpoint),
            }
            save_progress(out_dir, runs, config)
            t = runs[key][str(fold)]["test_tuned"]
            print(f"{name} fold {fold}: test AUROC {t['roc_auc']:.3f}, F1 {t['f1']:.3f} (cut-off {threshold})")

    summary = save_progress(out_dir, runs, config)
    export_final_models(out_dir, runs)
    plot_roc_curves(out_dir, runs, test_y)
    print("\n" + markdown_table(summary))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=str(PROCESSED_DATA_DIR), help="folder with train_clean.csv and test_clean.csv")
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent / "results"), help="where results and checkpoints go")
    parser.add_argument("--models", default="pwm,kmer_lr,cnn,cnn_bilstm,tri_branch")
    parser.add_argument("--tf", default="FOSL2", help="transcription factor name for the JASPAR PWM baseline")
    parser.add_argument("--kmer", type=int, default=5, help="k for the k-mer baseline")
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
