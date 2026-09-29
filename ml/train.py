import json
import random

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from torch import nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from config import (
    PROCESSED_DATA_DIR,
    MODEL_DIR,
    SEQUENCE_LENGTH,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    DEVICE,
    RANDOM_SEED
)

from preprocessing import one_hot_encode

from models.hybrid import TFBS_TriBranch


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_SEED)

np.random.seed(RANDOM_SEED)

torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        RANDOM_SEED
    )


# ============================================================
# DATASET CLASS
# ============================================================

class TFBS_Dataset(Dataset):

    def __init__(
        self,
        sequences,
        labels
    ):

        self.sequences = sequences

        self.labels = labels

    def __len__(self):

        return len(
            self.sequences
        )

    def __getitem__(self, index):

        sequence = self.sequences[index]

        label = self.labels[index]

        # --------------------------------------------
        # One-hot encode
        # --------------------------------------------

        encoded = one_hot_encode(
            sequence
        )

        # --------------------------------------------
        # Convert to PyTorch tensor
        # --------------------------------------------

        x = torch.tensor(
            encoded,
            dtype=torch.float32
        )

        y = torch.tensor(
            label,
            dtype=torch.float32
        )

        return x, y


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    train_path = (
        PROCESSED_DATA_DIR
        / "train_clean.csv"
    )

    test_path = (
        PROCESSED_DATA_DIR
        / "test_clean.csv"
    )

    train_df = pd.read_csv(
        train_path
    )

    test_df = pd.read_csv(
        test_path
    )

    print("\n========================================")

    print("DATASET")

    print("========================================")

    print(
        "Training samples:",
        len(train_df)
    )

    print(
        "Testing samples:",
        len(test_df)
    )

    print("\nTraining class distribution:")

    print(
        train_df["label"]
        .value_counts()
        .sort_index()
    )

    print("\nTesting class distribution:")

    print(
        test_df["label"]
        .value_counts()
        .sort_index()
    )

    return train_df, test_df


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_metrics(
    labels,
    probabilities
):

    # Convert probabilities to 0 / 1
    predictions = (
        probabilities >= 0.5
    ).astype(int)

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision = precision_score(
        labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0
    )

    auroc = roc_auc_score(
        labels,
        probabilities
    )

    return {

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "auroc": auroc
    }


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    model,
    loader,
    criterion=None
):

    model.eval()

    all_labels = []

    all_probabilities = []

    total_loss = 0.0

    total_samples = 0

    with torch.no_grad():

        for x, y in loader:

            # Move to device
            x = x.to(
                DEVICE
            )

            y = y.to(
                DEVICE
            )

            # Add output dimension
            y = y.unsqueeze(1)

            # Forward pass
            logits = model(x)

            # Probability
            probabilities = torch.sigmoid(
                logits
            )

            # Loss
            if criterion is not None:

                loss = criterion(
                    logits,
                    y
                )

                batch_size = x.size(0)

                total_loss += (
                    loss.item()
                    * batch_size
                )

                total_samples += batch_size

            # Store results
            all_labels.extend(
                y.squeeze(1)
                .cpu()
                .numpy()
            )

            all_probabilities.extend(
                probabilities
                .squeeze(1)
                .cpu()
                .numpy()
            )

    labels = np.array(
        all_labels
    )

    probabilities = np.array(
        all_probabilities
    )

    metrics = calculate_metrics(
        labels,
        probabilities
    )

    if total_samples > 0:

        loss = (
            total_loss
            / total_samples
        )

    else:

        loss = None

    return metrics, loss


# ============================================================
# PLOT TRAINING CURVES
# ============================================================

def plot_training_curves(
    train_losses,
    val_losses
):

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        train_losses,
        label="Training Loss"
    )

    plt.plot(
        val_losses,
        label="Validation Loss"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.title(
        "CNN Training and Validation Loss"
    )

    plt.legend()

    plt.grid(
        True
    )

    path = (
        MODEL_DIR
        / "training_curve.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"\nTraining curve saved to:\n{path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")

    print("========================================")

    print("TFBS CNN TRAINING")

    print("========================================")

    print(
        "\nDevice:",
        DEVICE
    )

    # ========================================================
    # LOAD DATA
    # ========================================================

    train_df, test_df = load_data()

    # ========================================================
    # TRAIN / VALIDATION SPLIT
    # ========================================================

    train_sequences = (
        train_df["sequence"]
        .astype(str)
        .values
    )

    train_labels = (
        train_df["label"]
        .astype(int)
        .values
    )

    test_sequences = (
        test_df["sequence"]
        .astype(str)
        .values
    )

    test_labels = (
        test_df["label"]
        .astype(int)
        .values
    )

    # 15% of training data becomes validation
    (
        train_seq,
        val_seq,
        train_y,
        val_y
    ) = train_test_split(

        train_sequences,

        train_labels,

        test_size=0.15,

        random_state=RANDOM_SEED,

        stratify=train_labels
    )

    print("\n========================================")

    print("DATA SPLIT")

    print("========================================")

    print(
        "Train:",
        len(train_seq)
    )

    print(
        "Validation:",
        len(val_seq)
    )

    print(
        "Test:",
        len(test_sequences)
    )

    # ========================================================
    # DATASETS
    # ========================================================

    train_dataset = TFBS_Dataset(
        train_seq,
        train_y
    )

    val_dataset = TFBS_Dataset(
        val_seq,
        val_y
    )

    test_dataset = TFBS_Dataset(
        test_sequences,
        test_labels
    )

    # ========================================================
    # DATALOADERS
    # ========================================================

    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True
    )

    val_loader = DataLoader(

        val_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False
    )

    test_loader = DataLoader(

        test_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = TFBS_TriBranch(seq_len=SEQUENCE_LENGTH).to(DEVICE)

    print("\n========================================")

    print("MODEL")

    print("========================================")

    print(model)

    # ========================================================
    # CLASS WEIGHT
    # ========================================================

    number_negative = np.sum(
        train_y == 0
    )

    number_positive = np.sum(
        train_y == 1
    )

    pos_weight_value = (
        number_negative
        / max(number_positive, 1)
    )

    pos_weight = torch.tensor(
        [pos_weight_value],
        dtype=torch.float32,
        device=DEVICE
    )

    print(
        "\nPositive class weight:",
        pos_weight_value
    )

    # ========================================================
    # LOSS
    # ========================================================

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    # ========================================================
    # ========================================================
    # OPTIMIZER (With L2 Weight Decay to Prevent Overfitting)
    # ========================================================

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4  # Constrains attention & recurrent weights
    )

    # ========================================================
    # TRAINING VARIABLES
    # ========================================================

    train_losses = []

    val_losses = []

    best_val_loss = float(
        "inf"
    )

    patience = 5

    patience_counter = 0

    best_model_path = MODEL_DIR / "best_hybrid.pth"

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    print("\n========================================")

    print("TRAINING")

    print("========================================")

    for epoch in range(
        EPOCHS
    ):

        # --------------------------------------------
        # Training
        # --------------------------------------------

        model.train()

        running_loss = 0.0

        total_samples = 0

        for x, y in train_loader:

            x = x.to(
                DEVICE
            )

            y = y.to(
                DEVICE
            )

            y = y.unsqueeze(1)

            # Clear gradients
            optimizer.zero_grad()

            # Forward
            logits = model(x)

            # Loss
            loss = criterion(
                logits,
                y
            )

            # Backpropagation
            loss.backward()

            # Update weights
            optimizer.step()

            batch_size = x.size(0)

            running_loss += (
                loss.item()
                * batch_size
            )

            total_samples += (
                batch_size
            )

        train_loss = (
            running_loss
            / total_samples
        )

        # --------------------------------------------
        # Validation
        # --------------------------------------------

        val_metrics, val_loss = evaluate_model(

            model,

            val_loader,

            criterion
        )

        train_losses.append(
            train_loss
        )

        val_losses.append(
            val_loss
        )

        # --------------------------------------------
        # Print
        # --------------------------------------------

        print(

            f"\nEpoch "
            f"{epoch + 1}/{EPOCHS}"

        )

        print(
            f"Train Loss : "
            f"{train_loss:.4f}"
        )

        print(
            f"Val Loss   : "
            f"{val_loss:.4f}"
        )

        print(
            f"Val AUROC  : "
            f"{val_metrics['auroc']:.4f}"
        )

        print(
            f"Val F1     : "
            f"{val_metrics['f1']:.4f}"
        )

        # --------------------------------------------
        # Save best model
        # --------------------------------------------

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({"model_state_dict": model.state_dict()}, best_model_path)
            print("✓ Best model saved")
        else:
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{patience})")

        if patience_counter >= patience:
            print("\nEarly stopping triggered.")
            break

    # ========================================================
    # TRAINING CURVE
    # ========================================================

    plot_training_curves(
        train_losses,
        val_losses
    )

    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    print(
        "\nLoading best model..."
    )

    checkpoint = torch.load(

        best_model_path,

        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    # ========================================================
    # FINAL TEST
    # ========================================================

    test_metrics, test_loss = evaluate_model(

        model,

        test_loader,

        criterion
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print("\n")

    print("========================================")

    print("FINAL TEST RESULTS")

    print("========================================")

    print(
        f"Test Loss  : "
        f"{test_loss:.4f}"
    )

    print(
        f"AUROC      : "
        f"{test_metrics['auroc']:.4f}"
    )

    print(
        f"Accuracy   : "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision  : "
        f"{test_metrics['precision']:.4f}"
    )

    print(
        f"Recall     : "
        f"{test_metrics['recall']:.4f}"
    )

    print(
        f"F1 Score   : "
        f"{test_metrics['f1']:.4f}"
    )

    # ========================================================
    # SAVE METRICS
    # ========================================================

    results = {
        "model": "Hybrid_CNN_BiLSTM",
        "sequence_length": SEQUENCE_LENGTH,
        "test_loss": test_loss,
        "AUROC": test_metrics["auroc"],
        "accuracy": test_metrics["accuracy"],
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "F1": test_metrics["f1"]
    }

    results_path = MODEL_DIR / "hybrid_results.json"
    with open(results_path, "w") as file:
        json.dump(results, file, indent=4)

    print(f"\nResults saved to: {results_path}")
    print(f"Best model saved to: {best_model_path}")


if __name__ == "__main__":

    main()