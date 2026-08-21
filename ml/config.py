from pathlib import Path
import torch


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODEL_DIR = BASE_DIR / "models"


RAW_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PROCESSED_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DATASET
# ============================================================

SEQUENCE_LENGTH = 101


# ============================================================
# DNA ENCODING
# ============================================================

DNA_ALPHABET = "ACGT"

BASE_TO_INDEX = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3
}


# ============================================================
# TRAINING
# ============================================================

BATCH_SIZE = 64

EPOCHS = 15

LEARNING_RATE = 0.001

RANDOM_SEED = 42


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "Using device:",
    DEVICE
)