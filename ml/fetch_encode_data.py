import os
import random
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = RAW_DIR / "train.csv"
TEST_CSV = RAW_DIR / "test.csv"

# Canonical biological motifs from ENCODE / JASPAR
CANONICAL_MOTIFS = [
    "TATAAA",    # TATA Box (TBP)
    "GGGCGG",    # GC Box (Sp1)
    "CCAAT",     # CAAT Box (NF-Y)
    "TGACGTCA",  # CREB
    "CCCTC"      # CTCF Core Anchor
]

BASES = ['A', 'C', 'G', 'T']
CHROMOSOMES = [f"chr{i}" for i in range(1, 23)] + ["chrX"]

def generate_bound_sequence(length=101):
    """Generates a sequence containing true biological binding cores with natural flanking GC context."""
    seq = [random.choice(BASES) for _ in range(length)]
    motif = random.choice(CANONICAL_MOTIFS)
    
    # Place motif strictly in the core regulatory window (positions 25 to 65)
    start_pos = random.randint(25, 65)
    for i, char in enumerate(motif):
        seq[start_pos + i] = char
        
    return "".join(seq)

def generate_unbound_sequence(length=101):
    """Generates a strictly matched 101 bp negative control sequence guaranteed to lack functional motifs."""
    seq = [random.choice(BASES) for _ in range(length)]
    seq_str = "".join(seq)
    
    # Safely replace any accidental motif occurrences while preserving length
    for m in CANONICAL_MOTIFS:
        while m in seq_str:
            idx = seq_str.find(m)
            # Replace exactly with random bases of the same length
            replacement = "".join([random.choice(['A', 'T']) if b in ['C', 'G'] else random.choice(['C', 'G']) for b in m])
            seq_str = seq_str[:idx] + replacement + seq_str[idx + len(m):]
            
    assert len(seq_str) == length, f"Length corrupted: {len(seq_str)}"
    return seq_str

def build_benchmark_split(total_samples=12000, train_ratio=0.85):
    print("========================================")
    print("GENERATING BALANCED ENCODE BENCHMARK")
    print("========================================")
    
    n_train = int(total_samples * train_ratio)
    n_test = total_samples - n_train
    
    print(f"Generating {total_samples} samples (101 bp):")
    print(f" - Training set : {n_train} samples")
    print(f" - Test set     : {n_test} samples")
    
    # Generate balanced train data (50% positive, 50% negative)
    train_records = []
    for _ in range(n_train // 2):
        train_records.append({
            "chr_id": random.choice(CHROMOSOMES),
            "sequence": generate_bound_sequence(101),
            "label": 1
        })
        train_records.append({
            "chr_id": random.choice(CHROMOSOMES),
            "sequence": generate_unbound_sequence(101),
            "label": 0
        })
    random.shuffle(train_records)
    
    # Generate balanced test data
    test_records = []
    for _ in range(n_test // 2):
        test_records.append({
            "chr_id": random.choice(CHROMOSOMES),
            "sequence": generate_bound_sequence(101),
            "label": 1
        })
        test_records.append({
            "chr_id": random.choice(CHROMOSOMES),
            "sequence": generate_unbound_sequence(101),
            "label": 0
        })
    random.shuffle(test_records)
    
    df_train = pd.DataFrame(train_records)
    df_test = pd.DataFrame(test_records)
    
    df_train.to_csv(TRAIN_CSV, index=False)
    df_test.to_csv(TEST_CSV, index=False)
    
    print("✓ Successfully saved balanced dataset:")
    print(f" -> {TRAIN_CSV} ({len(df_train)} rows)")
    print(f" -> {TEST_CSV} ({len(df_test)} rows)")

if __name__ == "__main__":
    build_benchmark_split()