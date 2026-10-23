"""Predict binding for one sequence or a FASTA file from the command line (no website needed).

  python predict.py ACGT...(101 bp)
  python predict.py --fasta my_sequences.fa
"""

import argparse

import torch

from config import DEVICE, MODEL_DIR, SEQUENCE_LENGTH
from data_utils import one_hot_batch, read_sequences
from models.loader import load_model
from preprocessing import clean_sequence


def load_best():
    for path in (MODEL_DIR / "best_hybrid.pth", MODEL_DIR / "best_cnn.pth"):
        if path.is_file():
            try:
                return load_model(path, device=DEVICE, seq_len=SEQUENCE_LENGTH)
            except RuntimeError as exc:
                print(f"Skipping {path.name}: {exc}")
    raise FileNotFoundError(f"No usable checkpoint in {MODEL_DIR}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sequence", nargs="?", help=f"A DNA sequence of exactly {SEQUENCE_LENGTH} bases (A/C/G/T).")
    parser.add_argument("--fasta", help="FASTA file with one or more 101 bp sequences")
    args = parser.parse_args()
    if not args.sequence and not args.fasta:
        parser.error("give a sequence or --fasta")

    if args.fasta:
        ids, seqs, _ = read_sequences(args.fasta, seq_len=SEQUENCE_LENGTH, make_negatives_for_fasta=False)
    else:
        ids, seqs = ["input"], [clean_sequence(args.sequence)]

    model, info = load_best()
    print(f"Model: {info['arch']} | checkpoint: {info['path']} | cut-off {info['threshold']:.2f} | device: {DEVICE}")
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.from_numpy(one_hot_batch(seqs)).to(DEVICE))).squeeze(1).cpu().numpy()
    for i, p in zip(ids, probs):
        verdict = "Potential TF Binding Site" if p >= info["threshold"] else "Low TF Binding Potential"
        print(f"{i}\t{p:.4f}\t{verdict}")


if __name__ == "__main__":
    main()
