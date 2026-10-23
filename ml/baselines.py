"""
Non-deep-learning baselines for the comparison table (paper Section II / VI):

- kmer_lr : logistic regression on k-mer counts (the classic SVM/k-mer style baseline)
- pwm     : the transcription factor's own JASPAR position weight matrix, scoring each
            sequence by its best motif match on either strand (no training at all)
"""

from __future__ import annotations

from itertools import product

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from motifs import best_hit, to_pwm


def kmer_matrix(seqs: list[str], k: int = 5) -> np.ndarray:
    """Counts of every k-mer, merging each k-mer with its reverse complement."""
    comp = str.maketrans("ACGT", "TGCA")
    canon, index = {}, {}
    for kmer in ("".join(p) for p in product("ACGT", repeat=k)):
        rc = kmer.translate(comp)[::-1]
        key = min(kmer, rc)
        if key not in index:
            index[key] = len(index)
        canon[kmer] = index[key]
    X = np.zeros((len(seqs), len(index)), dtype=np.float32)
    for n, s in enumerate(seqs):
        for i in range(len(s) - k + 1):
            j = canon.get(s[i:i + k])
            if j is not None:
                X[n, j] += 1
    return X


class KmerLogReg:
    name = "kmer_lr"

    def __init__(self, k: int = 5, C: float = 0.1):
        self.k = k
        self.model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))

    def fit(self, seqs, y):
        self.model.fit(kmer_matrix(seqs, self.k), y)
        return self

    def predict_proba(self, seqs) -> np.ndarray:
        return self.model.predict_proba(kmer_matrix(seqs, self.k))[:, 1]


class PWMScanner:
    """Best log-odds match of a JASPAR PWM, squashed to 0..1 so it can be thresholded like a probability."""
    name = "pwm"

    def __init__(self, pfm: np.ndarray, motif_id: str = ""):
        self.pwm = to_pwm(pfm)
        self.motif_id = motif_id

    def fit(self, seqs, y):
        return self  # nothing to learn

    def predict_proba(self, onehot: np.ndarray) -> np.ndarray:
        scores, _, _ = best_hit(onehot, self.pwm)
        return 1 / (1 + np.exp(-scores / 2))
