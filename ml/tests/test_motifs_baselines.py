"""Tests for motif scanning/comparison and the classical baselines (numpy + scikit-learn only)."""
import random, sys, tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_utils import one_hot_batch, dinucleotide_shuffle
from motifs import (to_ppm, to_pwm, best_hit, compare_ppms, revcomp_matrix, write_meme, consensus,
                    information_content, draw_logo)
from baselines import KmerLogReg, PWMScanner, kmer_matrix

CTCF_LIKE = "CCGCGNGGNGGCAG".replace("N", "A")


def pfm_from(consensus_str, strength=20):
    return np.array([[strength if b == c else 1 for b in "ACGT"] for c in consensus_str], float)


def test_scan_finds_planted_motif_both_strands():
    rng = random.Random(0)
    bg = "".join(rng.choice("ACGT") for _ in range(101))
    motif = "TGACGTCATT"
    rc = motif.translate(str.maketrans("ACGT", "TGCA"))[::-1]
    s1 = bg[:30] + motif + bg[40:]
    s2 = bg[:60] + rc + bg[70:]
    scores, pos, strand = best_hit(one_hot_batch([s1, s2]), to_pwm(pfm_from(motif)))
    assert pos.tolist() == [30, 60] and strand.tolist() == [1, -1]


def test_compare_identity_and_revcomp():
    a = to_ppm(pfm_from(CTCF_LIKE))
    assert compare_ppms(a, a)["score"] > 0.99
    r = compare_ppms(a, revcomp_matrix(a))
    assert r["score"] > 0.99 and r["strand"] == "-"
    shifted = to_ppm(pfm_from("TT" + CTCF_LIKE[:10]))
    assert compare_ppms(shifted, a)["offset"] == 2
    assert compare_ppms(a, to_ppm(pfm_from("ATATATATATAT")))["score"] < 0.5
    assert consensus(a) == CTCF_LIKE and information_content(a).min() > 1


def test_meme_and_logo():
    d = Path(tempfile.mkdtemp())
    a = to_ppm(pfm_from(CTCF_LIKE))
    write_meme([("filter_1", a)], d / "m.meme")
    txt = (d / "m.meme").read_text()
    assert "MOTIF filter_1" in txt and f"w= {len(a)}" in txt
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4, 1.4)); draw_logo(ax, a, "test"); fig.savefig(d / "l.png"); plt.close(fig)
    assert (d / "l.png").stat().st_size > 2000


def test_baselines_learn_planted_signal():
    rng = random.Random(1)
    motif = "GGAATTCCAG"
    pos = []
    for _ in range(300):
        s = "".join(rng.choice("ACGT") for _ in range(101)); p = rng.randint(0, 90)
        pos.append(s[:p] + motif + s[p + 10:])
    neg = [dinucleotide_shuffle(s, rng) for s in pos]
    seqs, y = pos + neg, np.array([1] * 300 + [0] * 300)
    idx = np.random.default_rng(0).permutation(600); tr, te = idx[:450], idx[450:]
    from sklearn.metrics import roc_auc_score
    m = KmerLogReg(k=4).fit([seqs[i] for i in tr], y[tr])
    assert roc_auc_score(y[te], m.predict_proba([seqs[i] for i in te])) > 0.85
    p = PWMScanner(pfm_from(motif)).predict_proba(one_hot_batch([seqs[i] for i in te]))
    assert roc_auc_score(y[te], p) > 0.95
    assert kmer_matrix(["ACGTA"], 2).shape[1] == 10  # 16 dimers -> 10 canonical


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
