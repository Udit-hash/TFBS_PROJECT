"""
Explain what the trained model learned (paper Section IV-C) and produce the figures/tables:

1. Motifs learned by the first-layer CNN filters
   - every filter is run over the bound test sequences; the 101-bp stretches that switch it on
     strongly (> 50% of its maximum) are stacked into a position frequency matrix -> sequence logo
   - each learned motif is compared with all JASPAR CORE vertebrate motifs (built-in comparison)
     and exported as filters.meme so it can be uploaded to the official Tomtom tool for E-values
2. Which explanation method finds the real binding site best
   - in bound test sequences, the TF's known JASPAR motif marks the "true" core
   - Gradient, Gradient x input, ShiftSmooth and Integrated Gradients are scored on how well
     their highest-importance positions land on that core (per-sequence AUROC and hit rate)

Usage (from ml/, after train_all.py):
  python interpret.py --model results/final_models/tri_branch.pth --out results/interpretation
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from config import DEVICE, PROCESSED_DATA_DIR
from data_utils import one_hot_batch
from models.loader import first_layer_convs, load_model
from motifs import (JASPAR_DB, best_hit, consensus, draw_logo, find_by_tf, information_content,
                    load_motif_db, match_against_db, score_range, to_ppm, to_pwm, write_meme)
import xai


# ============================================================
# 1. FILTER MOTIFS
# ============================================================

@torch.no_grad()
def filter_activations(conv, x: torch.Tensor, batch: int = 512) -> np.ndarray:
    outs = []
    for i in range(0, len(x), batch):
        outs.append(torch.relu(conv(x[i:i + batch].to(DEVICE))).cpu().numpy())
    return np.concatenate(outs)  # (N, F, L_out)


def trim(ppm: np.ndarray, min_ic: float = 0.25) -> tuple[np.ndarray, int]:
    ic = information_content(ppm)
    keep = np.where(ic >= min_ic)[0]
    if len(keep) == 0:
        return ppm[:0], 0
    return ppm[keep[0]:keep[-1] + 1], int(keep[0])


def learned_motifs(model, x_pos: np.ndarray, x_neg: np.ndarray, db: list[dict],
                   min_sites: int = 20, act_frac: float = 0.5) -> list[dict]:
    xp, xn = torch.from_numpy(x_pos), torch.from_numpy(x_neg)
    L = x_pos.shape[2]
    motifs, counter = [], 0
    for conv in first_layer_convs(model):
        k, pad = conv.kernel_size[0], conv.padding[0]
        act_p = filter_activations(conv, xp)
        act_n = filter_activations(conv, xn)
        windows = np.lib.stride_tricks.sliding_window_view(x_pos, k, axis=2)  # (N, 4, L-k+1, k)
        for f in range(act_p.shape[1]):
            counter += 1
            a = act_p[:, f, :]
            top = a.max()
            if top <= 0:
                continue
            n_idx, j_idx = np.nonzero(a > act_frac * top)
            starts = j_idx - pad
            ok = (starts >= 0) & (starts <= L - k)
            n_idx, starts = n_idx[ok], starts[ok]
            if len(n_idx) < min_sites:
                continue
            pfm = windows[n_idx, :, starts, :].sum(axis=0).T  # (k, 4)
            ppm_full = to_ppm(pfm, pseudocount=1.0)
            ppm, offset = trim(ppm_full)
            if len(ppm) < 4:
                continue
            # influence: how much more the filter fires on bound than on unbound sequences
            influence = float(act_p[:, f, :].max(axis=1).mean() - act_n[:, f, :].max(axis=1).mean())
            entry = {
                "filter": f"filter_{counter}",
                "conv_width": k,
                "sites": int(len(n_idx)),
                "influence": influence,
                "consensus": consensus(ppm),
                "ppm": ppm.round(4).tolist(),
                "matches": match_against_db(ppm, db, top=3) if db else [],
            }
            motifs.append(entry)
    motifs.sort(key=lambda m: -m["influence"])
    return motifs


def save_logo_grid(motifs: list[dict], path: Path, n: int = 12) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    show = motifs[:n]
    if not show:
        return
    cols = 3
    rows = int(np.ceil(len(show) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 1.7 * rows), squeeze=False)
    for ax, m in zip(axes.flat, show):
        best = m["matches"][0] if m["matches"] else None
        title = f"{m['filter']} ({m['sites']} sites)"
        if best:
            title += f"  ≈ {best['name']} {best['id']} (r={best['score']:.2f})"
        draw_logo(ax, np.asarray(m["ppm"]), title)
    for ax in list(axes.flat)[len(show):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ============================================================
# 2. ATTRIBUTION BENCHMARK
# ============================================================

def attribution_benchmark(model, x_pos: np.ndarray, tf_pfm: np.ndarray, min_rel_score: float = 0.75,
                          max_seqs: int = 500, batch: int = 64):
    pwm = to_pwm(tf_pfm)
    w = len(pwm)
    scores, pos, _ = best_hit(x_pos, pwm)
    lo, hi = score_range(pwm)
    rel = (scores - lo) / (hi - lo)
    keep = np.where(rel >= min_rel_score)[0][:max_seqs]
    if len(keep) < 20:
        return None, None
    x = x_pos[keep]
    L = x.shape[2]
    core = np.zeros((len(keep), L), dtype=bool)
    for r, p in enumerate(pos[keep]):
        core[r, p:p + w] = True

    rows, maps = [], {}
    for name, fn in xai.METHODS.items():
        started = time.time()
        attr = np.concatenate([fn(model, x[i:i + batch], device=DEVICE) for i in range(0, len(x), batch)])
        seconds = (time.time() - started) / len(x)
        aucs = [roc_auc_score(core[i], attr[i]) for i in range(len(x))]
        peak = attr.argmax(axis=1)
        hit = np.mean([core[i, max(0, peak[i] - 2):peak[i] + 3].any() for i in range(len(x))])
        # share of total importance that falls on the motif core (higher = less noise elsewhere)
        share = float(np.mean([(attr[i] * core[i]).sum() / (attr[i].sum() + 1e-12) for i in range(len(x))]))
        rows.append({"method": name, "sequences": len(x), "core_auroc": float(np.mean(aucs)),
                     "core_auroc_sd": float(np.std(aucs)), "peak_on_core": float(hit),
                     "importance_on_core": share, "ms_per_sequence": 1000 * seconds})
        maps[name] = attr
    expected_share = w / L
    return pd.DataFrame(rows), {"maps": maps, "core": core, "x": x, "expected_share": expected_share}


def plot_attribution_example(details: dict, path: Path, index: int = 0) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    core = details["core"][index]
    seq = "".join("ACGT"[i] for i in details["x"][index].argmax(axis=0))
    fig, axes = plt.subplots(len(details["maps"]), 1, figsize=(12, 1.4 * len(details["maps"])), sharex=True)
    for ax, (name, attr) in zip(axes, details["maps"].items()):
        v = xai.normalise(attr[index])
        ax.bar(range(len(v)), v, color="#3b82f6", width=0.9)
        ax.axvspan(np.where(core)[0][0] - 0.5, np.where(core)[0][-1] + 0.5, color="#f43f5e", alpha=0.15)
        ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=8)
        ax.set_yticks([])
    axes[-1].set_xticks(range(len(seq)))
    axes[-1].set_xticklabels(list(seq), fontsize=5, family="monospace")
    axes[0].set_title("Importance per position (red band = known motif site)", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(Path(__file__).parent / "results" / "final_models" / "tri_branch.pth"))
    ap.add_argument("--data-dir", default=str(PROCESSED_DATA_DIR))
    ap.add_argument("--out", default=str(Path(__file__).parent / "results" / "interpretation"))
    ap.add_argument("--tf", default="FOSL2")
    ap.add_argument("--n-seqs", type=int, default=2000, help="bound test sequences used for filter motifs")
    ap.add_argument("--n-attr", type=int, default=500, help="sequences used for the attribution benchmark")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model, info = load_model(args.model, device=DEVICE)
    print(f"Loaded {info['arch']} from {args.model}")

    test = pd.read_csv(Path(args.data_dir) / "test_clean.csv")
    pos = test[test.label == 1].sequence.str.upper().tolist()[: args.n_seqs]
    neg = test[test.label == 0].sequence.str.upper().tolist()[: args.n_seqs]
    x_pos, x_neg = one_hot_batch(pos), one_hot_batch(neg)

    db = load_motif_db(JASPAR_DB)
    if not db:
        print(f"Note: {JASPAR_DB} missing, so learned motifs are not matched to JASPAR "
              "(run: python -c 'import motifs; motifs.fetch_jaspar_core()').")

    # ---- 1. learned filter motifs
    motifs = learned_motifs(model, x_pos, x_neg, db)
    (out / "filter_motifs.json").write_text(json.dumps({"model": info["arch"], "tf": args.tf, "motifs": motifs}, indent=1))
    write_meme([(m["filter"], np.asarray(m["ppm"])) for m in motifs], out / "filters.meme")
    save_logo_grid(motifs, out / "filter_logos.png")
    lines = ["| Filter | Sites | Influence | Consensus | Best JASPAR match | Similarity (r) |", "|---|---|---|---|---|---|"]
    for m in motifs[:20]:
        b = m["matches"][0] if m["matches"] else {"name": "-", "id": "", "score": float("nan")}
        lines.append(f"| {m['filter']} | {m['sites']} | {m['influence']:.2f} | {m['consensus']} | "
                     f"{b['name']} {b['id']} | {b['score']:.2f} |")
    (out / "filter_matches.md").write_text("\n".join(lines) + "\n")
    tf_hits = [m for m in motifs if m["matches"] and any(h["name"].upper() == args.tf.upper() or
               h["name"].upper() in ("FOS", "JUN", "FOSL1", "FOSL2", "JUND", "BATF", "FOS::JUN") for h in m["matches"][:1])]
    print(f"Learned motifs: {len(motifs)} filters kept, {len(tf_hits)} look like {args.tf}/AP-1.")
    print("\n".join(lines[:12]))

    # ---- 2. attribution benchmark
    tf = find_by_tf(db, args.tf) if db else []
    if not tf:
        print(f"Skipping attribution benchmark: no JASPAR matrix for {args.tf}.")
        return
    table, details = attribution_benchmark(model, x_pos, tf[0]["pfm"], max_seqs=args.n_attr)
    if table is None:
        print("Too few test sequences contain a clear motif match for the benchmark.")
        return
    table.to_csv(out / "attribution_benchmark.csv", index=False)
    md = ["| Method | Sequences | Core AUROC | Peak on motif | Importance on motif | ms / sequence |", "|---|---|---|---|---|---|"]
    for r in table.itertuples():
        md.append(f"| {r.method} | {r.sequences} | {r.core_auroc:.3f} ± {r.core_auroc_sd:.3f} | "
                  f"{r.peak_on_core:.1%} | {r.importance_on_core:.1%} | {r.ms_per_sequence:.1f} |")
    md.append(f"\nMotif {tf[0]['id']} covers {details['expected_share']:.1%} of each sequence, "
              "so 'importance on motif' above that means the method concentrates on the real site.")
    (out / "attribution_benchmark.md").write_text("\n".join(md) + "\n")
    plot_attribution_example(details, out / "attribution_example.png")
    print("\n".join(md))


if __name__ == "__main__":
    main()
