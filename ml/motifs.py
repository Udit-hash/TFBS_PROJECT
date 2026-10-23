"""
Motif utilities (numpy only): JASPAR download/loading, PWM scanning, motif comparison,
MEME export (for the Tomtom web tool) and sequence-logo drawing.

Matrices are stored as (length, 4) arrays in A, C, G, T order.
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
JASPAR_DB = HERE / "motifs" / "jaspar_core_vertebrates.json"
JASPAR_API = "https://jaspar.elixir.no/api/v1/matrix/"
BASES = "ACGT"


# ============================================================
# JASPAR
# ============================================================

def _get_json(url: str, retries: int = 3):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode())
        except Exception:  # noqa: BLE001
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))


def fetch_jaspar_core(out_path: Path = JASPAR_DB) -> Path:
    """Download every CORE vertebrate matrix (latest versions) from the JASPAR REST API."""
    url = f"{JASPAR_API}?collection=CORE&tax_group=vertebrates&version=latest&page_size=500"
    ids = []
    while url:
        page = _get_json(url)
        ids += [(r["matrix_id"], r["name"]) for r in page["results"]]
        url = page.get("next")
    motifs = []
    for k, (mid, name) in enumerate(ids):
        d = _get_json(f"{JASPAR_API}{mid}/")
        pfm = [[d["pfm"][b][i] for b in BASES] for i in range(len(d["pfm"]["A"]))]
        motifs.append({"id": mid, "name": name, "family": ", ".join(d.get("family") or []), "pfm": pfm})
        if (k + 1) % 100 == 0:
            print(f"  {k + 1}/{len(ids)} matrices")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"source": "JASPAR CORE vertebrates (latest)", "motifs": motifs}))
    return out_path


def load_motif_db(path: Path = JASPAR_DB) -> list[dict]:
    if not Path(path).exists():
        return []
    data = json.loads(Path(path).read_text())
    for m in data["motifs"]:
        m["pfm"] = np.asarray(m["pfm"], dtype=float)
    return data["motifs"]


def find_by_tf(db: list[dict], tf: str) -> list[dict]:
    t = tf.upper()
    exact = [m for m in db if m["name"].upper() == t]
    return exact or [m for m in db if t in m["name"].upper().split("::")]


# ============================================================
# MATRIX CONVERSIONS
# ============================================================

def to_ppm(pfm: np.ndarray, pseudocount: float = 0.8) -> np.ndarray:
    pfm = np.asarray(pfm, dtype=float)
    return (pfm + pseudocount / 4) / (pfm.sum(axis=1, keepdims=True) + pseudocount)


def to_pwm(pfm: np.ndarray, background=(0.25, 0.25, 0.25, 0.25)) -> np.ndarray:
    return np.log2(to_ppm(pfm) / np.asarray(background))


def revcomp_matrix(m: np.ndarray) -> np.ndarray:
    return np.asarray(m)[::-1, ::-1]  # reverse positions, swap A<->T and C<->G


def information_content(ppm: np.ndarray) -> np.ndarray:
    p = np.clip(ppm, 1e-9, 1)
    return 2 + (p * np.log2(p)).sum(axis=1)


def consensus(ppm: np.ndarray) -> str:
    return "".join(BASES[i] for i in np.asarray(ppm).argmax(axis=1))


# ============================================================
# SCANNING
# ============================================================

def scan_scores(onehot: np.ndarray, pwm: np.ndarray) -> np.ndarray:
    """onehot (N, 4, L) -> (N, 2, L-w+1) log-odds scores for forward and reverse strand."""
    w = pwm.shape[0]
    n, _, L = onehot.shape
    windows = np.lib.stride_tricks.sliding_window_view(onehot, w, axis=2)  # (N, 4, L-w+1, w)
    fwd = np.einsum("nclw,wc->nl", windows, pwm)
    rev = np.einsum("nclw,wc->nl", windows, revcomp_matrix(pwm))
    return np.stack([fwd, rev], axis=1)


def best_hit(onehot: np.ndarray, pwm: np.ndarray):
    """Max score per sequence, plus its start position and strand (+1 / -1)."""
    s = scan_scores(onehot, pwm)
    flat = s.reshape(s.shape[0], -1)
    idx = flat.argmax(axis=1)
    strand = np.where(idx < s.shape[2], 1, -1)
    return flat.max(axis=1), idx % s.shape[2], strand


def score_range(pwm: np.ndarray) -> tuple[float, float]:
    return float(pwm.min(axis=1).sum()), float(pwm.max(axis=1).sum())


# ============================================================
# MOTIF COMPARISON (Tomtom-style column correlation)
# ============================================================

def _pcc_columns(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    num = (a * b).sum(axis=1)
    den = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1)) + 1e-12
    return float((num / den).mean())


def compare_ppms(query: np.ndarray, target: np.ndarray, min_overlap: int = 5) -> dict:
    """Best average per-column Pearson correlation over all offsets and both strands."""
    best = {"score": -1.0, "offset": 0, "strand": "+", "overlap": 0}
    for strand, t in (("+", target), ("-", revcomp_matrix(target))):
        lq, lt = len(query), len(t)
        for off in range(-(lt - min_overlap), lq - min_overlap + 1):
            qs, ts = max(0, off), max(0, -off)
            n = min(lq - qs, lt - ts)
            if n < min_overlap:
                continue
            sc = _pcc_columns(query[qs:qs + n], t[ts:ts + n])
            # small penalty so tiny overlaps do not win over long ones
            sc *= n / max(n, min(lq, lt))
            if sc > best["score"]:
                best = {"score": sc, "offset": off, "strand": strand, "overlap": n}
    return best


def match_against_db(ppm: np.ndarray, db: list[dict], top: int = 3) -> list[dict]:
    hits = []
    for m in db:
        r = compare_ppms(ppm, to_ppm(m["pfm"]))
        hits.append({"id": m["id"], "name": m["name"], **r})
    hits.sort(key=lambda h: -h["score"])
    return hits[:top]


# ============================================================
# EXPORT / PLOTTING
# ============================================================

def write_meme(motifs: list[tuple[str, np.ndarray]], path: Path, nsites: int = 20) -> None:
    """Write PPMs in MEME minimal format (upload to https://meme-suite.org/meme/tools/tomtom)."""
    lines = ["MEME version 4", "", "ALPHABET= ACGT", "", "strands: + -", "",
             "Background letter frequencies", "A 0.25 C 0.25 G 0.25 T 0.25", ""]
    for name, ppm in motifs:
        lines += [f"MOTIF {name}", f"letter-probability matrix: alength= 4 w= {len(ppm)} nsites= {nsites} E= 0"]
        lines += [" ".join(f"{v:.4f}" for v in row) for row in ppm]
        lines.append("")
    Path(path).write_text("\n".join(lines))


LOGO_COLORS = {"A": "#10b981", "C": "#3b82f6", "G": "#f59e0b", "T": "#ef4444"}


def draw_logo(ax, ppm: np.ndarray, title: str | None = None):
    """Information-content sequence logo drawn with matplotlib text paths (no extra packages)."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import PathPatch
    from matplotlib.textpath import TextPath
    from matplotlib.transforms import Affine2D

    font = FontProperties(family="DejaVu Sans Mono", weight="bold")
    ic = information_content(ppm)
    for i, (row, h) in enumerate(zip(ppm, ic)):
        y = 0.0
        for k in np.argsort(row):
            height = row[k] * h
            if height < 0.01:
                continue
            letter = BASES[k]
            tp = TextPath((0, 0), letter, size=1, prop=font)
            bb = tp.get_extents()
            t = (Affine2D().translate(-bb.x0, -bb.y0)
                 .scale(0.95 / bb.width, height / bb.height)
                 .translate(i + 0.025, y))
            ax.add_patch(PathPatch(t.transform_path(tp), color=LOGO_COLORS[letter], lw=0))
            y += height
    ax.set_xlim(0, len(ppm))
    ax.set_ylim(0, 2)
    ax.set_ylabel("bits")
    ax.set_xticks(np.arange(len(ppm)) + 0.5)
    ax.set_xticklabels(range(1, len(ppm) + 1), fontsize=7)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if title:
        ax.set_title(title, fontsize=9, loc="left")


# ============================================================
# DATASET NAME -> TRANSCRIPTION FACTOR
# ============================================================

_LABS = ("Broad", "Haib", "Sydh", "Uta", "Uw", "Uchicago", "Hudsonalpha", "Yale", "Stanford")
_CELLS = ("Gm12878", "Gm12891", "Gm12892", "K562", "Hepg2", "H1hesc", "Helas3", "Dnd41", "A549", "Hct116",
          "Mcf7", "Sknsh", "Sknshra", "Pfsk1", "Huvec", "Nb4", "Hek293", "Ecc1", "T47d", "Imr90", "Hl60",
          "Hmec", "Hsmm", "Nhek", "Nha", "Osteobl", "Bjfibroblast", "Gm12873", "Gm12874", "Gm12875",
          "Gm15510", "Gm18505", "Gm18526", "Gm18951", "Gm19099", "Gm19193", "Panc1", "Raji", "Sknmc",
          "Hepg2insln", "Pancislets", "Bewo", "Ishikawa", "U2os", "Gm10847", "Mcf10aes", "Nhdfad", "Nhlf",
          "Nt2d1", "Shsy5y", "Hepg2", "Wi38", "Saec", "Hre", "Hcfaa", "Hcpe", "Hpaf", "Hbmec", "Fibrobl",
          "Gliobla", "Hfh8", "Hpf", "Rptec", "Werirb1", "Ha")
_MARKERS = ("sc", "V04", "Pcr", "Iggrab", "Iggmus", "Igg", "Ab", "Ucd", "Std", "Nb", "Pravastatin",
            "Insulin", "Forskolin", "Estradiol", "Etoh", "Dmso", "Ifna", "Ifng", "Serumstarved", "UniPk")


def factor_token(dataset_name: str) -> str:
    """'wgEncodeAwgTfbsHaibHepg2Fosl2V0416101UniPk' -> 'Fosl2'"""
    s = dataset_name.replace("wgEncodeAwgTfbs", "")
    for lab in _LABS:
        if s.startswith(lab):
            s = s[len(lab):]
            break
    for cell in sorted(_CELLS, key=len, reverse=True):
        if s.startswith(cell):
            s = s[len(cell):]
            break
    cut = len(s)
    for mk in _MARKERS:
        i = s.find(mk, 1)
        if 0 < i < cut:
            cut = i
    return s[:cut]


def resolve_tf(db: list[dict], dataset_name: str) -> dict | None:
    """Find the JASPAR matrix for the factor in a dataset name (None if the factor has no JASPAR profile,
    e.g. EZH2, HDAC2 or P300, which do not bind DNA directly)."""
    names = {}
    for m in db:
        names.setdefault(m["name"].upper(), m)
    base = factor_token(dataset_name).upper()
    # ENCODE prefixes: "C" (c-Fos, c-Myc), "E" (eGFP-tagged), "HA" (HA-tagged)
    for tok in (base, base[1:], base[2:]):
        hit = _match_token(tok, names)
        if hit:
            return hit
    return None


def _match_token(tok: str, names: dict) -> dict | None:
    if len(tok) < 2:
        return None
    for L in range(len(tok), 1, -1):
        cand = tok[:L]
        if cand in names:
            return names[cand]
        if L >= 4:
            pref = sorted(n for n in names if n.startswith(cand) and "::" not in n)
            if pref:
                return names[pref[0]]
    return None
