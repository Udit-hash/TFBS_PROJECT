from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Literal, Optional

from pydantic import BaseModel, Field
import torch
import numpy as np

import os
from pathlib import Path

import chatbot  # loads ml/.env (if present) before config reads TORCH_NUM_THREADS
import xai
from config import DEVICE, SEQUENCE_LENGTH, MODEL_DIR
from models.loader import build_hybrid_from_state, load_model
from motifs import JASPAR_DB, load_motif_db, revcomp_matrix, scan_scores, score_range, to_ppm, to_pwm
from preprocessing import one_hot_encode

app = FastAPI(
    title="TFBS Multi-Architecture & Interpretability API",
    description="CNN + BiLSTM + Transformer TFBS Prediction & Multi-Modal XAI Service",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SequenceInput(BaseModel):
    sequence: str
    model: Optional[str] = None  # key from GET /models; default = the proposed tri-branch v2


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatInput(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    mode: Literal["beginner", "advanced"] = "beginner"
    history: List[ChatTurn] = []
    prediction: Optional[dict] = None  # compact summary of the current /predict result

# Only used when the JASPAR file is missing (exact consensus words)
KNOWN_MOTIFS = {
    "AP-1 (FOS/JUN)": "TGACTCA",
    "AP-1 (FOS/JUN) ": "TGAGTCA",
    "CRE (CREB)": "TGACGTCA",
    "TATA box (TBP)": "TATAAA",
    "GC box (SP1)": "GGGCGG",
    "CCAAT box (NF-Y)": "CCAAT",
}

# ============================================================
# LOAD MODEL (+ the cut-off tuned on validation data during training)
# ============================================================
# The website offers these trained models (ml/models/trained/, from results/run2_fosl2).
# Default = the team's proposed CNN + BiLSTM + Transformer (v2). Change with TFBS_MODEL_KEY,
# or point TFBS_MODEL at any other checkpoint file.
TRAINED_DIR = MODEL_DIR / "trained"
WEBSITE_MODELS = ["tri_branch_v2", "cnn_bilstm", "cnn", "tri_branch"]
DEFAULT_KEY = os.environ.get("TFBS_MODEL_KEY", "tri_branch_v2")
HYBRID_PATH = MODEL_DIR / "best_hybrid.pth"
CNN_PATH = MODEL_DIR / "best_cnn.pth"
TF_NAME = os.environ.get("TFBS_TF", "FOSL2")  # the factor the model was trained on
RESULTS_JSON = MODEL_DIR.parent.parent / "results" / "run2_fosl2" / "cv_results.json"


def _load_hybrid_checkpoint(checkpoint):
    """Kept for older tests: rebuild a tri-branch model from a checkpoint dict."""
    return build_hybrid_from_state(checkpoint["model_state_dict"], SEQUENCE_LENGTH)


def _test_auroc() -> dict:
    try:
        import json
        summary = json.loads(RESULTS_JSON.read_text())["summary"]
        return {k: round(v["roc_auc"]["mean"], 3) for k, v in summary.items()}
    except (OSError, KeyError, ValueError):
        return {}


LOADED = {}  # key -> (network, info)
_load_errors = []
_custom = os.environ.get("TFBS_MODEL")
if _custom:
    _p = Path(_custom) if os.path.isabs(_custom) else MODEL_DIR.parent / _custom
    LOADED["custom"] = load_model(_p, device=DEVICE, seq_len=SEQUENCE_LENGTH)
    DEFAULT_KEY = "custom"
for _key in WEBSITE_MODELS:
    _p = TRAINED_DIR / f"{_key}.pth"
    if _p.exists():
        try:
            LOADED[_key] = load_model(_p, device=DEVICE, seq_len=SEQUENCE_LENGTH)
        except RuntimeError as exc:
            _load_errors.append(f"{_p.name}: {exc}")
if not LOADED:  # older checkouts without ml/models/trained/
    for _p in (HYBRID_PATH, CNN_PATH):
        if _p.exists():
            try:
                LOADED[_p.stem] = load_model(_p, device=DEVICE, seq_len=SEQUENCE_LENGTH)
                break
            except RuntimeError as exc:
                _load_errors.append(f"{_p.name}: {exc}")
if not LOADED:
    raise FileNotFoundError(f"No usable model in {MODEL_DIR}. {' '.join(_load_errors)}")
if DEFAULT_KEY not in LOADED:
    DEFAULT_KEY = next(iter(LOADED))
TEST_AUROC = _test_auroc()

model, MODEL_INFO = LOADED[DEFAULT_KEY]
CURRENT_ARCH = MODEL_INFO["arch"]
THRESHOLD = MODEL_INFO["threshold"]
for _k, (_, _info) in LOADED.items():
    print(f"Model ready: {_k:22s} {_info['arch']} (cut-off {_info['threshold']:.2f})"
          + (" [default]" if _k == DEFAULT_KEY else ""))


def _pick(key):
    if key is None:
        key = DEFAULT_KEY
    if key not in LOADED:
        raise HTTPException(status_code=400, detail=f"Unknown model '{key}'. Available: {list(LOADED)}")
    return key, LOADED[key][0], LOADED[key][1]


def _forward_model(x, net=None):
    return (net or model)(x)


# ============================================================
# INTERPRETABILITY: 1. SHIFTSMOOTH SALIENCY (Section IV-C)
# ============================================================
def calculate_shiftsmooth_saliency(sequence: str, max_shift: int = 2, net=None):
    """Gradients averaged over small shifts of the sequence (see xai.py), scaled to 0..1."""
    x = one_hot_encode(sequence)[None]
    saliency = xai.normalise(xai.shiftsmooth(net or model, x, device=DEVICE, max_shift=max_shift)[0])
    return [round(float(v), 4) for v in saliency]


# ============================================================
# INTERPRETABILITY: 2. IN SILICO MUTAGENESIS (Section IV-C)
# ============================================================
def calculate_ism(sequence: str, base_prob: float, net=None):
    bases = ['A', 'C', 'G', 'T']
    seq_list = list(sequence)
    mutated_matrices = []
    mutations = []

    for i, orig_char in enumerate(seq_list):
        for b in bases:
            if b != orig_char:
                mut = seq_list.copy()
                mut[i] = b
                mutated_matrices.append(one_hot_encode("".join(mut)))
                mutations.append({"pos": i, "from": orig_char, "to": b})

    if not mutated_matrices:
        return [], []

    x_tensor = torch.tensor(np.array(mutated_matrices), dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        probs = torch.sigmoid(_forward_model(x_tensor, net)).cpu().numpy()

    ism_results = []
    # Per-position impact for the 3D view: the most damaging substitution at each position
    position_effect = [0.0] * len(seq_list)
    for meta, p in zip(mutations, probs):
        delta = float(np.asarray(p).squeeze() - base_prob)
        if delta < position_effect[meta["pos"]]:
            position_effect[meta["pos"]] = round(delta, 4)
        if abs(delta) > 0.03:
            ism_results.append({
                "position": meta["pos"],
                "mutation": f"{meta['from']}→{meta['to']}",
                "delta_p": round(delta, 4)
            })

    top = sorted(ism_results, key=lambda x: abs(x["delta_p"]), reverse=True)[:8]
    return top, position_effect

# ============================================================
# INTERPRETABILITY: 3. KNOWN MOTIFS (JASPAR) IN THE SEQUENCE (Section IV-C)
# ============================================================
MOTIF_DB = load_motif_db(JASPAR_DB)
_PWMS = []
for _m in MOTIF_DB:
    _pwm = to_pwm(_m["pfm"])
    _lo, _hi = score_range(_pwm)
    _PWMS.append((_m, _pwm, to_ppm(_m["pfm"]), _lo, _hi))
print(f"JASPAR motifs loaded: {len(_PWMS)}" if _PWMS else
      f"JASPAR file not found ({JASPAR_DB}); motif matching falls back to the saliency window.")

STRONG_MATCH = 0.85  # relative motif score (0 = worst possible, 1 = perfect consensus)


def _saliency_peak(saliency, width: int = 10) -> int:
    arr = np.asarray(saliency)
    sums = np.convolve(arr, np.ones(width), mode="valid")
    return int(np.argmax(sums))


def motif_analysis(sequence: str, saliency: list) -> dict:
    """Scan the sequence with every JASPAR motif and report the known motif that best explains
    the region the model found important. The logo shown is the real JASPAR matrix."""
    peak = _saliency_peak(saliency)
    sal = np.asarray(saliency)
    if not _PWMS:
        return {
            "core_sequence": sequence[peak:peak + 10],
            "window_start": peak,
            "matched_jaspar_id": "JASPAR database not installed",
            "match_score": None,
            "frequency_matrix": [],
            "known_hits": [],
        }

    x = one_hot_encode(sequence)[None]
    hits = []
    for m, pwm, ppm, lo, hi in _PWMS:
        sc = scan_scores(x, pwm)[0]                     # (2 strands, positions)
        strand_i, pos = np.unravel_index(int(sc.argmax()), sc.shape)
        rel = (float(sc[strand_i, pos]) - lo) / (hi - lo)
        if rel < 0.75:
            continue
        w = len(pwm)
        hits.append({
            "id": m["id"], "name": m["name"], "start": int(pos), "end": int(pos + w),
            "strand": "+" if strand_i == 0 else "-", "score": round(rel, 3),
            "saliency": round(float(sal[pos:pos + w].mean()), 3),
            "_ppm": ppm if strand_i == 0 else revcomp_matrix(ppm),
        })
    # Rank: strong motif match that also sits where the model is looking
    hits.sort(key=lambda h: -(h["score"] * (0.5 + h["saliency"])))
    best = hits[0] if hits else None
    trained_tf = [h for h in hits if h["name"].upper() == TF_NAME.upper()]
    if trained_tf and trained_tf[0]["score"] >= STRONG_MATCH:
        best = trained_tf[0]  # prefer the factor the model was trained on when it clearly matches

    known = [{k: v for k, v in h.items() if not k.startswith("_")} for h in hits[:8]]
    if best is None:
        return {
            "core_sequence": sequence[peak:peak + 10], "window_start": peak,
            "matched_jaspar_id": "No known motif matches the important region",
            "match_score": None, "frequency_matrix": [], "known_hits": known,
        }
    return {
        "core_sequence": sequence[best["start"]:best["end"]],
        "window_start": best["start"],
        "matched_jaspar_id": f"{best['id']} ({best['name']})",
        "match_score": best["score"],
        "strand": best["strand"],
        "frequency_matrix": [
            {b: round(float(v), 3) for b, v in zip("ACGT", row)} for row in best["_ppm"]
        ],
        "known_hits": known,
    }


# ============================================================
# ENDPOINTS
# ============================================================
@app.get("/")
async def root():
    return {
        "model": CURRENT_ARCH,
        "trained_on": TF_NAME,
        "threshold": THRESHOLD,
        "jaspar_motifs": len(_PWMS),
        "status": "operational",
        "device": str(DEVICE),
        "chatbot": "llm" if chatbot.llm_settings() else "offline"
    }


@app.get("/models")
async def list_models():
    """Trained models the website can switch between, with their 5-fold test AUROC."""
    return {
        "default": DEFAULT_KEY,
        "models": [
            {"key": k, "name": info["arch"], "threshold": round(info["threshold"], 3),
             "test_auroc": TEST_AUROC.get(k)}
            for k, (_, info) in LOADED.items()
        ],
    }


@app.post("/chat")
def chat(data: ChatInput):
    """Beginner-friendly tutor that can explain the user's current prediction."""
    return chatbot.answer(
        question=data.question,
        mode=data.mode,
        history=[t.model_dump() for t in data.history],
        prediction=data.prediction,
    )


@app.post("/predict")
async def predict_tfbs(data: SequenceInput):
    seq = data.sequence.strip().upper()

    if len(seq) != SEQUENCE_LENGTH or not all(c in "ATCG" for c in seq):
        raise HTTPException(status_code=400, detail=f"Sequence must be exactly {SEQUENCE_LENGTH} bp containing A, T, C, G.")

    key, net, info = _pick(data.model)
    threshold = info["threshold"]
    encoded = one_hot_encode(seq)
    x = torch.tensor(encoded, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        logits = _forward_model(x, net)
        probability = torch.sigmoid(logits).item()
    # 95% interval from Monte-Carlo dropout (30 extra passes with dropout switched on)
    _, ci_low, ci_high = xai.mc_dropout_interval(net, x, device=DEVICE, passes=30)

    prediction = 1 if probability >= threshold else 0
    prediction_label = "Potential TF Binding Site" if prediction == 1 else "Low TF Binding Potential"

    saliency = calculate_shiftsmooth_saliency(seq, net=net)
    ism, ism_position_effect = calculate_ism(seq, probability, net=net)
    motif_logo = motif_analysis(seq, saliency)
    detected = sorted({h["name"] for h in motif_logo["known_hits"] if h["score"] >= STRONG_MATCH})
    if not _PWMS:  # no JASPAR file: fall back to exact consensus words
        detected = [name for name, motif in KNOWN_MOTIFS.items() if motif in seq]

    response = {
        "status": "success",
        "model": info["arch"],
        "model_key": key,
        "model_test_auroc": TEST_AUROC.get(key),
        "input_length": len(seq),
        "binding_probability": round(probability, 4),
        "confidence_interval": [round(float(ci_low[0]), 4), round(float(ci_high[0]), 4)],
        "threshold": round(threshold, 3),
        "trained_on": TF_NAME,
        "prediction": prediction,
        "prediction_label": prediction_label,
        "motifs_detected": detected,
        "visualizations": {
            "saliency_map": saliency,
            "in_silico_mutagenesis": ism,
            "ism_position_effect": ism_position_effect,
            "sequence_logo": motif_logo
        }
    }
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)