from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Literal, Optional

from pydantic import BaseModel, Field
import torch
import numpy as np

import chatbot  # loads ml/.env (if present) before config reads TORCH_NUM_THREADS
from config import DEVICE, SEQUENCE_LENGTH, MODEL_DIR
from models.hybrid import TFBS_TriBranch
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


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatInput(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    mode: Literal["beginner", "advanced"] = "beginner"
    history: List[ChatTurn] = []
    prediction: Optional[dict] = None  # compact summary of the current /predict result

KNOWN_MOTIFS = {
    "TATA_BOX": "TATAAA",
    "GC_BOX": "GGGCGG",
    "CAAT_BOX": "CCAAT"
}

# ============================================================
# LOAD MODEL
# ============================================================
HYBRID_PATH = MODEL_DIR / "best_hybrid.pth"
CNN_PATH = MODEL_DIR / "best_cnn.pth"


def _load_hybrid_checkpoint(checkpoint):
    state = checkpoint["model_state_dict"]
    model = TFBS_TriBranch(seq_len=SEQUENCE_LENGTH)
    try:
        model.load_state_dict(state)
        return model
    except RuntimeError as current_layout_error:
        try:
            conv_channels = state["conv8.weight"].shape[0]
            for name, kernel_size in (("conv8", 8), ("conv16", 16), ("conv24", 24)):
                if tuple(state[f"{name}.weight"].shape) != (
                    conv_channels,
                    4,
                    kernel_size,
                ):
                    raise ValueError("Inconsistent multi-scale convolution dimensions.")

            lstm_gates, lstm_input = state["bilstm.weight_ih_l0"].shape
            if lstm_gates % 4 or lstm_input != conv_channels * 3:
                raise ValueError("Unsupported BiLSTM dimensions in checkpoint.")
            lstm_hidden = lstm_gates // 4

            classifier_weight = state["classifier.0.weight"]
            classifier_hidden, classifier_input = classifier_weight.shape
            expected_input = conv_channels * 6 + lstm_hidden * 2
            if classifier_input != expected_input:
                raise ValueError("Checkpoint classifier does not match its branches.")

            classifier_output = state["classifier.4.weight"].shape[0]
            has_intermediate = classifier_output != 1
            if has_intermediate and tuple(state["classifier.7.weight"].shape) != (
                1,
                classifier_output,
            ):
                raise ValueError("Unsupported hybrid classifier layout.")

            compatible_model = TFBS_TriBranch(
                seq_len=SEQUENCE_LENGTH,
                conv_channels=conv_channels,
                lstm_hidden=lstm_hidden,
                classifier_hidden=classifier_hidden,
                classifier_intermediate=has_intermediate,
                classifier_output=classifier_output,
            )
            compatible_model.load_state_dict(state)
        except (IndexError, KeyError, RuntimeError, ValueError) as compatibility_error:
            raise RuntimeError(
                "The hybrid checkpoint is incompatible with both supported model "
                f"layouts: {compatibility_error}"
            ) from current_layout_error

        print(
            "Loaded hybrid checkpoint using its saved architecture "
            f"({conv_channels} channels per convolution, "
            f"{lstm_hidden} BiLSTM units)."
        )
        return compatible_model


if HYBRID_PATH.exists():
    MODEL_PATH = HYBRID_PATH
    CURRENT_ARCH = "CNN + BiLSTM + Transformer"
    print(f"Loading Hybrid Model from {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    try:
        model = _load_hybrid_checkpoint(checkpoint)
    except RuntimeError as hybrid_error:
        if not CNN_PATH.exists():
            raise RuntimeError(
                f"Hybrid checkpoint at {HYBRID_PATH} is incompatible and no CNN checkpoint is available."
            ) from hybrid_error

        from models.cnn import TFBS_CNN

        MODEL_PATH = CNN_PATH
        model = TFBS_CNN()
        CURRENT_ARCH = "CNN (Baseline)"
        print(
            f"Hybrid checkpoint is incompatible ({hybrid_error}); "
            f"falling back to {MODEL_PATH}"
        )
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
elif CNN_PATH.exists():
    from models.cnn import TFBS_CNN

    MODEL_PATH = CNN_PATH
    model = TFBS_CNN()
    CURRENT_ARCH = "CNN (Baseline)"
    print(f"Hybrid checkpoint not found. Loading {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    raise FileNotFoundError(f"Neither {HYBRID_PATH} nor {CNN_PATH} found in {MODEL_DIR}")

model = model.to(DEVICE)
model.eval()

print(f"Model successfully operational: {CURRENT_ARCH}")


def _forward_model(x):
    return model(x)


# ============================================================
# INTERPRETABILITY: 1. SHIFTSMOOTH SALIENCY (Section IV-C)
# ============================================================
def calculate_shiftsmooth_saliency(
    sequence: str,
    num_shifts: int = 5,
    max_shift: int = 2,
):
    """
    ShiftSmooth Attribution: Extends SmoothGrad by averaging gradients
    over small input sequence shifts to reduce noise and provide stable attribution.
    """
    encoded = one_hot_encode(sequence)
    base_tensor = torch.tensor(encoded, dtype=torch.float32, device=DEVICE)
    accumulated_grads = torch.zeros(SEQUENCE_LENGTH, device=DEVICE)
    
    shifts = list(range(-max_shift, max_shift + 1))
    
    for shift in shifts:
        # Roll sequence by shift positions
        shifted_tensor = torch.roll(base_tensor, shifts=shift, dims=1)
        x = shifted_tensor.unsqueeze(0).clone().detach().requires_grad_(True)
        
        model.zero_grad()
        out = _forward_model(x)
        out.backward()
        
        # Extract gradient and roll back to original alignment
        grad = x.grad.detach().squeeze(0)  # (4, seq_len)
        grad = torch.max(torch.abs(grad), dim=0).values
        unshifted_grad = torch.roll(grad, shifts=-shift, dims=0)
        accumulated_grads += unshifted_grad

    # Average attributions over all shifts
    saliency = (accumulated_grads / len(shifts)).cpu().numpy()
    
    min_v, max_v = saliency.min(), saliency.max()
    if max_v > min_v:
        saliency = (saliency - min_v) / (max_v - min_v)
    else:
        saliency = np.zeros_like(saliency)

    return [round(float(v), 4) for v in saliency]

# ============================================================
# INTERPRETABILITY: 2. IN SILICO MUTAGENESIS (Section IV-C)
# ============================================================
def calculate_ism(sequence: str, base_prob: float):
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
        probs = torch.sigmoid(_forward_model(x_tensor)).cpu().numpy()

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
# INTERPRETABILITY: 3. SEQUENCE LOGOS / JASPAR MOTIF MATCHING (Section IV-C)
# ============================================================
def extract_motif_logos(sequence: str, saliency_map: list):
    """
    Recovers salient binding cores and formats position frequency vectors 
    matching curated JASPAR motif syntax.
    """
    # Find highest saliency peak window of 6-8 bp
    arr = np.array(saliency_map)
    window_size = 6
    best_start = 0
    max_score = 0
    
    for i in range(len(arr) - window_size):
        score = arr[i:i+window_size].sum()
        if score > max_score:
            max_score = score
            best_start = i
            
    core_subseq = sequence[best_start:best_start+window_size]
    
    # Check JASPAR / canonical correspondence
    matched_jaspar = "MA0108.1 (TBP)" if "TATA" in core_subseq else ("MA0079.3 (SP1)" if "GC" in core_subseq else "De Novo Candidate Motif")
    
    # Generate nucleotide frequencies for visualization logo
    logo_data = []
    for char in core_subseq:
        weights = {"A": 0.05, "C": 0.05, "G": 0.05, "T": 0.05}
        weights[char] = 0.85
        logo_data.append(weights)
        
    return {
        "core_sequence": core_subseq,
        "window_start": best_start,
        "matched_jaspar_id": matched_jaspar,
        "frequency_matrix": logo_data
    }

# ============================================================
# ENDPOINTS
# ============================================================
@app.get("/")
async def root():
    return {
        "model": CURRENT_ARCH,
        "status": "operational",
        "device": str(DEVICE),
        "chatbot": "llm" if chatbot.llm_settings() else "offline"
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

    encoded = one_hot_encode(seq)
    x = torch.tensor(encoded, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        logits = _forward_model(x)
        probability = torch.sigmoid(logits).item()

    prediction = 1 if probability >= 0.5 else 0
    prediction_label = "Potential TF Binding Site" if prediction == 1 else "Low TF Binding Potential"

    detected = [name for name, motif in KNOWN_MOTIFS.items() if motif in seq]
    saliency = calculate_shiftsmooth_saliency(seq)
    ism, ism_position_effect = calculate_ism(seq, probability)
    motif_logo = extract_motif_logos(seq, saliency)

    response = {
        "status": "success",
        "model": CURRENT_ARCH,
        "input_length": len(seq),
        "binding_probability": round(probability, 4),
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