from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import numpy as np

from config import DEVICE, SEQUENCE_LENGTH, MODEL_DIR
from models.hybrid import TFBS_TriBranch
from models.shape_fusion import TFBS_TriBranchShape
from dna_shape import DNAShapePredictionError, DNAshapeRPredictor, FEATURE_NAMES
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
SHAPE_PATH = MODEL_DIR / "best_shape_fusion.pth"
shape_predictor = None
shape_matrix_predictor = DNAshapeRPredictor()
SHAPE_MODEL_ACTIVE = False
SHAPE_POSITIONS = None
SHAPE_NATIVE_LENGTHS = None


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


if SHAPE_PATH.exists():
    checkpoint = torch.load(SHAPE_PATH, map_location=DEVICE)
    expected_names = tuple(checkpoint.get("shape_feature_names", ()))
    if checkpoint.get("model_type") != "tri_branch_dna_shape":
        raise RuntimeError(f"Unsupported DNA-shape checkpoint: {SHAPE_PATH}")
    if checkpoint.get("sequence_length") != SEQUENCE_LENGTH:
        raise RuntimeError(
            f"DNA-shape checkpoint expects {checkpoint.get('sequence_length')} bp; "
            f"the current API is configured for {SEQUENCE_LENGTH} bp."
        )
    if expected_names != FEATURE_NAMES:
        raise RuntimeError(
            f"DNA-shape checkpoint features {expected_names} do not match "
            f"{FEATURE_NAMES}."
        )
    shape_native_lengths = tuple(
        checkpoint.get("shape_native_lengths", ())
    )
    if len(shape_native_lengths) != len(FEATURE_NAMES) or min(
        shape_native_lengths, default=0
    ) < 1:
        raise RuntimeError(
            "DNA-shape checkpoint must store a native position count per feature."
        )
    shape_mean = torch.tensor(checkpoint["shape_mean"], dtype=torch.float32)
    shape_std = torch.tensor(checkpoint["shape_std"], dtype=torch.float32)
    model = TFBS_TriBranchShape(
        seq_len=SEQUENCE_LENGTH,
        shape_mean=shape_mean,
        shape_std=shape_std,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()
    shape_predictor = shape_matrix_predictor
    if shape_predictor.version != checkpoint.get("shape_predictor_version"):
        raise RuntimeError(
            "Installed DNAshapeR version does not match the version used to "
            "train the shape-fusion checkpoint."
        )
    CURRENT_ARCH = "CNN + BiLSTM + Transformer + DNAshapeR"
    SHAPE_MODEL_ACTIVE = True
    SHAPE_POSITIONS = int(checkpoint.get("shape_positions", 0))
    SHAPE_NATIVE_LENGTHS = shape_native_lengths
    if SHAPE_POSITIONS != max(shape_native_lengths):
        raise RuntimeError(
            "DNA-shape checkpoint max position count does not match its "
            "per-feature native lengths."
        )
    print(f"Loading DNA-shape fusion model from {SHAPE_PATH}")
elif HYBRID_PATH.exists():
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


def _forward_model(x, shape_features=None):
    if SHAPE_MODEL_ACTIVE:
        if shape_features is None:
            raise RuntimeError("DNA-shape features are required by the loaded model.")
        return model(x, shape_features)
    return model(x)


# ============================================================
# INTERPRETABILITY: 1. SHIFTSMOOTH SALIENCY (Section IV-C)
# ============================================================
def calculate_shiftsmooth_saliency(
    sequence: str,
    shape_features=None,
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
        out = _forward_model(x, shape_features)
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
        return []

    x_tensor = torch.tensor(np.array(mutated_matrices), dtype=torch.float32, device=DEVICE)
    shape_tensor = None
    if SHAPE_MODEL_ACTIVE:
        if shape_predictor is None:
            raise RuntimeError("DNAshapeR predictor is unavailable.")
        mutated_sequences = [
            "".join(
                (mutation["to"] if position == mutation["pos"] else base)
                for position, base in enumerate(sequence)
            )
            for mutation in mutations
        ]
        shape_batch = shape_predictor.predict_batch(mutated_sequences)
        if (
            shape_batch.values.shape[2] != SHAPE_POSITIONS
            or shape_batch.native_lengths != SHAPE_NATIVE_LENGTHS
        ):
            raise DNAShapePredictionError(
                "DNAshapeR returned native feature dimensions that differ "
                "from the shape model."
            )
        shape_tensor = torch.tensor(
            shape_batch.values, dtype=torch.float32, device=DEVICE
        )
    with torch.no_grad():
        probs = torch.sigmoid(_forward_model(x_tensor, shape_tensor)).cpu().numpy()

    ism_results = []
    for meta, p in zip(mutations, probs):
        delta = float(np.asarray(p).squeeze() - base_prob)
        if abs(delta) > 0.03:
            ism_results.append({
                "position": meta["pos"],
                "mutation": f"{meta['from']}→{meta['to']}",
                "delta_p": round(delta, 4)
            })

    return sorted(ism_results, key=lambda x: abs(x["delta_p"]), reverse=True)[:8]

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
        "device": str(DEVICE)
    }


@app.post("/dna-shape")
async def predict_dna_shape(data: SequenceInput):
    seq = data.sequence.strip().upper()
    if len(seq) != SEQUENCE_LENGTH or not all(c in "ATCG" for c in seq):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Sequence must be exactly {SEQUENCE_LENGTH} bp containing "
                "A, T, C, G."
            ),
        )

    try:
        shape = shape_matrix_predictor.predict(seq)
    except DNAShapePredictionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    rows = []
    for position in range(shape.values.shape[1]):
        row = {"position": position + 1}
        for index, feature_name in enumerate(FEATURE_NAMES):
            value = shape.values[index, position]
            row[feature_name] = (
                round(float(value), 4)
                if position < shape.native_lengths[index] and np.isfinite(value)
                else None
            )
        rows.append(row)

    return {
        "predictor": shape.predictor,
        "predictor_version": shape.predictor_version,
        "feature_names": list(shape.feature_names),
        "native_lengths": list(shape.native_lengths),
        "rows": rows,
    }


@app.post("/predict")
async def predict_tfbs(data: SequenceInput):
    seq = data.sequence.strip().upper()

    if len(seq) != SEQUENCE_LENGTH or not all(c in "ATCG" for c in seq):
        raise HTTPException(status_code=400, detail=f"Sequence must be exactly {SEQUENCE_LENGTH} bp containing A, T, C, G.")

    encoded = one_hot_encode(seq)
    x = torch.tensor(encoded, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    shape_tensor = None
    shape_metadata = None
    if SHAPE_MODEL_ACTIVE:
        if shape_predictor is None:
            raise RuntimeError("DNAshapeR predictor is unavailable.")
        try:
            shape_batch = shape_predictor.predict_batch([seq])
        except DNAShapePredictionError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        if shape_batch.values.shape[2] != SHAPE_POSITIONS:
            raise HTTPException(
                status_code=503,
                detail=(
                    "DNAshapeR returned a different number of native positions "
                    "than the shape model was trained with."
                ),
            )
        if shape_batch.native_lengths != SHAPE_NATIVE_LENGTHS:
            raise HTTPException(
                status_code=503,
                detail=(
                    "DNAshapeR returned per-feature native lengths different "
                    "from those used to train the shape model."
                ),
            )
        shape_tensor = torch.tensor(
            shape_batch.values, dtype=torch.float32, device=DEVICE
        )
        shape_metadata = {
            "predictor": shape_batch.predictor,
            "predictor_version": shape_batch.predictor_version,
            "feature_names": list(shape_batch.feature_names),
            "native_position_count": int(shape_batch.values.shape[2]),
            "native_lengths": list(shape_batch.native_lengths),
        }

    with torch.no_grad():
        logits = _forward_model(x, shape_tensor)
        probability = torch.sigmoid(logits).item()

    prediction = 1 if probability >= 0.5 else 0
    prediction_label = "Potential TF Binding Site" if prediction == 1 else "Low TF Binding Potential"

    detected = [name for name, motif in KNOWN_MOTIFS.items() if motif in seq]
    saliency = calculate_shiftsmooth_saliency(seq, shape_tensor)
    try:
        ism = calculate_ism(seq, probability)
    except DNAShapePredictionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
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
            "sequence_logo": motif_logo
        }
    }
    if shape_metadata is not None:
        response["dna_shape"] = shape_metadata
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)