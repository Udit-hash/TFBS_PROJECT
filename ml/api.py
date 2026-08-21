from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import torch
import numpy as np

from config import DEVICE, SEQUENCE_LENGTH
from models.cnn import TFBS_CNN
from preprocessing import one_hot_encode


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="TFBS Prediction API",
    description="CNN-based Transcription Factor Binding Site Prediction API",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# INPUT MODEL
# ============================================================

class SequenceInput(BaseModel):
    sequence: str


# ============================================================
# KNOWN MOTIFS
# ============================================================

# These are supplementary biological annotations.
# They are NOT used to generate the CNN prediction.

KNOWN_MOTIFS = {
    "TATA_BOX": "TATAAA",
    "GC_BOX": "GGGCGG",
    "CAAT_BOX": "CCAAT"
}


# ============================================================
# LOAD TRAINED CNN
# ============================================================

print("\n========================================")
print("LOADING TFBS CNN MODEL")
print("========================================")

print("Device:", DEVICE)
print("Sequence length:", SEQUENCE_LENGTH)


MODEL_PATH = "models/best_cnn.pth"


# Create the same CNN architecture
model = TFBS_CNN()


# Load trained checkpoint
checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)


# Load learned weights
model.load_state_dict(
    checkpoint["model_state_dict"]
)


# Move model to GPU
model = model.to(DEVICE)


# Evaluation mode
model.eval()


print("✓ CNN model loaded successfully")
print("✓ Model path:", MODEL_PATH)


# ============================================================
# MOTIF DETECTION
# ============================================================

def detect_motifs(sequence: str):

    found_motifs = []

    for motif_name, motif_sequence in KNOWN_MOTIFS.items():

        if motif_sequence in sequence:

            found_motifs.append(
                motif_name
            )

    return found_motifs


# ============================================================
# GRADIENT-BASED SALIENCY
# ============================================================

def calculate_saliency(sequence: str):

    """
    Calculate a simple gradient-based saliency score
    for every nucleotide position.

    Higher value = greater influence on the CNN output.

    Output:
        List of 101 values between 0 and 1.
    """

    # --------------------------------------------------------
    # One-hot encode
    # --------------------------------------------------------

    encoded = one_hot_encode(
        sequence
    )

    # --------------------------------------------------------
    # Convert to tensor
    # --------------------------------------------------------

    x = torch.tensor(
        encoded,
        dtype=torch.float32,
        device=DEVICE
    )

    # Add batch dimension
    #
    # [4, 101]
    #      ↓
    # [1, 4, 101]

    x = x.unsqueeze(0)

    # Enable gradients
    x.requires_grad_(True)

    # Clear previous gradients
    model.zero_grad()

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    output = model(x)

    # --------------------------------------------------------
    # Backpropagate from model output
    # --------------------------------------------------------

    output.backward()

    # --------------------------------------------------------
    # Get gradients
    # --------------------------------------------------------

    gradients = x.grad.detach()

    # Remove batch dimension
    #
    # [1, 4, 101]
    #      ↓
    # [4, 101]

    gradients = gradients.squeeze(0)

    # Absolute gradient
    gradients = torch.abs(
        gradients
    )

    # --------------------------------------------------------
    # Combine nucleotide channels
    # --------------------------------------------------------

    saliency = torch.max(
        gradients,
        dim=0
    ).values

    # Convert to numpy
    saliency = saliency.cpu().numpy()

    # --------------------------------------------------------
    # Normalize to 0 - 1
    # --------------------------------------------------------

    minimum = saliency.min()
    maximum = saliency.max()

    if maximum > minimum:

        saliency = (
            saliency - minimum
        ) / (
            maximum - minimum
        )

    else:

        saliency = np.zeros_like(
            saliency
        )

    # --------------------------------------------------------
    # Convert to normal Python floats
    # --------------------------------------------------------

    return [
        round(float(value), 4)
        for value in saliency
    ]


# ============================================================
# ROOT / HEALTH CHECK
# ============================================================

@app.get("/")
async def root():

    return {
        "status": "running",
        "message": "TFBS CNN Prediction API",
        "model": "CNN",
        "sequence_length": SEQUENCE_LENGTH,
        "device": str(DEVICE)
    }


# ============================================================
# MODEL INFORMATION
# ============================================================

@app.get("/model-info")
async def model_info():

    return {
        "model": "TFBS_CNN",
        "sequence_length": SEQUENCE_LENGTH,
        "input_channels": 4,
        "device": str(DEVICE),
        "checkpoint": MODEL_PATH
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
async def predict_tfbs(
    data: SequenceInput
):

    # ========================================================
    # CLEAN SEQUENCE
    # ========================================================

    seq = (
        data.sequence
        .strip()
        .upper()
    )


    # ========================================================
    # EMPTY CHECK
    # ========================================================

    if not seq:

        raise HTTPException(
            status_code=400,
            detail="Sequence is empty."
        )


    # ========================================================
    # DNA VALIDATION
    # ========================================================

    if not all(
        char in "ATCG"
        for char in seq
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid sequence. "
                "Only A, T, C, G are allowed."
            )
        )


    # ========================================================
    # LENGTH VALIDATION
    # ========================================================

    if len(seq) != SEQUENCE_LENGTH:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Sequence must be exactly "
                f"{SEQUENCE_LENGTH} bp. "
                f"Received {len(seq)} bp."
            )
        )


    # ========================================================
    # ONE-HOT ENCODING
    # ========================================================

    encoded = one_hot_encode(
        seq
    )


    # ========================================================
    # CREATE MODEL INPUT
    # ========================================================

    x = torch.tensor(
        encoded,
        dtype=torch.float32
    )

    # Add batch dimension
    #
    # [4, 101]
    #      ↓
    # [1, 4, 101]

    x = x.unsqueeze(0)

    # Move to RTX 4050
    x = x.to(DEVICE)


    # ========================================================
    # CNN PREDICTION
    # ========================================================

    model.eval()

    with torch.no_grad():

        logits = model(x)

        probability = torch.sigmoid(
            logits
        ).item()


    # ========================================================
    # CLASSIFICATION
    # ========================================================

    prediction = (
        1
        if probability >= 0.5
        else 0
    )


    if prediction == 1:

        prediction_label = (
            "Potential TF Binding Site"
        )

    else:

        prediction_label = (
            "Low TF Binding Potential"
        )


    # ========================================================
    # BIOLOGICAL MOTIFS
    # ========================================================

    found_motifs = detect_motifs(
        seq
    )


    # ========================================================
    # MODEL SALIENCY
    # ========================================================

    saliency_map = calculate_saliency(
        seq
    )


    # ========================================================
    # RESPONSE
    # ========================================================

    return {

        "status": "success",

        "model": "CNN",

        "input_length": len(seq),

        "binding_probability":
            round(
                probability,
                4
            ),

        "prediction":
            prediction,

        "prediction_label":
            prediction_label,

        "motifs_detected":
            found_motifs,

        "visualizations": {

            "saliency_map":
                saliency_map
        }
    }


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )