from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import random
import asyncio

app = FastAPI(title="TFBS Prediction API (Heuristic Engine)")

# Allow requests from the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SequenceInput(BaseModel):
    sequence: str


# Known biological motifs
KNOWN_MOTIFS = {
    "TATA_BOX": "TATAAA",
    "GC_BOX": "GGGCGG",
    "CAAT_BOX": "CCAAT",
}


@app.post("/predict")
async def predict_tfbs(data: SequenceInput):

    # Convert sequence to uppercase and remove whitespace
    seq = data.sequence.strip().upper()

    # 1. Strict validation
    if not seq:
        raise HTTPException(
            status_code=400,
            detail="Sequence is empty."
        )

    if not all(char in "ATCG" for char in seq):
        raise HTTPException(
            status_code=400,
            detail="Invalid sequence. Only A, T, C, G are allowed."
        )

    seq_length = len(seq)

    # Simulate processing time
    await asyncio.sleep(1.2)

    # 2. Base saliency and probability
    saliency_map = [
        round(random.uniform(0.01, 0.15), 3)
        for _ in range(seq_length)
    ]

    base_probability = random.uniform(0.10, 0.35)

    found_motifs = []

    # 3. Scan for known biological motifs
    for motif_name, motif_seq in KNOWN_MOTIFS.items():

        start_idx = seq.find(motif_seq)

        if start_idx != -1:

            found_motifs.append(motif_name)

            # Increase probability when a motif is found
            base_probability = random.uniform(0.88, 0.98)

            # Highlight the motif in the saliency map
            for i in range(len(motif_seq)):
                saliency_map[start_idx + i] = round(
                    random.uniform(0.85, 1.0),
                    3
                )

            # Add contextual halo around motif
            if start_idx > 0:
                saliency_map[start_idx - 1] = 0.4

            if start_idx + len(motif_seq) < seq_length:
                saliency_map[start_idx + len(motif_seq)] = 0.4

    # 4. Return prediction result
    return {
        "status": "success",
        "input_length": seq_length,
        "binding_probability": round(base_probability, 4),
        "motifs_detected": found_motifs,
        "visualizations": {
            "saliency_map": saliency_map
        }
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )