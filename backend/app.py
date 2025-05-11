import os
import io
import json
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchaudio
import numpy as np

from src.model import AudioClassifier
from src.utils import compute_log_mel_spectrogram

# App setup 
load_dotenv()
app = FastAPI(title="Audio Genre Classifier")
origins = [
    "https://audioclassifier.cc",
    "https://www.audioclassifier.cc",
    "https://audioclassifier.cc"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Config & label map
MODEL_PATH  = os.getenv("MODEL_PATH", "models/best_ast_fma_small.pt")
LABEL_FILE  = os.getenv("LABEL_FILE", "models/labels.json")
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", 16000))
SUPPORTED_EXTS = (".wav", ".mp3")

# Load labels.json → dict[int, str]
with open(LABEL_FILE, "r") as f:
    raw_labels = json.load(f)
LABEL_MAP = {int(k): v for k, v in raw_labels.items()}
NUM_LABELS = len(LABEL_MAP)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AudioClassifier(num_labels=NUM_LABELS)
state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state)
model.to(device).eval()

# Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1) Check extension
    ext = os.path.splitext(file.filename.lower())[1]
    if ext not in SUPPORTED_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type {ext!r}. Allowed: {SUPPORTED_EXTS}"
        )

    # 2) Read bytes
    data = await file.read()

    # 3) Try torchaudio.load
    try:
        waveform, sr = torchaudio.load(io.BytesIO(data))
    except Exception:
        # fallback: librosa
        try:
            import librosa
            wav_np, sr = librosa.load(io.BytesIO(data), sr=None, mono=False)
            waveform = torch.from_numpy(wav_np)
        except Exception as e:
            raise HTTPException(415, f"Could not decode audio: {e}")

    # 4) Mono & resample
    if waveform.dim() > 1 and waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=sr, new_freq=SAMPLE_RATE
        )

    # 5) To NumPy for librosa‐style spec (your util expects np.ndarray)
    wave_np = waveform.squeeze(0).cpu().numpy()

    # 6) Compute log-mel spectrogram
    spec = compute_log_mel_spectrogram(wave_np, sample_rate=SAMPLE_RATE)
    # spec is a torch.Tensor on CPU, shape (128, T)

    # 7) Batch + to device
    spec = spec.to(device).unsqueeze(0)  # (1, 128, T)

    # 8) Inference
    with torch.no_grad():
        logits = model(spec)                     # (1, NUM_LABELS)
        probs  = torch.softmax(logits, dim=-1)[0]  # (NUM_LABELS,)

    # 9) Sort & build response
    probs_np = probs.cpu().numpy()
    idxs = np.argsort(probs_np)[::-1]  # descending
    results = [
        {"genre": LABEL_MAP[i], "confidence": float(probs_np[i])}
        for i in idxs
    ]

    return {"predictions": results}


# Optional: health check
@app.get("/health")
def health():
    return {"status": "ok"}


# Entrypoint
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True
    )
