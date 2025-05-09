import os
import io
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchaudio
from src.model import AudioClassifier
from src.utils import compute_log_mel_spectrogram
from src.data_loader import AudioDataset

load_dotenv()
PROCESSED_DATA_DIR = os.getenv("PROCESSED_DATA_DIR", "data/processed")
METADATA_FILE = os.getenv("METADATA_FILE", "data/raw/fma_small/tracks.csv")
LABEL_FIELD = os.getenv("LABEL_FIELD", "track_genre_top")
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
MODEL_PATH = os.getenv("MODEL_PATH", "models/best_ast_fma_small.pt")

ds = AudioDataset(
    processed_data_dir=PROCESSED_DATA_DIR,
    metadata_file=METADATA_FILE,
    label_field=LABEL_FIELD,
    transform=None)

NUM_LABELS = len(ds.label_map) # should be 8 for fma small
INV_LABEL_MAP = {v: k for k, v in ds.label_map.items()}

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AudioClassifier(num_labels=NUM_LABELS)
state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state)
model.to(device).eval()

# Initialize FastAPI app
app = FastAPI(title="Audio Genre Classifier")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPPORTED_EXTENSIONS = (".wav", ".mp3")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    filename = file.filename.lower()
    if not filename.endswith(SUPPORTED_EXTENSIONS):
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Only {SUPPORTED_EXTENSIONS} files are supported.")
    
    data = await file.read()

    try:
        waveform, sr = torchaudio.load(io.BytesIO(data))
    except Exception:
        # Fallback: use librosa to load the file
        try:
            import librosa
            waveform, sr = librosa.load(io.BytesIO(data), sr=None, mono=False)
            waveform = torch.from_numpy(waveform)
        except Exception as e:
            raise HTTPException(415, f"Could not decode audio file: {e}")
    
    if waveform.dim() > 1 and waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    
    wave = waveform.squeeze(0).cpu().numpy()

    spec = compute_log_mel_spectrogram(wave, sample_rate=SAMPLE_RATE)

    spec = spec.to(device).unsqueeze(0)

    with torch.no_grad():
        logits = model(spec)
        probs = torch.softmax(logits, dim = -1)[0]
    
    results = [
        {"genre": INV_LABEL_MAP[i], "confidence": float(probs[i])}
        for i in range(NUM_LABELS)
    ]

    results.sort(key=lambda x: x["confidence"], reverse=True)

    return {"predictions": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port= int(os.getenv("PORT", 8000)),
        reload=True
    )
