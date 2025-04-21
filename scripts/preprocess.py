# This script processes audio files in the specified directory, 
# computes their log-mel spectrograms, and saves them as PyTorch tensors.

# This script is intended to be run from the command line.
# Though I initially had problems with running the script with uv package manager,
# I was able to run it successfully using the command: uv run python -m scripts/preprocess.py
# I recommend doing the same if you encounter issues with uv.

import os
import sys
import torch
from tqdm import tqdm
from src.utils import load_audio, compute_log_mel_spectrogram # custom utility functions
# Add the src directory to the Python path
# so that we can import the utils module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

RAW_AUDIO_DIR = "data/raw/fma_small"
OUT_DIR = "data/processed"
audio_paths = []
os.makedirs(OUT_DIR, exist_ok=True)

# Collect all audio file paths
# in the specified directory and its subdirectories
# and store them in a list
# The audio files should be in .wav or .mp3 format

for root, _, files in os.walk(RAW_AUDIO_DIR):
    for fname in files:
        if fname.lower().endswith((".wav", ".mp3")):
            audio_paths.append(os.path.join(root, fname))

# Process each audio file
# Compute the log-mel spectrogram
# and save it as a PyTorch tensor
# The output file name should be the same as the input file name
# but with a .pt extension

for in_path in tqdm(audio_paths, desc="Processing audio files"):
    track_id = os.path.splitext(os.path.basename(in_path))[0]  # eg. "000002"
    try:
        waveform = load_audio(in_path)
        spec = compute_log_mel_spectrogram(waveform)
        out_path = os.path.join(OUT_DIR, f"{track_id}.pt")
        torch.save(spec, out_path)
    except Exception as e:
        print(f"Failed to process {track_id}: {e}")            


print("All files processed.")

