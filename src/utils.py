import numpy as np
import torch
import librosa

def load_audio(path: str, sample_rate: int = 16000) -> np.ndarray:
    y, sr = librosa.load(path, sr=sample_rate, mono=True)
    return y

def compute_log_mel_spectrogram(
        waveform : np.ndarray, 
        sample_rate: int = 16000, 
        n_mels: int = 128, 
        n_fft: int = 1024, 
        hop_length: int = 512
        ) -> torch.Tensor:
    # power spectrogram
    S = librosa.feature.melspectrogram(
        y=waveform, 
        sr=sample_rate, 
        n_fft=n_fft, 
        hop_length=hop_length, 
        n_mels=n_mels
    )
    S_db = librosa.power_to_db(S, ref=np.max)

    #return as (n_mels, time) float32 tensor
    return torch.from_numpy(S_db).to(torch.float32)
