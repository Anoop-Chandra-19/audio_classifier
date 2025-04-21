import numpy as np
import torch
import librosa

def load_audio(path: str, sample_rate: int = 16000) -> np.ndarray:
    '''
    Loads an audio file and returns the waveform as a numpy array.
    
    Args:
        path (str): Path to the audio file.
        sample_rate (int): Sample rate for loading the audio. Default is 16000 Hz.
    
    Returns:
        np.ndarray: Waveform of the audio file.
    '''
    y, sr = librosa.load(path, sr=sample_rate, mono=True)
    return y

def compute_log_mel_spectrogram(
        waveform : np.ndarray, 
        sample_rate: int = 16000, 
        n_mels: int = 128, 
        n_fft: int = 1024, 
        hop_length: int = 512
        ) -> torch.Tensor:
    '''
    Computes the log-mel spectrogram of a waveform.
    
    Args:
        waveform (np.ndarray): Waveform of the audio file.
        sample_rate (int): Sample rate for loading the audio. Default is 16000 Hz.
        n_mels (int): Number of mel bands to generate. Default is 128.
        n_fft (int): Length of the FFT window. Default is 1024.
        hop_length (int): Number of samples between frames. Default is 512.
    
    Returns:
        torch.Tensor: Log-mel spectrogram of the audio file.
    '''
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
