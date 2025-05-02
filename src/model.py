
import os
from dotenv import load_dotenv
import torch
import torch.nn as nn
from transformers import AutoFeatureExtractor, ASTForAudioClassification

load_dotenv()

hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
class AudioClassifier(nn.Module):
    """
    Audio Spectrogram Transformer for classfication on FMA-small.
    Wraps a pretrained AST model with a new classifier head.
    """
    def __init__(
            self,
            num_labels: int,
            pretrained_model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
    ):
        super().__init__()
        # Feature extractor: turns log-mel spectrograms into model inputs
        self.extractor = AutoFeatureExtractor.from_pretrained(pretrained_model_name, 
                                                              token=hf_token, sample_rate=16000)
        # AST Model with a classification head of size 'num_labels'
        self.model = ASTForAudioClassification.from_pretrained(pretrained_model_name, 
                                                               num_labels = num_labels, token=hf_token,  
                                                               ignore_mismatched_sizes = True)
        self.mean = self.extractor.mean
        self.std = self.extractor.std
        self.max_length = self.extractor.max_length

    def forward(self, specs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            specs: float Tensor of shape (batch_size, n_mels, time_frames)
                   containing log-mel spectrograms
        Returns:
            logits: float Tensor of shape (batch_size, num_labels)
                    containing the model's predictions
        """
        # AST expects inputs as images: add a channel dimension and normalize
        # extractor will handle any resizing / normalization needed
        
        # Extractor expects inputs on the same device as the model
        specs = specs.to(next(self.model.parameters()).device)

        # specs must be (B, 128, T)
        if specs.ndim != 3:
            raise ValueError(f"Expected specs.shape==(B,128,T), got {specs.ndim} dimensions")

        # pad or truncate time dim to exactly max_length
        B, H, T = specs.shape
        if T < self.max_length:
            # pad on the right
            pad_amt = self.max_length - T
            specs = torch.nn.functional.pad(specs, (0, pad_amt))
        elif T > self.max_length:
            # truncate on the right
            specs = specs[:, :, :self.max_length]
        
        # AST expects inputs as (B, T, H)
        pixel_values = specs.permute(0, 2, 1)        
        # Normalize the input
        pixel_values = (pixel_values - self. mean) / self.std

        # Forward pass through AST
        outputs = self.model(input_values = pixel_values)
        return outputs.logits

        
        