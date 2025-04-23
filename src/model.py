
import torch
import torch.nn as nn
from transformers import AutoFeatureExtractor, ASTForAudioClassification

class AudioClassifier(nn.Module):
    """
    Audio Spectrogram Transformer for classfication on FMA-small.
    Wraps a pretrained AST model with a new classifier head.
    """
    def __init__(
            self,
            num_labels: int,
            pretrained_model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
            feature_extractor_name: str = "MIT/ast-base"
    ):
        super().__init__()
        # Feature extractor: turns log-mel spectrograms into model inputs
        self.extractor = AutoFeatureExtractor.from_pretrained(feature_extractor_name)
        # AST Model with a classification head of size 'num_labels'
        self.model = ASTForAudioClassification.from_pretrained(pretrained_model_name, num_labels = num_labels)
    
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
        specs = specs.unsqueeze(1)

        cpu_specs = specs.detach().cpu().numpy()
        pixel_values = self.extractor(cpu_specs, return_tensors="pt").pixel_values

        # Move pixel values to the same device as the model
        pixel_values = pixel_values.to(next(self.model.parameters()).device)

        # Forward pass through AST
        outputs = self.model(pixel_values=pixel_values)
        return outputs.logits

        
        