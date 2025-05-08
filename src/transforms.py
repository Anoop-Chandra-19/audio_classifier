import torch
import torchaudio

class SpecAugment(torch.nn.Module):
    def __init__(self,
                 time_mask_param: int = 40,
                 freq_mask_param: int = 20,
                 num_time_masks: int = 2,
                 num_freq_masks: int = 2
                ):
        super().__init__()
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.num_time_masks = num_time_masks
        self.num_freq_masks = num_freq_masks

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        # spec : [n_mels, time]

        for _ in range(self.num_freq_masks):
            spec = torchaudio.transforms.FrequencyMasking(self.freq_mask_param)(spec)
        for _ in range(self.num_time_masks):
            spec = torchaudio.transforms.TimeMasking(self.time_mask_param)(spec)
        return spec