import torch
from torch import nn
from typing import Tuple
from model_parts import *

class KeyViTSeg(nn.Module):
    def __init__(self,  model_type: str, version: str, stride: int, facet: str) -> None:
        
        super(KeyViTSeg, self).__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.extractor = ViTExtractor(model_type=model_type, stride=stride, device=device, version=version, facet=facet)
        self.adapter_proj = nn.Conv2d(384, 1024, kernel_size=1)
        self.decoder = Decoder_classic(n_channels=3, n_classes=2, bilinear=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        features = self.extractor(x)

        if features.dim() == 4:
            if features.shape[1] == 1:
                features = features.squeeze(1)

            if features.shape[-1] == 384:
                batch_size = features.shape[0]
                num_patches = features.shape[1]
                h = w = int(num_patches ** 0.5)
                features = features.view(batch_size, h, w, 384).permute(0, 3, 1, 2).contiguous()

        projected_features = self.adapter_proj(features)
        out = self.decoder(projected_features)
        return out
    
