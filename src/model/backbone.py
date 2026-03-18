import torch
import torch.nn as nn
from torchvision import models


class EfficientNetFeatureExtractor(nn.Module):
    def __init__(self, pretrained: bool = True, freeze: bool = True):
        super().__init__()
        base = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        if freeze:
            for p in self.features.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return x.flatten(1)

    @property
    def out_dim(self) -> int:
        return 1280
