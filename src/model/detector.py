import torch
import torch.nn as nn

from .backbone import EfficientNetFeatureExtractor
from .head import ProjectionHead


class CanineAnomalyModel(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        embed_dim: int = 64,
    ):
        super().__init__()
        self.backbone = EfficientNetFeatureExtractor(pretrained, freeze_backbone)
        self.projector = ProjectionHead(self.backbone.out_dim, embed_dim)
        self.embed_dim = embed_dim
        self.register_buffer("center", torch.zeros(embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.projector(feats)

    @torch.no_grad()
    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.forward(x)
        return torch.norm(emb - self.center, dim=1)
