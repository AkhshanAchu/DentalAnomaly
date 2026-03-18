import torch
import torch.nn as nn


class SVDDLoss(nn.Module):
    def __init__(self, nu: float = 0.1):
        super().__init__()
        self.nu = nu
        self.R = nn.Parameter(torch.tensor(0.0))

    def forward(self, embeddings: torch.Tensor, center: torch.Tensor):
        dist = torch.norm(embeddings - center, dim=1)
        scores = dist - self.R
        loss = self.R ** 2 + (1 / self.nu) * torch.mean(torch.clamp(scores, min=0) ** 2)
        return loss, dist.detach()


@torch.no_grad()
def init_center(model, loader, device, eps: float = 0.1) -> None:
    model.eval()
    embeddings = []
    for imgs, _, _ in loader:
        imgs = imgs.to(device)
        emb = model(imgs)
        embeddings.append(emb.cpu())

    c = torch.cat(embeddings, dim=0).mean(dim=0)
    c[(c.abs() < eps) & (c >= 0)] = eps
    c[(c.abs() < eps) & (c < 0)] = -eps

    model.center.copy_(c.to(device))
    print(f"[SVDD] Center initialised | norm={c.norm():.4f}")
