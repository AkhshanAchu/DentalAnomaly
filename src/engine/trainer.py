import numpy as np
import torch


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    all_dists = []

    for imgs, _, _ in loader:
        imgs = imgs.to(device)
        emb = model(imgs)
        loss, dists = loss_fn(emb, model.center)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        all_dists.extend(dists.cpu().numpy().tolist())

    return total_loss / len(loader), np.mean(all_dists)
