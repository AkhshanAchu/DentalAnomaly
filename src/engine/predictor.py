import os
import json
import torch
from PIL import Image

from src.model import CanineAnomalyModel
from src.data import crop_canine_roi, get_eval_transforms


def load_model(checkpoint_dir: str, device: torch.device):
    ckpt_path = os.path.join(checkpoint_dir, "best_model.pt")
    config_path = os.path.join(checkpoint_dir, "config.json")

    with open(config_path) as f:
        config = json.load(f)

    ckpt = torch.load(ckpt_path, map_location=device)
    model = CanineAnomalyModel(
        pretrained=False,
        freeze_backbone=False,
        embed_dim=config["embed_dim"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.center.copy_(ckpt["center"].to(device))
    model.eval()

    return model, config


@torch.no_grad()
def predict_image(img_path: str, model, config: dict, device: torch.device) -> dict:
    transform = get_eval_transforms(config["img_size"])

    img = Image.open(img_path).convert("L")
    if config.get("use_roi", True):
        img = crop_canine_roi(img)

    tensor = transform(img).unsqueeze(0).to(device)
    score = model.anomaly_score(tensor).item()
    threshold = config["threshold"]
    has_problem = score <= threshold

    return {
        "file":       os.path.basename(img_path),
        "score":      round(score, 4),
        "threshold":  round(threshold, 4),
        "prediction": "POSITIVE — Canine problem detected" if has_problem
                      else "NEGATIVE — No canine problem detected",
        "confidence": _confidence(score, threshold),
    }


def _confidence(score: float, threshold: float) -> str:
    ratio = score / (threshold + 1e-8)
    if ratio < 0.5:
        return "High confidence (well inside hypersphere)"
    elif ratio < 0.85:
        return "Moderate confidence"
    elif ratio < 1.15:
        return "Low confidence (near decision boundary)"
    else:
        return "High confidence (well outside hypersphere)"
