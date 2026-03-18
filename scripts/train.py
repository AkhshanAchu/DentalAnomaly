import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data import OPGDataset
from src.model import CanineAnomalyModel, SVDDLoss, init_center
from src.engine import train_one_epoch, evaluate, calibrate_threshold


def parse_args():
    p = argparse.ArgumentParser(description="Train Deep SVDD canine anomaly detector")
    p.add_argument("--data_root",  default="./data",        help="Root with positive/ [negative/]")
    p.add_argument("--output",     default="./checkpoints", help="Where to save model and threshold")
    p.add_argument("--img_size",   type=int, default=224)
    p.add_argument("--embed_dim",  type=int, default=64,    help="Hypersphere embedding size")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--augment",    type=int, default=30,    help="Virtual augmentation factor per image")
    p.add_argument("--epochs",     type=int, default=150)
    p.add_argument("--warmup",     type=int, default=10,    help="Frozen-backbone warmup epochs")
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--nu",         type=float, default=0.1, help="SVDD soft-boundary nu")
    p.add_argument("--no_roi",     action="store_true",     help="Disable canine ROI crop")
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    train_ds = OPGDataset(
        root=args.data_root, mode="train",
        img_size=args.img_size,
        augment_factor=args.augment,
        use_roi_crop=not args.no_roi,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, num_workers=0, drop_last=False,
    )

    eval_ds = OPGDataset(
        root=args.data_root, mode="eval",
        img_size=args.img_size, augment_factor=1,
        use_roi_crop=not args.no_roi,
    )
    eval_loader = DataLoader(eval_ds, batch_size=4, shuffle=False, num_workers=0)

    model = CanineAnomalyModel(
        pretrained=True,
        freeze_backbone=True,
        embed_dim=args.embed_dim,
    ).to(device)

    loss_fn = SVDDLoss(nu=args.nu).to(device)
    optimizer = optim.AdamW(
        list(model.projector.parameters()) + list(loss_fn.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print("\n[Warm-up] Initialising hypersphere center...")
    init_center(model, train_loader, device)

    best_loss = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        if epoch == args.warmup + 1:
            print(f"\n[Epoch {epoch}] Unfreezing backbone for fine-tuning...")
            for p in model.backbone.parameters():
                p.requires_grad = True
            optimizer.add_param_group({
                "params": model.backbone.parameters(),
                "lr": args.lr * 0.1,
            })

        train_loss, mean_dist = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device
        )
        scheduler.step()

        history.append({"epoch": epoch, "loss": train_loss, "mean_dist": mean_dist})

        if epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch:4d}/{args.epochs} | "
                f"loss={train_loss:.4f} | mean_dist={mean_dist:.4f} | "
                f"R={loss_fn.R.item():.4f}"
            )

        if train_loss < best_loss:
            best_loss = train_loss
            torch.save({
                "epoch":     epoch,
                "model":     model.state_dict(),
                "loss_fn":   loss_fn.state_dict(),
                "center":    model.center.cpu(),
                "embed_dim": args.embed_dim,
                "img_size":  args.img_size,
            }, os.path.join(args.output, "best_model.pt"))

    print("\n[Calibration] Computing anomaly scores on eval set...")
    scores, labels = evaluate(model, eval_loader, device)
    threshold = calibrate_threshold(scores, labels)

    config = {
        "threshold": threshold,
        "embed_dim": args.embed_dim,
        "img_size":  args.img_size,
        "use_roi":   not args.no_roi,
        "interpretation": (
            "score <= threshold -> POSITIVE (canine problem detected); "
            "score > threshold  -> NEGATIVE (normal / no canine problem)"
        ),
    }
    with open(os.path.join(args.output, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n[Done] Best model + config saved to {args.output}/")
    print(f"       Anomaly threshold = {threshold:.4f}")
    print(f"       Training history  = {len(history)} epochs")


if __name__ == "__main__":
    main()
