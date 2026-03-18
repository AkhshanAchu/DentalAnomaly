import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.engine import load_model, predict_image

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def parse_args():
    p = argparse.ArgumentParser(description="Run inference on OPG dental X-rays")
    p.add_argument("--image",      default=None,            help="Path to a single OPG image")
    p.add_argument("--folder",     default=None,            help="Folder of OPG images to batch-predict")
    p.add_argument("--checkpoint", default="./checkpoints", help="Folder with best_model.pt and config.json")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_model(args.checkpoint, device)
    print(f"[Loaded] threshold={config['threshold']:.4f}\n")

    paths = []
    if args.image:
        paths = [args.image]
    elif args.folder:
        paths = [
            os.path.join(args.folder, f)
            for f in sorted(os.listdir(args.folder))
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS
        ]
    else:
        print("Provide --image or --folder")
        return

    for path in paths:
        result = predict_image(path, model, config, device)
        print("-" * 60)
        print(f"File:       {result['file']}")
        print(f"Score:      {result['score']}  (threshold={result['threshold']})")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']}")

    print(f"\n{'-' * 60}")
    print(f"Processed {len(paths)} image(s).")


if __name__ == "__main__":
    main()
