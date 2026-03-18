import numpy as np
import torch


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    scores, labels = [], []

    for imgs, lbls, _ in loader:
        imgs = imgs.to(device)
        s = model.anomaly_score(imgs)
        scores.extend(s.cpu().numpy().tolist())
        labels.extend(lbls.numpy().tolist())

    return np.array(scores), np.array(labels)


def calibrate_threshold(scores, labels):
    pos_scores = scores[labels == 1]

    if labels.sum() == len(labels):
        threshold = pos_scores.mean() + 2.0 * pos_scores.std()
        print(f"[Threshold] No negatives found. Using mean+2sigma = {threshold:.4f}")
        return float(threshold)

    from sklearn.metrics import roc_auc_score, balanced_accuracy_score

    auc = roc_auc_score(labels, -scores)
    print(f"[Eval] ROC-AUC = {auc * 100:.2f}")

    best_thresh, best_bacc = 0.0, 0.0
    for t in np.percentile(scores, np.arange(5, 95, 2)):
        preds = (scores <= t).astype(int)
        bacc = balanced_accuracy_score(labels, preds)
        if bacc > best_bacc:
            best_bacc, best_thresh = bacc, t

    print(f"[Threshold] Best balanced acc={best_bacc:.4f} at thresh={best_thresh:.4f}")
    return float(best_thresh)
