import argparse

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader

from src.dataset import WeatherMapDataset
from src.labels import LABELS
from src.model import build_model
from src.train import get_transforms


def find_best_thresholds(probs: np.ndarray, labels: np.ndarray, steps: int = 19) -> np.ndarray:
    """ラベルごとにF1を最大化する閾値を探索する。

    マルチラベルではラベルごとに陽性/陰性の出現頻度が大きく異なり、一律0.5では
    少数派ラベルを取りこぼしやすい。0.05刻みでF1が最大になる点をラベルごとに選ぶ。
    """
    candidates = np.linspace(0.05, 0.95, steps)
    best_thresholds = np.full(labels.shape[1], 0.5)
    for i in range(labels.shape[1]):
        col_labels = labels[:, i]
        if col_labels.sum() == 0:
            continue
        best_f1 = -1.0
        for t in candidates:
            preds = (probs[:, i] > t).astype(float)
            f1 = f1_score(col_labels, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresholds[i] = t
    return best_thresholds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--optimize-thresholds",
        action="store_true",
        help="一律の--thresholdの代わりに、ラベルごとにF1が最大になる閾値を探索して評価する",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = WeatherMapDataset(args.data_dir, args.labels, transform=get_transforms(train=False))
    loader = DataLoader(dataset, batch_size=args.batch_size)

    model = build_model(num_classes=len(LABELS), pretrained=False).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu()
            all_probs.append(probs)
            all_labels.append(labels)

    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()

    if args.optimize_thresholds:
        thresholds = find_best_thresholds(all_probs, all_labels)
        print("best thresholds:", {label: round(t, 2) for label, t in zip(LABELS, thresholds)})
    else:
        thresholds = np.full(len(LABELS), args.threshold)

    all_preds = (all_probs > thresholds).astype(float)
    print(classification_report(all_labels, all_preds, target_names=LABELS, zero_division=0))


if __name__ == "__main__":
    main()
