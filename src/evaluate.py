import argparse

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader, random_split

from src.dataset import WeatherMapDataset
from src.labels import LABELS
from src.model import build_model, load_checkpoint
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
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="train.pyと同じ値を指定すること。指定した重みの学習で使われたのと"
        "同じtrain/val分割を再現し、検証セット(学習に使われなかった分)だけを"
        "評価するために使う",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="train.pyの--seedと同じ値を指定すること。異なると学習時とは別の"
        "分割になり、学習に使った画像を評価に含めてしまう(不当に高いスコアが出る)",
    )
    parser.add_argument(
        "--full-dataset",
        action="store_true",
        help="train/val分割をせず、labels.csvの全件を評価する(train-limitで"
        "比較する場合は使わないこと。学習に使った画像が評価に混ざり、"
        "件数が多いモデルほど不当に有利になる)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 前処理の解像度は重みに同梱されたものを使う(学習時と揃えないと精度が落ちる)。
    # そのため、データセットを作る前に重みを読み込む。
    model = build_model(num_classes=len(LABELS), pretrained=False).to(device)
    meta = load_checkpoint(args.weights, model, map_location=device)
    model.eval()
    print(f"入力解像度: {meta['image_size']}(重みに記録された値)")

    dataset = WeatherMapDataset(
        args.data_dir, args.labels, transform=get_transforms(train=False, image_size=meta["image_size"])
    )

    if args.full_dataset:
        eval_dataset = dataset
    else:
        # train.pyのrandom_splitと同じseed・val-ratioで分割を再現する。
        # --train-limitで学習用件数だけを絞っていても、train/val分割自体は
        # train-limit適用前のfull_dataset全体に対して行われる(train.py参照)
        # ので、ここでも同じ手順(seed固定のrandom_split)を再現すればよい。
        generator = torch.Generator().manual_seed(args.seed)
        val_size = int(len(dataset) * args.val_ratio)
        train_size = len(dataset) - val_size
        _, eval_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    loader = DataLoader(eval_dataset, batch_size=args.batch_size)

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
