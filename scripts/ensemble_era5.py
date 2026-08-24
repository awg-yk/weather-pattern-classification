"""CNNとERA5のロジスティック回帰を、確率の平均で組み合わせて評価する。

ERA5の数値をCNNの特徴ベクトルに連結する方式(FeatureFusion)は効果がなかった。
画像側1280次元に対しERA5側17次元と次元比が大きく、ERA5の信号が埋もれたためと
考えられる。そこで両者を別々のモデルとして動かし、出力する確率だけを混ぜる。
この方式ならERA5側の寄与が次元比に左右されない。

    p = (1 - w) * CNNの確率 + w * ロジスティック回帰の確率

wを0(CNNのみ)から1(ERA5のみ)まで振り、検証データで最も良いwを選ぶ。
学習済みのCNNをそのまま使うので再学習は不要。

閾値と重みwはどちらも検証データで決め、テストデータには一度も触れずに適用する。

使い方:
    python -m scripts.ensemble_era5 --data-dir <画像> --labels data/labels.csv \
        --features data/era5_features.csv --weights-dir runs/loyo_coordconv \
        --years 2023 2024 2025
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset

from src.dataset import WeatherMapDataset
from src.evaluate import find_best_thresholds
from src.labels import LABEL_JA, LABELS
from src.model import load_model
from src.split import make_splits
from src.train import get_transforms


def cnn_probabilities(model, dataset, rows, device, batch_size):
    """指定した行に対するCNNの予測確率と正解ラベルを返す。"""
    loader = DataLoader(Subset(dataset, rows), batch_size=batch_size)
    probs, targets = [], []
    with torch.no_grad():
        for images, _features, labels in loader:
            probs.append(torch.sigmoid(model(images.to(device))).cpu().numpy())
            targets.append(labels.numpy())
    return np.concatenate(probs), np.concatenate(targets)


def logistic_probabilities(features, targets, train_rows, apply_rows):
    """ERA5特徴量のロジスティック回帰を学習し、指定行の確率を返す。

    ラベルごとに独立した2値分類器を作る(マルチラベルなので相互排他ではない)。
    学習用の行に陽性が1件も無いラベルは学習できないため、確率0を返す。
    """
    out = np.zeros((len(apply_rows), len(LABELS)))
    for i in range(len(LABELS)):
        y = targets[train_rows, i]
        if y.sum() < 2 or y.sum() == len(y):
            continue
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        ).fit(features[train_rows], y)
        out[:, i] = model.predict_proba(features[apply_rows])[:, 1]
    return out


def macro_f1(probs, targets, thresholds):
    preds = (probs > thresholds).astype(float)
    return f1_score(targets, preds, average="macro", zero_division=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--features", required=True, help="scripts/era5_features.py の出力")
    parser.add_argument("--weights-dir", required=True, help="model_test<年>.pt が入ったディレクトリ")
    parser.add_argument("--years", type=int, nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--gap-days", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None, help="結果を書き出すJSON")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_dir = Path(args.weights_dir)
    grid = np.round(np.arange(0.0, 1.01, 0.05), 2)

    per_fold = []
    for test_year in args.years:
        weights_path = weights_dir / f"model_test{test_year}.pt"
        if not weights_path.exists():
            raise SystemExit(
                f"{weights_path} がありません。\n"
                "先に scripts/cross_validate.py を同じ --out-dir で実行してください。"
            )
        model, meta = load_model(weights_path, map_location=device)
        model.to(device).eval()

        dataset = WeatherMapDataset(
            args.data_dir,
            args.labels,
            transform=get_transforms(train=False, image_size=meta["image_size"]),
            years=args.years,
            features_csv=args.features,
        )
        splits = make_splits(
            dataset.df, mode="loyo", val_ratio=args.val_ratio, test_ratio=0.0,
            seed=args.seed, test_year=test_year, gap_days=args.gap_days,
        )
        features = dataset.features.numpy()
        all_targets = np.zeros((len(dataset), len(LABELS)))
        for row_index, parsed in enumerate(dataset.df["parsed_labels"]):
            for label in parsed:
                all_targets[row_index, LABELS.index(label)] = 1.0

        val_cnn, val_y = cnn_probabilities(model, dataset, splits["val"], device, args.batch_size)
        test_cnn, test_y = cnn_probabilities(model, dataset, splits["test"], device, args.batch_size)
        val_lr = logistic_probabilities(features, all_targets, splits["train"], splits["val"])
        test_lr = logistic_probabilities(features, all_targets, splits["train"], splits["test"])

        # 重みwと閾値はどちらも検証データだけで決める
        best = None
        for w in grid:
            blended = (1 - w) * val_cnn + w * val_lr
            thresholds = find_best_thresholds(blended, val_y)
            score = macro_f1(blended, val_y, thresholds)
            if best is None or score > best["val_f1"]:
                best = {"w": float(w), "thresholds": thresholds, "val_f1": float(score)}

        test_blend = (1 - best["w"]) * test_cnn + best["w"] * test_lr
        cnn_only_th = find_best_thresholds(val_cnn, val_y)
        lr_only_th = find_best_thresholds(val_lr, val_y)

        preds = (test_blend > best["thresholds"]).astype(float)
        fold = {
            "test_year": test_year,
            "w": best["w"],
            "cnn_only": macro_f1(test_cnn, test_y, cnn_only_th),
            "era5_only": macro_f1(test_lr, test_y, lr_only_th),
            "ensemble": macro_f1(test_blend, test_y, best["thresholds"]),
            "per_label": {
                label: {
                    "cnn": f1_score(test_y[:, i], (test_cnn[:, i] > cnn_only_th[i]), zero_division=0),
                    "ensemble": f1_score(test_y[:, i], preds[:, i], zero_division=0),
                    "support": int(test_y[:, i].sum()),
                }
                for i, label in enumerate(LABELS)
            },
        }
        per_fold.append(fold)
        print(f"テスト={test_year}年: 選ばれた重み w={fold['w']:.2f}  "
              f"CNNのみ {fold['cnn_only']:.3f} / ERA5のみ {fold['era5_only']:.3f} / "
              f"平均 {fold['ensemble']:.3f}")

    print(f"\n{'=' * 68}\n確率平均によるアンサンブル({len(per_fold)}fold)\n{'=' * 68}\n")
    for key, name in (("cnn_only", "CNNのみ"), ("era5_only", "ERA5のみ"), ("ensemble", "確率の平均")):
        values = np.array([f[key] for f in per_fold])
        print(f"  {name:<14}{values.mean():.3f} ± {values.std():.3f}   "
              f"(各fold: {', '.join(f'{v:.3f}' for v in values)})")
    print(f"\n  選ばれた重み w: {[f['w'] for f in per_fold]}  (0=CNNのみ, 1=ERA5のみ)")

    print(f"\n【ラベル別 F1】\n  {'ラベル':<22}{'CNNのみ':>10}{'平均':>10}{'差':>9}{'件数':>7}")
    print("  " + "-" * 58)
    for label in LABELS:
        cnn = np.mean([f["per_label"][label]["cnn"] for f in per_fold])
        ens = np.mean([f["per_label"][label]["ensemble"] for f in per_fold])
        support = sum(f["per_label"][label]["support"] for f in per_fold)
        print(f"  {LABEL_JA[label]:<22}{cnn:>10.3f}{ens:>10.3f}{ens - cnn:>+9.3f}{support:>7}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in f.items()}
                   for f in per_fold]
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n書き出しました: {out_path}")


if __name__ == "__main__":
    main()
