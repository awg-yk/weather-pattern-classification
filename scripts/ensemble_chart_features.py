r"""天気図CNNと、検出した特徴量の木モデルを、確率の平均で組み合わせる。

狙い
----
CNNは強いが、**H や L の記号を数えていない**。Grad-CAMで見ると、
オホーツク海高気圧を正解している58枚でもオホーツク海を見ている割合は
8.6%しかなかった(`docs/2026-08-25-attention-regions.md`)。低気圧しか
無い天気図で移動性高気圧を出す、といった誤りがここから出る。

一方 `scripts/build_features.py` は H と L の位置と個数を明示的に数えている。
全体では CNN に及ばない(0.408 対 0.641)が、**誤り方が違う**。

前例
----
天気図 + ERA5格子で同じことをしたときの結果
(`docs/2026-08-21-chart-vs-era5-grid.md`):

  - 全体 +0.010(3foldとも正)
  - **いちばん伸びたのは移動性高気圧 0.726 -> 0.748(+0.023)**
  - 伸びたのは「両者とも中程度で、違う誤りをしているラベル」だった
  - オホーツク海高気圧は混ぜると格子単独より下がった。**混ぜないほうが
    良いラベルもある**ので、重みはラベルごとに決める

学習済みのCNNをそのまま使うので再学習は不要。木のほうは数秒で学習し直す。
重みと閾値はどちらも検証データで決め、テストには一度も触れずに適用する。

**特徴量をCNNの特徴ベクトルに連結する方式は採らない。**
`scripts/ensemble_era5.py` に記録があるとおり、画像側1280次元に対して
数十次元では次元比で信号が埋もれた。確率だけを混ぜればこれに左右されない。

使い方:
    python -m scripts.ensemble_chart_features --data-dir <画像> `
        --labels data/labels_v2.csv --features data/features.csv `
        --chart-weights runs/cv_baseline --years 2023 2024 2025 `
        --out runs/cv_blend
"""

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Subset

from src.blend import (
    WEIGHTS,
    align_by_filename,
    blend,
    inherit_split_settings,
    per_label_weights,
)
from src.dataset import WeatherMapDataset
from src.labels import LABEL_JA, LABELS
from src.metrics import find_best_thresholds, trivial_macro_f1
from src.model import load_model
from src.split import make_splits
from src.train import get_transforms

from scripts.cv_features import KEY_COLUMNS, fit_predict

def chart_probabilities(model, dataset, rows, device, batch_size):
    """指定した行に対するCNNの予測確率と正解ラベルを返す。"""
    loader = DataLoader(Subset(dataset, rows), batch_size=batch_size)
    probs, targets = [], []
    with torch.no_grad():
        for inputs, _unused, labels, _aux in loader:
            probs.append(torch.sigmoid(model(inputs.to(device))).cpu().numpy())
            targets.append(labels.numpy())
    return np.concatenate(probs), np.concatenate(targets)


def targets_of(df: pd.DataFrame) -> np.ndarray:
    y = np.zeros((len(df), len(LABELS)), dtype=np.int64)
    for row, text in enumerate(df["label"].fillna("")):
        for name in str(text).split("|"):
            if name in LABELS:
                y[row, LABELS.index(name)] = 1
    return y


def fold_report(y_true, preds, test_year) -> dict:
    """`scripts/cross_validate.py` と同じ形の1foldぶんの結果。"""
    report = classification_report(y_true, preds, target_names=LABELS,
                                   output_dict=True, zero_division=0)
    supports = {label: int(y_true[:, i].sum()) for i, label in enumerate(LABELS)}
    evaluable = [label for label in LABELS if supports[label] > 0]
    macro_evaluable = (float(np.mean([report[label]["f1-score"] for label in evaluable]))
                       if evaluable else 0.0)
    trivial, trivial_per_label = trivial_macro_f1(y_true)
    return {
        "test_year": test_year,
        "n_eval": int(len(y_true)),
        "macro_f1_all_labels": report["macro avg"]["f1-score"],
        "macro_f1_evaluable": macro_evaluable,
        "trivial_macro_f1": trivial,
        "macro_f1_over_trivial": macro_evaluable - trivial,
        "micro_f1": report["micro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "per_label": {
            label: {
                "f1": report[label]["f1-score"],
                "precision": report[label]["precision"],
                "recall": report[label]["recall"],
                "support": supports[label],
                "trivial_f1": float(trivial_per_label[LABELS.index(label)]),
            }
            for label in LABELS
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", required=True, help="天気図画像のディレクトリ")
    parser.add_argument("--labels", default="data/labels_v2.csv")
    parser.add_argument("--features", required=True,
                        help="scripts/build_features.py の出力")
    parser.add_argument("--chart-weights", required=True,
                        help="model_test<年>.pt が入ったディレクトリ(runs/cv_baseline など)")
    parser.add_argument("--years", type=int, nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    # 既定は None。重みの summary.json から引き継ぐため(settings_from_weights)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--gap-days", type=int, default=None)
    parser.add_argument("--val-mode", default=None, choices=["spread", "tail"])
    parser.add_argument("--model", default="hgb", choices=("hgb", "xgboost"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", required=True, help="summary.json を書き出すディレクトリ")
    args = parser.parse_args()
    inherit_split_settings(Path(args.chart_weights), args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features_df = pd.read_csv(args.features)
    feature_columns = [c for c in features_df.columns if c not in KEY_COLUMNS]
    print(f"特徴量 {len(feature_columns)}個、モデル {args.model}")

    folds = []
    for test_year in args.years:
        path = Path(args.chart_weights) / f"model_test{test_year}.pt"
        if not path.exists():
            raise SystemExit(
                f"{path} がありません。\n"
                f"先に scripts/cross_validate.py を --out-dir {args.chart_weights} "
                "で実行してください。"
            )
        model, meta = load_model(path, map_location=device)
        model.to(device).eval()

        dataset = WeatherMapDataset(
            args.data_dir, args.labels,
            transform=get_transforms(train=False, image_size=meta["image_size"]),
            years=args.years,
        )
        # 特徴量は filename で天気図の行順に並べ替える。行番号で結合すると
        # ずれても動いて、別の日付の特徴量を混ぜることになる
        X = align_by_filename(dataset.df, features_df, feature_columns)
        y_all = targets_of(dataset.df)

        splits = make_splits(
            dataset.df, mode="loyo", val_ratio=args.val_ratio, test_ratio=0.0,
            seed=args.seed, test_year=test_year, gap_days=args.gap_days,
            val_mode=args.val_mode,
        )
        print(f"\n{'=' * 68}\n=== fold: テスト={test_year}年 "
              f"(学習{len(splits['train'])} / 検証{len(splits['val'])} / "
              f"テスト{len(splits['test'])}) ===")

        val_chart, val_y = chart_probabilities(model, dataset, splits["val"],
                                               device, args.batch_size)
        test_chart, test_y = chart_probabilities(model, dataset, splits["test"],
                                                 device, args.batch_size)
        # 木は学習データだけで当てはめる。CNNの学習に使った年と同じ範囲
        train = np.asarray(splits["train"])
        val_feat = fit_predict(X[train], y_all[train], X[np.asarray(splits["val"])],
                               args.model, args.seed)
        test_feat = fit_predict(X[train], y_all[train], X[np.asarray(splits["test"])],
                                args.model, args.seed)

        label_w, label_th = per_label_weights(val_chart, val_feat, val_y, WEIGHTS)
        test_blend = blend(test_chart, test_feat, label_w)

        chart_th = find_best_thresholds(val_chart, val_y)
        feat_th = find_best_thresholds(val_feat, val_y)
        # **3つとも同じ定義で測る。**混合だけ「評価できたラベルのみ」、単独は
        # 「全ラベル」にすると、テストに1件も出ないラベルがあったときに混合だけ
        # 得をする。cross_validate.py が報告しているのは評価できたラベルのみ
        reports = {
            "chart": fold_report(test_y, (test_chart > chart_th).astype(int), test_year),
            "features": fold_report(test_y, (test_feat > feat_th).astype(int), test_year),
            "blend": fold_report(test_y, (test_blend > label_th).astype(int), test_year),
        }
        scores = {k: v["macro_f1_evaluable"] for k, v in reports.items()}
        trivial = reports["blend"]["trivial_macro_f1"]
        print(f"  macro F1 -- 天気図 {scores['chart']:.3f} / "
              f"特徴量 {scores['features']:.3f} / "
              f"混合 {scores['blend']:.3f}   (自明な予測 {trivial:.3f})")
        chosen = ", ".join(
            "{} {:.2f}".format(LABEL_JA[label], label_w[i])
            for i, label in enumerate(LABELS) if val_y[:, i].sum() > 0
        )
        print(f"  混ぜた割合(0=天気図のみ / 1=特徴量のみ): {chosen}")

        result = reports["blend"]
        result["label_weights"] = {label: float(label_w[i])
                                   for i, label in enumerate(LABELS)}
        result["macro_f1_chart_only"] = scores["chart"]
        result["macro_f1_features_only"] = scores["features"]
        result["per_label_chart_only"] = {
            label: reports["chart"]["per_label"][label]["f1"] for label in LABELS
        }
        folds.append(result)

    print(f"\n{'=' * 72}\n混合の結果({len(folds)}fold)\n{'=' * 72}")
    for key, name in (("macro_f1_chart_only", "天気図のみ"),
                      ("macro_f1_features_only", "特徴量のみ"),
                      ("macro_f1_evaluable", "混合(ラベルごとの重み)")):
        values = [f[key] for f in folds]
        sd = statistics.stdev(values) if len(values) > 1 else 0.0
        print(f"  {name:<22} {statistics.mean(values):.3f} ± {sd:.3f}"
              f"   (各fold: {', '.join(f'{v:.3f}' for v in values)})")

    print(f"\n【ラベル別 F1(平均)】")
    print(f"  {'ラベル':<24}{'天気図':>9}{'混合':>9}{'差':>9}   混ぜた割合")
    print("  " + "-" * 66)
    for i, label in enumerate(LABELS):
        chart = [f["per_label_chart_only"][label] for f in folds]
        mixed = [f["per_label"][label]["f1"] for f in folds
                 if f["per_label"][label]["support"] > 0]
        if not mixed:
            continue
        w = statistics.mean([f["label_weights"][label] for f in folds])
        diff = statistics.mean(mixed) - statistics.mean(chart)
        mark = " ★" if diff > 0.01 else ""
        print(f"  {LABEL_JA[label]:<24}{statistics.mean(chart):>9.3f}"
              f"{statistics.mean(mixed):>9.3f}{diff:>+9.3f}   {w:.2f}{mark}")
    print("\n  「混ぜた割合」が0に近いラベルは、検証データが「混ぜないほうがよい」と"
          "判断したもの。\n  そこは天気図のみと同じ結果になる。")

    summary = {
        "method": "blend(chart + features)",
        "config": {
            "features": str(args.features), "chart_weights": str(args.chart_weights),
            "model": args.model, "val_mode": args.val_mode, "seed": args.seed,
        },
        "folds": folds,
        "mean": {
            key: statistics.mean([f[key] for f in folds])
            for key in ("macro_f1_evaluable", "macro_f1_all_labels", "micro_f1",
                        "weighted_f1", "trivial_macro_f1", "macro_f1_over_trivial")
        },
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nまとめを書き出しました: {out_dir / 'summary.json'}")
    print("既存CNNと並べて見るには:")
    print(f"  python -m scripts.compare_runs {args.chart_weights} {args.out}")


if __name__ == "__main__":
    main()
