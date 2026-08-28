"""天気図モデルとERA5格子モデルの確率を混ぜて評価する。学習済みの重みを使う。

両者は得意な気圧配置が違う。天気図は台風(0.764対0.492)や低気圧に強く、格子は
オホーツク海高気圧(0.345対0.131)に強い。移動性高気圧はほぼ同点だった。
片方だけでは、もう片方が得意な型を取りこぼす。

特徴ベクトルを連結する方式は採らない。scripts/ensemble_era5.py に記録した
とおり、ERA5の数値をCNNの特徴に連結したときは次元比のせいで信号が埋もれ、
効果が出なかった。出力する確率だけを混ぜれば、その問題は起きない。

    p = (1 - w) * 天気図モデルの確率 + w * 格子モデルの確率

wと閾値はどちらも検証データで決め、テストデータには一度も触れずに適用する。

使い方:
    python -m scripts.ensemble_chart_grid --data-dir <画像> --labels data/labels.csv \
        --chart-weights runs/loyo_chart_ap --grid-weights runs/loyo_grid_w128 \
        --years 2023 2024 2025
"""

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.dataset import WeatherMapDataset
from src.era5_grid import ERA5GridDataset, compute_grid_stats
from src.blend import WEIGHTS, macro_f1, per_label_weights
from src.evaluate import find_best_thresholds, trivial_macro_f1
from src.labels import LABEL_JA, LABELS
from src.model import load_model
from src.split import make_splits
from src.train import get_transforms
from sklearn.metrics import f1_score


def probabilities(model, dataset, rows, device, batch_size):
    """指定した行に対する予測確率と正解ラベルを返す。"""
    loader = DataLoader(Subset(dataset, rows), batch_size=batch_size)
    probs, targets = [], []
    with torch.no_grad():
        for inputs, _unused, labels, _aux in loader:
            probs.append(torch.sigmoid(model(inputs.to(device))).cpu().numpy())
            targets.append(labels.numpy())
    return np.concatenate(probs), np.concatenate(targets)


def require_aligned(chart_df, grid_df) -> None:
    """2つのデータセットが同じ行を同じ順で持っていることを確かめる。

    どちらもlabels.csvから同じ手順(ラベルのある行だけ・対象年で絞る)で作るので
    揃うはずだが、揃っていなければ天気図の予測と格子の予測を別の日付どうしで
    混ぜることになる。エラーにはならず、静かに無意味な数字が出る。
    """
    if len(chart_df) != len(grid_df):
        raise SystemExit(
            f"行数が違います(天気図{len(chart_df)}件・格子{len(grid_df)}件)。"
            "同じ--labelsと同じ--yearsを指定しているか確認してください。"
        )
    mismatched = (chart_df["parsed_datetime"].values != grid_df["parsed_datetime"].values).sum()
    if mismatched:
        raise SystemExit(f"{mismatched}件で日時が一致しません。行の並びが揃っていません。")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="天気図画像のディレクトリ")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--chart-weights", required=True, help="model_test<年>.pt が入ったディレクトリ")
    parser.add_argument("--grid-weights", required=True, help="同上(ERA5格子モデル)")
    parser.add_argument("--era5-grid-dir", default="data/raw/era5")
    parser.add_argument("--years", type=int, nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--gap-days", type=int, default=3)
    parser.add_argument("--val-mode", default="tail", choices=["spread", "tail"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None, help="結果を書き出すJSON")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = WEIGHTS
    per_fold = []

    for test_year in args.years:
        print(f"\n{'=' * 60}\n=== fold: テスト={test_year}年 ===\n{'=' * 60}")
        paths = {}
        for name, directory in (("chart", args.chart_weights), ("grid", args.grid_weights)):
            path = Path(directory) / f"model_test{test_year}.pt"
            if not path.exists():
                raise SystemExit(
                    f"{path} がありません。\n"
                    "先に scripts/cross_validate.py を同じ --out-dir で実行してください。"
                )
            paths[name] = path

        chart_model, chart_meta = load_model(paths["chart"], map_location=device)
        grid_model, grid_meta = load_model(paths["grid"], map_location=device)
        chart_model.to(device).eval()
        grid_model.to(device).eval()

        chart_ds = WeatherMapDataset(
            args.data_dir, args.labels,
            transform=get_transforms(train=False, image_size=chart_meta["image_size"]),
            years=args.years,
        )
        grid_ds = ERA5GridDataset(
            args.labels, args.era5_grid_dir, years=args.years,
            grid_size=grid_meta["image_size"], augment=False,
        )
        require_aligned(chart_ds.df, grid_ds.df)

        splits = make_splits(
            chart_ds.df, mode="loyo", val_ratio=args.val_ratio, test_ratio=0.0,
            seed=args.seed, test_year=test_year, gap_days=args.gap_days, val_mode=args.val_mode,
        )
        # 格子の正規化は学習に使った行だけから求める(学習時と同じ手順)
        grid_ds.set_stats(*compute_grid_stats(grid_ds, splits["train"]))

        val_chart, val_y = probabilities(chart_model, chart_ds, splits["val"], device, args.batch_size)
        test_chart, test_y = probabilities(chart_model, chart_ds, splits["test"], device, args.batch_size)
        val_grid, _ = probabilities(grid_model, grid_ds, splits["val"], device, args.batch_size)
        test_grid, _ = probabilities(grid_model, grid_ds, splits["test"], device, args.batch_size)

        # 重みwと閾値はどちらも検証データだけで決める
        best = None
        for w in weights:
            blended = (1 - w) * val_chart + w * val_grid
            thresholds = find_best_thresholds(blended, val_y)
            score = macro_f1(blended, val_y, thresholds)
            if best is None or score > best["val_f1"]:
                best = {"w": float(w), "thresholds": thresholds, "val_f1": float(score)}

        test_blend = (1 - best["w"]) * test_chart + best["w"] * test_grid

        # ラベルごとに重みを選ぶ場合(per_label_weightsを参照)
        label_w, label_th = per_label_weights(val_chart, val_grid, val_y, weights)
        test_per_label = (1 - label_w) * test_chart + label_w * test_grid

        scores = {
            "chart": macro_f1(test_chart, test_y, find_best_thresholds(val_chart, val_y)),
            "grid": macro_f1(test_grid, test_y, find_best_thresholds(val_grid, val_y)),
            "blend": macro_f1(test_blend, test_y, best["thresholds"]),
            "per_label_blend": macro_f1(test_per_label, test_y, label_th),
        }
        trivial, _ = trivial_macro_f1(test_y)

        print(f"  検証で選ばれた重み w={best['w']:.2f}"
              f" (0=天気図のみ / 1=格子のみ, 検証macro F1 {best['val_f1']:.3f})")
        print(f"  テスト macro F1 -- 天気図 {scores['chart']:.3f}"
              f" / 格子 {scores['grid']:.3f} / 一律の重みで混合 {scores['blend']:.3f}"
              f" / ラベルごとの重みで混合 {scores['per_label_blend']:.3f}"
              f"   (自明な予測 {trivial:.3f})")
        chosen = ", ".join(
            "{} {:.2f}".format(LABEL_JA[label], label_w[i])
            for i, label in enumerate(LABELS) if val_y[:, i].sum() > 0
        )
        print(f"  ラベルごとの重み: {chosen}")

        per_label = {}
        for i, label in enumerate(LABELS):
            if test_y[:, i].sum() == 0:
                continue
            per_label[label] = {
                key: float(f1_score(test_y[:, i], (probs[:, i] > th[i]).astype(float), zero_division=0))
                for key, probs, th in (
                    ("chart", test_chart, find_best_thresholds(val_chart, val_y)),
                    ("grid", test_grid, find_best_thresholds(val_grid, val_y)),
                    ("blend", test_blend, best["thresholds"]),
                    ("per_label_blend", test_per_label, label_th),
                )
            }
        per_fold.append({
            "test_year": test_year, "w": best["w"], "trivial_macro_f1": trivial,
            **{f"macro_f1_{k}": float(v) for k, v in scores.items()},
            "per_label": per_label,
            "label_weights": {label: float(label_w[i]) for i, label in enumerate(LABELS)},
        })

    print(f"\n{'=' * 72}\nまとめ({len(per_fold)}fold)\n{'=' * 72}")
    for key, name in (("chart", "天気図のみ"), ("grid", "ERA5格子のみ"),
                      ("blend", "混合(一律の重み)"), ("per_label_blend", "混合(ラベルごとの重み)")):
        values = [f[f"macro_f1_{key}"] for f in per_fold]
        base = statistics.mean([f["trivial_macro_f1"] for f in per_fold])
        sd = statistics.stdev(values) if len(values) > 1 else 0.0
        print(f"  {name:<14} {statistics.mean(values):.3f} ± {sd:.3f}"
              f"   自明な予測比 {statistics.mean(values) - base:+.3f}"
              f"   (各fold: {', '.join(f'{v:.3f}' for v in values)})")
    chosen = ", ".join("{}年 {:.2f}".format(f["test_year"], f["w"]) for f in per_fold)
    print(f"\n  選ばれた重み w: {chosen}  (0=天気図のみ / 1=格子のみ)")

    print(f"\n【ラベル別 F1(平均)】")
    print(f"  {'ラベル':<24}{'天気図':>10}{'格子':>10}{'一律':>10}{'ラベル毎':>10}   上積み  重み")
    print("  " + "-" * 78)
    for label in LABELS:
        values = {k: [f["per_label"][label][k] for f in per_fold if label in f["per_label"]]
                  for k in ("chart", "grid", "blend", "per_label_blend")}
        if not values["chart"]:
            continue
        means = {k: statistics.mean(v) for k, v in values.items()}
        gain = means["per_label_blend"] - max(means["chart"], means["grid"])
        weight = statistics.mean([f["label_weights"][label] for f in per_fold])
        print(f"  {LABEL_JA[label]:<24}{means['chart']:>10.3f}{means['grid']:>10.3f}"
              f"{means['blend']:>10.3f}{means['per_label_blend']:>10.3f}   {gain:+.3f}  {weight:.2f}")
    print("\n  ※ 上積みは、天気図・格子の良いほうと「ラベルごとの重みで混合」との差")
    print("  ※ 重みは0=天気図のみ・1=格子のみ。foldごとに検証データで選んだ値の平均")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps({"folds": per_fold}, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\n結果を書き出しました: {args.out}")


if __name__ == "__main__":
    main()
