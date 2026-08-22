import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score

from src import calibration as calib
from src.dataset import WeatherMapDataset
from src.labels import LABELS
from src.model import load_model
from src.split import SPLIT_MODES, make_splits
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
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="一律の判定しきい値。既定は校正ファイルのラベルごとのしきい値"
        "(校正ファイルが無ければ0.5)",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="校正を使わず、生のsigmoid出力で評価する。校正の前後を比べるとき用",
    )
    parser.add_argument(
        "--era5-features",
        default=None,
        help="学習時に --era5-features を指定した場合は、同じCSVを渡すこと",
    )
    parser.add_argument(
        "--optimize-thresholds",
        action="store_true",
        help="一律の--thresholdの代わりに、ラベルごとにF1が最大になる閾値を探索して評価する",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="train.pyと同じ値を指定すること。学習時と同じ分割を再現するために使う",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.0,
        help="train.pyと同じ値を指定すること。--split test を使うなら必須",
    )
    parser.add_argument(
        "--split-mode",
        choices=list(SPLIT_MODES),
        default="temporal",
        help="train.pyと同じ値を指定すること。異なると学習に使った画像が評価に混ざる",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test", "all"],
        default="val",
        help="どの分割を評価するか。val=閾値の調整やモデル比較に使う。"
        "test=最終報告の数値(--optimize-thresholdsと併用すると閾値はvalで決めてtestに適用する)。"
        "all=全件(学習画像を含むため報告には使えない)",
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=None,
        help="train.pyと同じ値を指定すること",
    )
    parser.add_argument(
        "--test-year",
        type=int,
        default=None,
        help="train.pyと同じ値を指定すること(--split-mode loyo のとき必須)",
    )
    parser.add_argument(
        "--gap-days",
        type=int,
        default=3,
        help="train.pyと同じ値を指定すること",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="評価結果をJSONで書き出す先。交差検証の集計に使う",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="train.pyの--seedと同じ値を指定すること。異なると学習時とは別の"
        "分割になり、学習に使った画像を評価に含めてしまう(不当に高いスコアが出る)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 前処理の解像度は重みに同梱されたものを使う(学習時と揃えないと精度が落ちる)。
    # そのため、データセットを作る前に重みを読み込む。
    model, meta = load_model(args.weights, map_location=device)
    model.to(device)
    model.eval()
    print(f"入力解像度: {meta['image_size']}(重みに記録された値)")

    dataset = WeatherMapDataset(
        args.data_dir,
        args.labels,
        transform=get_transforms(train=False, image_size=meta["image_size"]),
        years=args.years,
        features_csv=args.era5_features,
    )
    if meta["num_features"] != len(dataset.feature_cols):
        raise SystemExit(
            f"この重みはERA5特徴量{meta['num_features']}個を前提にしていますが、"
            f"渡されたデータには{len(dataset.feature_cols)}個しかありません。\n"
            "学習時と同じ --era5-features を指定してください。"
        )

    # train.pyと同じ手順で分割を復元する(--split-mode/--val-ratio/--test-ratio/--seedを
    # 学習時と揃えることが前提)。
    splits = make_splits(
        dataset.df,
        mode=args.split_mode,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        test_year=args.test_year,
        gap_days=args.gap_days,
    )

    # 確信度の校正。--raw なら素通し(生のsigmoid出力)になる。
    calibration = (
        calib.Calibration.identity() if args.raw else calib.load_for_weights(args.weights)
    )

    def infer(rows):
        """指定した行に対して推論し、(校正済み確率, 生のロジット, 正解ラベル)を返す。"""
        logits, labels = calib.collect_logits(model, dataset, rows, device, args.batch_size)
        return calibration.probabilities(logits), logits, labels

    if args.split == "all":
        eval_rows = list(range(len(dataset)))
        print("警告: --split all は学習に使った画像を含むため、報告用の数値には使えません")
    else:
        eval_rows = splits[args.split]
    if not eval_rows:
        raise SystemExit(
            f"--split {args.split} の対象が0件です。"
            "--test-ratio を学習時と同じ値にしているか確認してください。"
        )

    all_probs, all_logits, all_labels = infer(eval_rows)

    if args.optimize_thresholds:
        if args.split == "test":
            # 閾値はvalで決めてtestに適用する。testで探索して同じtestで報告すると、
            # 10ラベル×19候補をtestに合わせ込むことになり、数値が楽観方向に偏る。
            tune_probs, _, tune_labels = infer(splits["val"])
            print(f"閾値は val({len(splits['val'])}件)で決定し、test({len(eval_rows)}件)に適用します")
        else:
            tune_probs, tune_labels = all_probs, all_labels
            print(
                "注意: 閾値の探索と結果の報告に同じ分割を使っています。"
                "最終報告には --split test を使ってください"
            )
        thresholds = find_best_thresholds(tune_probs, tune_labels)
        print("best thresholds:", {label: round(t, 2) for label, t in zip(LABELS, thresholds)})
    elif args.threshold is not None:
        thresholds = np.full(len(LABELS), args.threshold)
    elif calibration.is_fitted:
        # 校正時にラベルごとにF1が最大になる値を選んである。校正後の確率は少数
        # ラベルほど小さい値に収まるため、一律0.5だとそれらが全く拾えなくなる。
        thresholds = calibration.thresholds()
        print("しきい値(校正ファイル):", {l: round(t, 3) for l, t in zip(LABELS, thresholds)})
    else:
        thresholds = np.full(len(LABELS), 0.5)

    all_preds = (all_probs > thresholds).astype(float)
    print(classification_report(all_labels, all_preds, target_names=LABELS, zero_division=0))

    report = classification_report(
        all_labels, all_preds, target_names=LABELS, zero_division=0, output_dict=True
    )

    # そのラベルが評価セットに1件も無い場合、sklearnはF1を0として返す。これは
    # 性能が0という意味ではなく「測れていない」という意味なので、macro平均から
    # 外した値も併記する。テストが特定の季節に偏ると実際に起きる。
    supports = {label: int(report[label]["support"]) for label in LABELS}
    evaluable = [l for l in LABELS if supports[l] > 0]
    macro_evaluable = (
        float(np.mean([report[l]["f1-score"] for l in evaluable])) if evaluable else 0.0
    )
    print(
        f"\n評価できたラベル {len(evaluable)}/{len(LABELS)} に限った macro F1: {macro_evaluable:.3f}"
    )
    missing = [l for l in LABELS if supports[l] == 0]
    if missing:
        print(f"  評価セットに出現しなかったラベル: {', '.join(missing)}")

    # ---- 確信度がどれだけ当てになるか ----
    # F1は「どのラベルを選んだか」しか見ないので、表示する%が実態と合っているかは
    # 別に測る必要がある。人が見て明らかに違う判定に高い%が付く問題は、ここに出る。
    summary = calib.summarize(all_logits, all_labels, calibration)
    print(f"\n{'=' * 62}")
    print("1位ラベルの確信度と、実際の的中率")
    print("=" * 62)
    print(f"1位ラベルが正解に含まれていた割合: {summary['top1_accuracy'] * 100:.1f}%")
    shown = "calibrated" if calibration.is_fitted else "raw"
    print(f"平均確信度: {summary[shown]['mean_confidence'] * 100:.1f}% / "
          f"ECE {summary[shown]['ece']:.3f}"
          f"{'' if calibration.is_fitted else '(未校正)'}")
    print(calib.format_reliability(summary[shown]["reliability"]))
    if calibration.is_fitted:
        print(f"\n参考: 校正前は平均確信度 {summary['raw']['mean_confidence'] * 100:.1f}% / "
              f"ECE {summary['raw']['ece']:.3f} でした。")
    else:
        print("\n確信度が未校正です。python -m scripts.calibrate で校正すると、"
              "表示%が実際に当たる割合に近づきます。")

    if args.json_out:
        import json

        payload = {
            "weights": args.weights,
            "split": args.split,
            "split_mode": args.split_mode,
            "test_year": args.test_year,
            "seed": args.seed,
            "n_eval": len(eval_rows),
            "top1_accuracy": summary["top1_accuracy"],
            "ece": summary[shown]["ece"],
            "ece_raw": summary["raw"]["ece"],
            "calibrated": calibration.is_fitted,
            "macro_f1_all_labels": report["macro avg"]["f1-score"],
            "macro_f1_evaluable": macro_evaluable,
            "micro_f1": report["micro avg"]["f1-score"],
            "weighted_f1": report["weighted avg"]["f1-score"],
            "per_label": {
                label: {
                    "f1": report[label]["f1-score"],
                    "precision": report[label]["precision"],
                    "recall": report[label]["recall"],
                    "support": supports[label],
                }
                for label in LABELS
            },
        }
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"結果を書き出しました: {args.json_out}")


if __name__ == "__main__":
    main()
