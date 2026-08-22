"""確信度(表示する%)を検証データに合わせて校正し、重みの隣に保存する。

学習済みの重みは触らない。ロジットを確率に直す部分だけを検証データに当てはめ、
weights/model.pt に対して weights/model.calib.json を作る。推論側
(scripts/predict.py / webapp / scripts/classify_dates.py)はこのファイルが
あれば自動的に読み込み、校正済みの確信度を表示する。

なぜ必要かは src/calibration.py の冒頭を参照。要点だけ書くと、学習時の
pos_weight のせいで生の出力は真の確率より構造的に高く出るため、
「人が見れば明らかに違うのに確信度60%」が起きる。

使い方(学習時と同じ分割の引数を渡すこと):
    python -m scripts.calibrate \
        --data-dir data/processed --labels data/labels.csv \
        --weights weights/model.pt --test-ratio 0.1

    # 校正の効き目をテストセットで確認する(当てはめは常にvalで行う)
    python -m scripts.calibrate ... --report-split test
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from src import calibration as calib
from src.dataset import WeatherMapDataset
from src.labels import LABEL_JA, LABELS
from src.model import load_model
from src.split import SPLIT_MODES, make_splits
from src.train import compute_pos_weight, get_transforms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--era5-features",
        default=None,
        help="学習時に --era5-features を指定した場合は、同じCSVを渡すこと",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2, help="train.pyと同じ値を指定すること")
    parser.add_argument("--test-ratio", type=float, default=0.0, help="train.pyと同じ値を指定すること")
    parser.add_argument("--split-mode", choices=list(SPLIT_MODES), default="temporal",
                        help="train.pyと同じ値を指定すること")
    parser.add_argument("--seed", type=int, default=42, help="train.pyと同じ値を指定すること")
    parser.add_argument("--years", type=int, nargs="+", default=None, help="train.pyと同じ値を指定すること")
    parser.add_argument("--test-year", type=int, default=None, help="train.pyと同じ値を指定すること")
    parser.add_argument("--gap-days", type=int, default=3, help="train.pyと同じ値を指定すること")
    parser.add_argument(
        "--report-split",
        choices=["val", "test"],
        default="val",
        help="校正の効き目を報告する分割。当てはめは常にvalで行う。"
        "valで報告すると当てはめたデータ自身で測ることになり、やや良く見える。"
        "--test-ratio を取ってあるなら test を指定するのが正しい",
    )
    parser.add_argument(
        "--min-positives",
        type=int,
        default=calib.MIN_POSITIVES_FOR_FIT,
        help="検証データの陽性がこの数に満たないラベルは、当てはめずpos_weightの"
        "解析的な補正だけを使う",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="書き出し先。既定は重みと同じ場所の <重み名>.calib.json",
    )
    parser.add_argument("--bins", type=int, default=10, help="信頼度図のビン数")
    parser.add_argument(
        "--pos-weight-cap",
        type=float,
        default=8.0,
        help="重みにpos_weightが記録されていない場合に、学習時の値を学習データから"
        "計算し直すための上限。学習時の --pos-weight-cap と同じ値を指定すること",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_model(args.weights, map_location=device)
    model.to(device).eval()
    print(f"重み: {args.weights}(入力解像度 {meta['image_size']})")

    pos_weight = meta.get("pos_weight")
    if pos_weight is not None:
        pos_weight = np.asarray(pos_weight, dtype="float64")
        print("学習時のpos_weight(重みに記録された値):",
              {l: round(w, 2) for l, w in zip(LABELS, pos_weight)})

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

    splits = make_splits(
        dataset.df,
        mode=args.split_mode,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        test_year=args.test_year,
        gap_days=args.gap_days,
    )
    if not splits["val"]:
        raise SystemExit("valが0件です。--val-ratio / --split-mode を学習時と揃えてください。")

    if pos_weight is None:
        # pos_weightを記録する前に学習された重み。compute_pos_weight() は学習用
        # サブセットのラベル分布だけで決まる決定的な計算なので、同じ分割と同じ
        # --pos-weight-cap を与えれば学習時の値をそのまま再現できる。
        # これが無いと、検証データに陽性が少ないラベルが未補正のまま残り、
        # 補正済みのラベルと確率の意味が揃わなくなる(1位の取り合いが歪む)。
        from torch.utils.data import Subset

        pos_weight = compute_pos_weight(
            Subset(dataset, splits["train"]), num_classes=len(LABELS), cap=args.pos_weight_cap
        ).numpy().astype("float64")
        print(
            f"この重みにはpos_weightが記録されていないため、学習データ"
            f"({len(splits['train'])}件)から再計算しました"
            f"(--pos-weight-cap {args.pos_weight_cap})。"
            "学習時と違う値を使っていた場合は、その値を --pos-weight-cap に指定してください。"
        )
        print("再計算したpos_weight:", {l: round(w, 2) for l, w in zip(LABELS, pos_weight)})

    # 当てはめは必ずvalで行う。テストセットに当てはめて同じテストセットで
    # 報告すると、校正の効き目が実際より良く見える。
    val_logits, val_targets = calib.collect_logits(
        model, dataset, splits["val"], device, args.batch_size
    )
    print(f"校正の当てはめ: val {len(splits['val'])}件")

    calibration = calib.fit(
        val_logits, val_targets, pos_weight=pos_weight, min_positives=args.min_positives
    )
    calibration.source = {
        "weights": str(args.weights),
        "fitted_on": "val",
        "n_fit": int(len(splits["val"])),
        "split_mode": args.split_mode,
        "seed": args.seed,
        "pos_weight_available": pos_weight is not None,
    }

    print("\nラベルごとの校正")
    print(f"  {'ラベル':<20}{'方法':>8}{'陽性数':>7}{'a':>8}{'b':>9}{'しきい値':>10}")
    degenerate = []
    for label in LABELS:
        c = calibration[label]
        print(f"  {LABEL_JA[label]:<20}{c.method:>8}{c.n_positive:>7}"
              f"{c.a:>8.2f}{c.b:>9.2f}{c.threshold:>10.3f}")
        if c.method == "platt" and (c.threshold <= 0.03 or c.threshold >= 0.97):
            degenerate.append(label)
    if degenerate:
        print(
            "\n  警告: 次のラベルはしきい値が端に張り付いています"
            "(事実上『常に陽性』または『常に陰性』):\n    "
            + ", ".join(LABEL_JA[l] for l in degenerate)
            + "\n  検証データの件数が足りないか、そのラベルをモデルが区別できていません。"
            "確信度で絞り込む運用には使えないので、そのラベルは人の確認に回してください。"
        )

    # ---- 効き目の確認 ----
    report_rows = splits[args.report_split]
    if not report_rows:
        raise SystemExit(
            f"--report-split {args.report_split} が0件です。"
            "--test-ratio を学習時と同じ値にしているか確認してください。"
        )
    if args.report_split == "val":
        report_logits, report_targets = val_logits, val_targets
        print("\n注意: 当てはめたvalで報告しています。--test-ratio を取ってあるなら "
              "--report-split test を使ってください。")
    else:
        report_logits, report_targets = calib.collect_logits(
            model, dataset, report_rows, device, args.batch_size
        )

    summary = calib.summarize(report_logits, report_targets, calibration, bins=args.bins)
    calibration.metrics = {
        "report_split": args.report_split,
        "n": summary["n"],
        "top1_accuracy": summary["top1_accuracy"],
        "ece_raw": summary["raw"]["ece"],
        "ece_calibrated": summary["calibrated"]["ece"],
    }

    print(f"\n{'=' * 62}")
    print(f"1位ラベルの確信度と、実際の的中率({args.report_split} {summary['n']}件)")
    print("=" * 62)
    print(f"1位ラベルが正解に含まれていた割合: {summary['top1_accuracy'] * 100:.1f}%")
    print(f"\n[校正前] 平均確信度 {summary['raw']['mean_confidence'] * 100:.1f}% / "
          f"ECE {summary['raw']['ece']:.3f}")
    print(calib.format_reliability(summary["raw"]["reliability"]))
    print(f"\n[校正後] 平均確信度 {summary['calibrated']['mean_confidence'] * 100:.1f}% / "
          f"ECE {summary['calibrated']['ece']:.3f}")
    print(calib.format_reliability(summary["calibrated"]["reliability"]))
    print(
        "\nECEは「表示した%」と「実際に当たった割合」のズレの平均。0に近いほど、"
        "画面の数字がそのまま当たる確率として読める。"
    )

    out_path = Path(args.out) if args.out else calib.default_path(args.weights)
    calibration.save(out_path)
    print(f"\n校正を書き出しました: {out_path}")
    print("これ以降、predict / webapp / classify_dates はこのファイルを自動で読みます。")


if __name__ == "__main__":
    main()
