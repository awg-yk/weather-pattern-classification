import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader, Subset

from src import calibration as calib
from src.dataset import WeatherMapDataset
from src.era5_grid import ERA5GridDataset, compute_grid_stats
from src.labels import LABELS
from src.model import load_model
from src.split import SPLIT_MODES, VAL_MODES, make_splits
from src.train import get_transforms



def trivial_macro_f1(labels: np.ndarray) -> tuple:
    """「全部を陽性と予測する」だけで得られるmacro F1と、そのラベル別の値。

    出現率pのラベルは、常に陽性と答えるだけでF1 = 2p/(1+p) を得る。頻出ラベルでは
    これが0.5を超えるため、macro F1の絶対値は学習の成果を表さない。この下駄を
    引いて初めて、モデルが何を足したのかが分かる。

    実際、ERA5格子を224で学習した回はmacro F1 0.250で、この基準(約0.29)を
    下回っていた -- 数字だけ見ていると「低いが学習はできている」と読めてしまう。
    """
    prevalence = labels.mean(axis=0)
    per_label = np.where(prevalence > 0, 2 * prevalence / (1 + prevalence), 0.0)
    evaluable = prevalence > 0
    return (float(per_label[evaluable].mean()) if evaluable.any() else 0.0), per_label


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
    parser.add_argument("--data-dir", default=None, help="天気図画像のディレクトリ(chartモードで必須)")
    parser.add_argument("--labels", required=True)
    parser.add_argument(
        "--input-mode",
        default="chart",
        choices=["chart", "era5-grid"],
        help="train.pyと同じ値を指定すること",
    )
    parser.add_argument(
        "--era5-grid-dir",
        default="data/raw/era5",
        help="--input-mode era5-grid のときの、ERA5 netCDFの置き場所",
    )
    parser.add_argument("--weights", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--calibrated",
        action="store_true",
        help="確信度の校正(<重み名>.calib.json)を適用して評価する。既定では適用しない。"
        "校正は単調変換なので、--optimize-thresholds と併用するかぎり F1 も macro AP も"
        "変わらない(過去の報告値との比較を壊さないため、既定を素の出力のままにしてある)",
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
        "--val-mode",
        default="tail",
        choices=VAL_MODES,
        help="学習時と同じ値を指定すること。違うと別の分割を復元してしまう",
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

    if args.input_mode == "chart" and not args.data_dir:
        raise SystemExit("chartモードでは --data-dir が必要です")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 前処理の解像度は重みに同梱されたものを使う(学習時と揃えないと精度が落ちる)。
    # そのため、データセットを作る前に重みを読み込む。
    model, meta = load_model(args.weights, map_location=device)
    model.to(device)
    model.eval()
    print(f"入力解像度: {meta['image_size']}(重みに記録された値)")

    if args.input_mode == "era5-grid":
        dataset = ERA5GridDataset(
            args.labels, args.era5_grid_dir, years=args.years,
            grid_size=meta["image_size"], augment=False,
        )
    else:
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
        val_mode=args.val_mode,
    )

    if args.input_mode == "era5-grid":
        # train.pyと同じ手順(学習用サブセットだけから正規化の統計を求める)を
        # ここで再現する。統計はチェックポイントに埋め込まれていないため、
        # train.pyと同じ分割になっている前提でこの場で計算し直す。
        mean, std = compute_grid_stats(dataset, splits["train"])
        dataset.set_stats(mean, std)

    def infer(rows):
        """指定した行に対して推論し、(確率, 正解ラベル)を返す。"""
        loader = DataLoader(Subset(dataset, rows), batch_size=args.batch_size)
        probs_list, labels_list = [], []
        with torch.no_grad():
            for images, features, labels in loader:
                images = images.to(device)
                outputs = (
                    model(images, features.to(device)) if features.numel() else model(images)
                )
                probs_list.append(torch.sigmoid(outputs).cpu())
                labels_list.append(labels)
        return torch.cat(probs_list).numpy(), torch.cat(labels_list).numpy()

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

    all_probs, all_labels = infer(eval_rows)

    # 校正は既定では「測るだけ」で、判定には使わない。--calibrated を付けたときだけ
    # 確率としても適用する。過去の報告値と地続きにしておくため。
    calibration = calib.load_for_weights_cli(args.weights, verbose=args.calibrated)
    raw_probs = all_probs
    if args.calibrated:
        if not calibration.is_fitted:
            raise SystemExit(
                "--calibrated が指定されましたが、校正ファイルがありません。"
                "python -m scripts.calibrate で作成してください。"
            )
        all_probs = calibration.from_probabilities(all_probs)

    if args.optimize_thresholds:
        if args.split == "test":
            # 閾値はvalで決めてtestに適用する。testで探索して同じtestで報告すると、
            # 10ラベル×19候補をtestに合わせ込むことになり、数値が楽観方向に偏る。
            tune_probs, tune_labels = infer(splits["val"])
            if args.calibrated:
                tune_probs = calibration.from_probabilities(tune_probs)
            print(f"閾値は val({len(splits['val'])}件)で決定し、test({len(eval_rows)}件)に適用します")
        else:
            tune_probs, tune_labels = all_probs, all_labels
            print(
                "注意: 閾値の探索と結果の報告に同じ分割を使っています。"
                "最終報告には --split test を使ってください"
            )
        thresholds = find_best_thresholds(tune_probs, tune_labels)
        print("best thresholds:", {label: round(t, 2) for label, t in zip(LABELS, thresholds)})
    elif args.calibrated:
        # 校正後の確率は少数ラベルほど小さい値に収まるため、一律0.5では拾えない
        thresholds = calibration.thresholds()
        print("しきい値(校正ファイル):", {l: round(t, 3) for l, t in zip(LABELS, thresholds)})
    else:
        thresholds = np.full(len(LABELS), args.threshold)

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

    trivial, trivial_per_label = trivial_macro_f1(all_labels)
    print(
        f"\n全部を陽性と答えるだけで得られる macro F1: {trivial:.3f}"
        f"  → このモデルの上積み: {macro_evaluable - trivial:+.3f}"
    )
    if macro_evaluable <= trivial:
        print("  警告: 自明な予測を上回っていません。学習が機能していない可能性があります")

    # ---- 確信度がどれだけ当てになるか ----
    # F1は「どのラベルを選んだか」しか見ないので、表示する%が実態と合っているかは
    # 別に測る必要がある。人が見て明らかに違う判定に高い%が付く問題は、ここに出る。
    shown_probs = all_probs if args.calibrated else raw_probs
    conf, hit = calib.top1_confidence_and_correctness(shown_probs, all_labels)
    ece = calib.expected_calibration_error(conf, hit)
    print(f"\n{'=' * 62}")
    print("1位ラベルの確信度と、実際の的中率" + ("(校正後)" if args.calibrated else "(未校正)"))
    print("=" * 62)
    print(f"1位ラベルが正解に含まれていた割合: {hit.mean() * 100:.1f}%")
    print(f"平均確信度: {conf.mean() * 100:.1f}% / ECE {ece:.3f}")
    print(calib.format_reliability(calib.reliability_table(conf, hit)))
    if not args.calibrated and calibration.is_fitted:
        cal_conf, cal_hit = calib.top1_confidence_and_correctness(
            calibration.from_probabilities(raw_probs), all_labels
        )
        print(f"\n参考: 校正を適用すると 平均確信度 {cal_conf.mean() * 100:.1f}% / "
              f"ECE {calib.expected_calibration_error(cal_conf, cal_hit):.3f} になります"
              "(--calibrated で適用)。")
    elif not calibration.is_fitted:
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
            "top1_accuracy": float(hit.mean()),
            "ece_top1": ece,
            "calibrated": bool(args.calibrated),
            "macro_f1_all_labels": report["macro avg"]["f1-score"],
            "macro_f1_evaluable": macro_evaluable,
            # 学習の成果は絶対値ではなくこの基準との差で読む(trivial_macro_f1を参照)
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
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"結果を書き出しました: {args.json_out}")


if __name__ == "__main__":
    main()
