"""leave-one-year-out 交差検証を実行し、結果を平均±標準偏差でまとめる。

年を1つずつテストに回して学習・評価を繰り返す。テストが必ず通年になるため、
台風のように特定の季節にしか現れないラベルも必ず評価対象に入る。また、
foldごとに推定値が得られるので、ラベルごとのばらつき(標準偏差)を出せる。

各foldは src/train.py と src/evaluate.py を別プロセスで呼ぶ。学習の途中経過は
そのまま画面に出るので、進捗が分かる。

使い方:
    python -m scripts.cross_validate \
        --data-dir data/processed --labels data/labels.csv \
        --years 2023 2024 2025 --out-dir runs/loyo
"""

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

from src.labels import LABEL_JA, LABELS


def run(cmd: list) -> None:
    print("\n$ " + " ".join(str(c) for c in cmd), flush=True)
    result = subprocess.run([sys.executable, "-m"] + [str(c) for c in cmd])
    if result.returncode != 0:
        raise SystemExit(f"失敗しました(終了コード {result.returncode}): {' '.join(map(str, cmd))}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None, help="天気図画像のディレクトリ(chartモードで必須)")
    parser.add_argument("--labels", required=True)
    parser.add_argument(
        "--input-mode", default="chart", choices=["chart", "era5-grid"],
        help="chart(既定)=天気図の画像。era5-grid=ERA5の格子を圧縮せず直接入力する",
    )
    parser.add_argument("--era5-grid-dir", default="data/raw/era5")
    parser.add_argument("--grid-size", type=int, default=128)
    parser.add_argument("--years", type=int, nargs="+", required=True, help="対象年(この中から1年ずつテストに回す)")
    parser.add_argument("--out-dir", default="runs/loyo", help="重みと結果JSONの保存先")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--coordconv", action="store_true")
    parser.add_argument("--era5-features", default=None)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--gap-days", type=int, default=3)
    parser.add_argument("--val-mode", default="tail", choices=["spread", "tail"])
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="結果JSONが既にあるfoldは飛ばす(途中から再開したいとき)",
    )
    args = parser.parse_args()

    if args.input_mode == "chart" and not args.data_dir:
        raise SystemExit("chartモードでは --data-dir が必要です")
    if args.input_mode == "era5-grid" and args.era5_features:
        raise SystemExit("--input-mode era5-grid と --era5-features は併用できません")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    common = [
        "--labels", args.labels,
        "--years", *args.years,
        "--split-mode", "loyo",
        "--val-ratio", args.val_ratio,
        "--gap-days", args.gap_days,
        "--val-mode", args.val_mode,
        "--seed", args.seed,
        "--input-mode", args.input_mode,
    ]
    if args.input_mode == "era5-grid":
        # --grid-sizeはtrain.pyだけが受け取る(evaluate.pyはチェックポイントに
        # 保存された値を使うため、CLI引数を持たない)。
        common += ["--era5-grid-dir", args.era5_grid_dir]
    else:
        common += ["--data-dir", args.data_dir]
    if args.era5_features:
        common += ["--era5-features", args.era5_features]

    results = []
    for test_year in args.years:
        weights = out_dir / f"model_test{test_year}.pt"
        result_json = out_dir / f"result_test{test_year}.json"

        if args.skip_existing and result_json.exists():
            print(f"\n=== fold: テスト={test_year}年 (既存の結果を再利用) ===")
        else:
            print(f"\n{'=' * 60}\n=== fold: テスト={test_year}年 ===\n{'=' * 60}")
            # 学習は終わっているのに評価で落ちた場合、--skip-existing を付ければ
            # 学習をやり直さずに評価だけ再開できる(1foldに数十分かかるため)
            if args.skip_existing and weights.exists():
                print(f"  学習済みの重みを再利用します: {weights}")
                train_cmd = None
            else:
                train_cmd = [
                    "src.train", *common,
                    "--test-year", test_year,
                    "--epochs", args.epochs,
                    "--batch-size", args.batch_size,
                    "--image-size", args.image_size,
                    "--out", weights,
                ]
                if args.input_mode == "era5-grid":
                    train_cmd += ["--grid-size", args.grid_size]
            if train_cmd is not None:
                if args.coordconv:
                    train_cmd.append("--coordconv")
                run(train_cmd)
            run([
                "src.evaluate", *common,
                "--test-year", test_year,
                "--batch-size", args.batch_size,
                "--weights", weights,
                "--split", "test",
                "--optimize-thresholds",
                "--json-out", result_json,
            ])

        results.append(json.loads(result_json.read_text()))

    # ---- 集計 ----
    print(f"\n{'=' * 72}\n交差検証のまとめ({len(results)}fold)\n{'=' * 72}")

    def summarize(values):
        mean = statistics.mean(values)
        sd = statistics.stdev(values) if len(values) > 1 else 0.0
        return f"{mean:.3f} ± {sd:.3f}"

    print("\n【全体】")
    for key, name in (
        ("macro_f1_evaluable", "macro F1(評価できたラベルのみ)"),
        ("macro_f1_all_labels", "macro F1(全ラベル)"),
        ("micro_f1", "micro F1"),
        ("weighted_f1", "weighted F1"),
    ):
        per_fold = ", ".join(f"{r[key]:.3f}" for r in results)
        print(f"  {name:<32} {summarize([r[key] for r in results])}   (各fold: {per_fold})")

    print("\n【ラベル別 F1】")
    header = f"  {'ラベル':<22} {'平均 ± 標準偏差':<18} " + " ".join(
        f"{r['test_year']}年" for r in results
    )
    print(header)
    print("  " + "-" * (len(header) + 6))
    for label in LABELS:
        f1s = [r["per_label"][label]["f1"] for r in results]
        supports = [r["per_label"][label]["support"] for r in results]
        detail = " ".join(f"{f:.2f}({s})" for f, s in zip(f1s, supports))
        print(f"  {LABEL_JA[label]:<22} {summarize(f1s):<18} {detail}")
    print("\n  ※ 括弧内はそのfoldのテストセットでの出現件数(support)")

    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "folds": results,
                "mean": {
                    key: statistics.mean([r[key] for r in results])
                    for key in ("macro_f1_evaluable", "macro_f1_all_labels", "micro_f1", "weighted_f1")
                },
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"\nまとめを書き出しました: {summary_path}")


if __name__ == "__main__":
    main()
