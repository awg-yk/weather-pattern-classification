"""学習データの件数を変えて学習・評価し、件数と性能の関係をまとめる。

分割方式は交差検証と同じものを使う。ランダム分割で測ると隣接日のリークで
件数の効果が実態より大きく見えてしまうため、必ず時間ブロック分割で測ること。

テストセットは件数によらず同一なので、変わっているのは学習件数だけになる。

使い方:
    python -m scripts.learning_curve \
        --data-dir data/processed --labels data/labels.csv \
        --years 2023 2024 2025 --split-mode loyo --test-year 2025 \
        --limits 100 300 600 900 --out-dir runs/curve
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from src.labels import LABEL_JA, LABELS


def run(cmd: list) -> None:
    print("\n$ " + " ".join(str(c) for c in cmd), flush=True)
    result = subprocess.run([sys.executable, "-m"] + [str(c) for c in cmd])
    if result.returncode != 0:
        raise SystemExit(f"失敗しました(終了コード {result.returncode})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--limits", type=int, nargs="+", required=True, help="試す学習件数")
    parser.add_argument("--out-dir", default="runs/curve")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    # 分割の指定(交差検証と同じ値を使うこと)
    parser.add_argument("--years", type=int, nargs="+", default=None)
    parser.add_argument("--split-mode", default="loyo")
    parser.add_argument("--test-year", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.0)
    parser.add_argument("--gap-days", type=int, default=3)
    parser.add_argument("--skip-existing", action="store_true", help="結果JSONがあるものは飛ばす")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    common = [
        "--data-dir", args.data_dir,
        "--labels", args.labels,
        "--split-mode", args.split_mode,
        "--val-ratio", args.val_ratio,
        "--test-ratio", args.test_ratio,
        "--gap-days", args.gap_days,
        "--seed", args.seed,
    ]
    if args.years:
        common += ["--years", *args.years]
    if args.test_year is not None:
        common += ["--test-year", args.test_year]

    # None は「制限なし(全件)」を意味する
    settings = [*args.limits, None]
    results = []

    for limit in settings:
        tag = "full" if limit is None else str(limit)
        weights = out_dir / f"model_n{tag}.pt"
        result_json = out_dir / f"result_n{tag}.json"

        if args.skip_existing and result_json.exists():
            print(f"\n=== 学習件数 {tag} (既存の結果を再利用) ===")
        else:
            print(f"\n{'=' * 60}\n=== 学習件数 {tag} ===\n{'=' * 60}")
            train_cmd = ["src.train", *common,
                         "--epochs", args.epochs,
                         "--batch-size", args.batch_size,
                         "--image-size", args.image_size,
                         "--out", weights]
            if limit is not None:
                train_cmd += ["--train-limit", limit]
            run(train_cmd)

            run(["src.evaluate", *common,
                 "--batch-size", args.batch_size,
                 "--weights", weights,
                 "--split", "test",
                 "--optimize-thresholds",
                 "--json-out", result_json])

        payload = json.loads(result_json.read_text())
        payload["train_limit"] = limit
        payload["tag"] = tag
        results.append(payload)

    # ---- まとめ ----
    print(f"\n{'=' * 72}\n学習件数と性能\n{'=' * 72}\n")
    print(f"  {'学習件数':<10}{'macro F1':>10}{'micro F1':>10}{'weighted F1':>13}")
    print("  " + "-" * 43)
    for r in results:
        label = "全件" if r["train_limit"] is None else str(r["train_limit"])
        print(
            f"  {label:<10}{r['macro_f1_evaluable']:>10.3f}"
            f"{r['micro_f1']:>10.3f}{r['weighted_f1']:>13.3f}"
        )

    print(f"\n【ラベル別 F1】")
    header = f"  {'ラベル':<22}" + "".join(
        f"{('全件' if r['train_limit'] is None else r['tag']):>8}" for r in results
    )
    print(header)
    print("  " + "-" * (len(header)))
    for label in LABELS:
        row = "".join(f"{r['per_label'][label]['f1']:>8.2f}" for r in results)
        print(f"  {LABEL_JA[label]:<22}{row}")

    summary = out_dir / "summary.json"
    summary.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nまとめを書き出しました: {summary}")


if __name__ == "__main__":
    main()
