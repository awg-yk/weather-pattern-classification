"""交差検証の結果どうしを、ラベルを絞って比べ直す。

macro F1は全ラベルの平均なので、一部のラベルだけが極端に不利な入力を使うと、
その入力の実力より低い数字が出る。ERA5格子(--input-mode era5-grid)には前線の
記号が入っていない -- 前線は人間が解析して描いたものであって、気圧・気温の
格子から復元できるものではない。天気図との差が「前線が見えないこと」で
説明できてしまうのか、それ以外にも差があるのかは、macro F1をひとつ見ている
限り分からない。

そこで、保存済みのsummary.jsonから、指定したラベルを除いて集計しなおす。
学習はやり直さない(同じfold・同じ予測から、平均の取り方だけを変える)。

使い方:
    python -m scripts.compare_runs runs/loyo_scratch runs/loyo_grid_pretrained \
        --exclude front_passage stationary_front
"""

import argparse
import json
import statistics
from pathlib import Path

from src.labels import LABEL_JA, LABELS


def load(run_dir: Path) -> dict:
    path = run_dir / "summary.json" if run_dir.is_dir() else run_dir
    if not path.exists():
        raise SystemExit(f"{path} がありません。先に scripts/cross_validate.py を実行してください。")
    return json.loads(path.read_text(encoding="utf-8"))


def describe(summary: dict) -> str:
    """設定を1行で表す。configを記録する前に作ったsummary.jsonでも落ちないようにする。"""
    config = summary.get("config")
    if not config:
        return "(設定の記録なし: configを保存する前に実行された結果)"
    parts = [config.get("input_mode", "chart")]
    parts.append("事前学習なし" if config.get("no_pretrained") else "事前学習あり")
    if config.get("coordconv"):
        parts.append("coordconv")
    if config.get("era5_features"):
        parts.append("era5特徴量あり")
    return " / ".join(parts)


def per_fold_macro(summary: dict, labels: list) -> list:
    """foldごとに、指定したラベルだけでmacro F1を取り直す。

    そのfoldのテストセットに1件も出現しないラベルは平均から外す。支持数0の
    ラベルのF1は0と記録されるため、含めると「予測できなかった」のと区別が
    つかず、平均を不当に押し下げる。
    """
    scores = []
    for fold in summary["folds"]:
        f1s = [
            fold["per_label"][label]["f1"]
            for label in labels
            if fold["per_label"][label]["support"] > 0
        ]
        scores.append(statistics.mean(f1s) if f1s else float("nan"))
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="比較したい交差検証の出力ディレクトリ(またはsummary.json)")
    parser.add_argument(
        "--exclude", nargs="*", default=[],
        help="平均から外すラベル(英語キー)。例: front_passage stationary_front",
    )
    args = parser.parse_args()

    unknown = [label for label in args.exclude if label not in LABELS]
    if unknown:
        raise SystemExit(f"知らないラベルです: {unknown}\n使えるのは: {LABELS}")

    kept = [label for label in LABELS if label not in args.exclude]
    if not kept:
        raise SystemExit("すべてのラベルを除外しています")

    summaries = [(Path(run), load(Path(run))) for run in args.runs]

    print("=" * 72)
    if args.exclude:
        print(f"除外したラベル: {', '.join(LABEL_JA[l] for l in args.exclude)}")
    print(f"平均に使うラベル: {len(kept)}個")
    print("=" * 72)

    print(f"\n{'実行':<28} {'全ラベル':>10} {'除外後':>10}   差")
    print("-" * 72)
    for path, summary in summaries:
        full = [f["macro_f1_evaluable"] for f in summary["folds"]]
        subset = per_fold_macro(summary, kept)
        delta = statistics.mean(subset) - statistics.mean(full)
        print(f"{path.name:<28} {statistics.mean(full):>10.3f} {statistics.mean(subset):>10.3f}   {delta:+.3f}")
        print(f"  {describe(summary)}")
        line = f"  除外後の各fold: {', '.join(f'{s:.3f}' for s in subset)}"
        if len(subset) > 1:
            line += f"  (標準偏差 {statistics.stdev(subset):.3f})"
        print(line)

    print(f"\n【ラベル別 F1(平均)】")
    header = "".join(f"{p.name[:14]:>16}" for p, _ in summaries)
    print(f"  {'ラベル':<24}{header}")
    print("  " + "-" * (24 + 16 * len(summaries)))
    for label in LABELS:
        marker = " *" if label in args.exclude else "  "
        cells = ""
        for _, summary in summaries:
            f1s = [f["per_label"][label]["f1"] for f in summary["folds"]
                   if f["per_label"][label]["support"] > 0]
            cells += f"{statistics.mean(f1s):>16.3f}" if f1s else f"{'-':>16}"
        print(f"{marker}{LABEL_JA[label]:<24}{cells}")
    if args.exclude:
        print("\n  * = 平均から除外したラベル")


if __name__ == "__main__":
    main()
