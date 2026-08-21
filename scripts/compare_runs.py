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



def fold_trivial(fold: dict, labels: list) -> float:
    """そのfoldで「全部を陽性と答える」だけで得られる、指定ラベルでのmacro F1。

    ラベルを絞って平均を取り直すなら、基準も同じラベルで取り直さないと比べられない
    -- 前線2つは出現率が高く(停滞前線は3割)、基準を押し上げているので、外したまま
    全ラベルの基準と比べると上積みを過大評価する。

    出現率はper_labelのsupportとn_evalから復元する。src/evaluate.pyが基準を
    記録するようになる前の結果でも当てられるようにするため。
    """
    n = fold.get("n_eval")
    if not n:
        return float("nan")
    scores = []
    for label in labels:
        prevalence = fold["per_label"][label]["support"] / n
        if prevalence > 0:
            scores.append(2 * prevalence / (1 + prevalence))
    return statistics.mean(scores) if scores else float("nan")


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



def paired_diff(baseline: list, other: list, test_years: list) -> str:
    """同じfoldどうしを引き算して差を示す。

    どの実行も同じ年をテストに使うので、平均どうしを別々の標準偏差と見比べるより、
    fold単位で対応させたほうが小さい差を判定できる -- foldによる難易度の違いが
    引き算で消えるため。年ごとの差が揃って同符号なら、平均の差が標準偏差より
    小さくても実質的な改善と読める。逆に符号がばらつくなら、平均が動いていても
    たまたまである可能性が高い。
    """
    diffs = [b - a for a, b in zip(baseline, other)]
    detail = " ".join(f"{y}:{d:+.3f}" for y, d in zip(test_years, diffs))
    mean = statistics.mean(diffs)
    same_sign = all(d > 0 for d in diffs) or all(d < 0 for d in diffs)
    verdict = "全foldで同符号" if same_sign else "foldで符号がばらつく(差は不確か)"
    return f"  基準比: {mean:+.3f}  [{detail}]  {verdict}"


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
    baseline_subset = None
    test_years = [f.get("test_year", "?") for f in summaries[0][1]["folds"]]

    print("=" * 72)
    if args.exclude:
        print(f"除外したラベル: {', '.join(LABEL_JA[l] for l in args.exclude)}")
    print(f"平均に使うラベル: {len(kept)}個")
    print("=" * 72)

    print(f"\n{'実行':<28} {'全ラベル':>10} {'除外後':>10}   差")
    print("-" * 72)
    for path, summary in summaries:
        full = [f["macro_f1_evaluable"] for f in summary["folds"]]
        trivial_full = [t for t in (fold_trivial(f, LABELS) for f in summary["folds"]) if t == t]
        trivial_kept = [t for t in (fold_trivial(f, kept) for f in summary["folds"]) if t == t]
        subset = per_fold_macro(summary, kept)
        delta = statistics.mean(subset) - statistics.mean(full)
        print(f"{path.name:<28} {statistics.mean(full):>10.3f} {statistics.mean(subset):>10.3f}   {delta:+.3f}")
        print(f"  {describe(summary)}")
        if trivial_full:
            base_full = statistics.mean(trivial_full)
            base_kept = statistics.mean(trivial_kept)
            margin_full = statistics.mean(full) - base_full
            margin_kept = statistics.mean(subset) - base_kept
            flag = "  ← 自明な予測を上回っていません" if margin_kept <= 0 else ""
            print(f"  自明な予測(全部陽性): 全ラベル {base_full:.3f} / 除外後 {base_kept:.3f}")
            print(f"  上積み:               全ラベル {margin_full:+.3f} / 除外後 {margin_kept:+.3f}{flag}")
        line = f"  除外後の各fold: {', '.join(f'{s:.3f}' for s in subset)}"
        if len(subset) > 1:
            line += f"  (標準偏差 {statistics.stdev(subset):.3f})"
        print(line)
        if baseline_subset is None:
            baseline_subset = subset
            print("  (以降の実行はこれを基準に比較します)")
        elif len(baseline_subset) == len(subset):
            print(paired_diff(baseline_subset, subset, test_years))

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
