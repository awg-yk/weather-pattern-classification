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
    """結果JSONを読む。UTF-8で書かれていないものも読めるようにする。

    書き出し側でencodingを指定していなかった時期があり、Windowsで実行した結果は
    cp932(Shift-JIS)で保存されている。ラベル名に日本語が入るので、UTF-8として
    読むと途中のバイトで落ちる。過去の結果を捨てずに比べられるよう、順に試す。
    """
    path = run_dir / "summary.json" if run_dir.is_dir() else run_dir
    if not path.exists():
        raise SystemExit(f"{path} がありません。先に scripts/cross_validate.py を実行してください。")
    for encoding in ("utf-8", "cp932"):
        try:
            return json.loads(path.read_text(encoding=encoding))
        except UnicodeDecodeError:
            continue
    raise SystemExit(f"{path} の文字コードを判別できません(utf-8・cp932のどちらでもありません)。")


SPLIT_KEYS = ("val_mode", "seed", "gap_days", "val_ratio", "split_mode", "years")


def check_same_split(summaries) -> list:
    """分割の条件がそろっているかを確かめる。

    **そろっていないと、モデルの差と学習データ量の差が混ざる。**
    実測で、`--val-mode spread` は `tail` より学習データが2割少なくなった
    (テスト2025年で 906件 対 1157件)。片方だけ spread で回した結果を
    tail の結果と並べると、+0.025 の上積みがモデルのものか学習データ量の
    ものか分けられない。

    `docs/2026-08-26-stale-run-comparisons.md` と同じ落とし穴である。
    ラベルの食い違いは既に見ているが、分割の条件は見ていなかった。
    """
    base_path, base = summaries[0]
    base_config = base.get("config") or {}
    problems = []
    for path, summary in summaries[1:]:
        config = summary.get("config") or {}
        if not config or not base_config:
            continue
        differing = {}
        for key in SPLIT_KEYS:
            if key in base_config and key in config and base_config[key] != config[key]:
                differing[key] = (base_config[key], config[key])
        if differing:
            problems.append((path, differing))
    return problems


def describe(summary: dict) -> str:
    """設定を1行で表す。configを記録する前に作ったsummary.jsonでも落ちないようにする。"""
    config = summary.get("config")
    if not config:
        return "(設定の記録なし: configを保存する前に実行された結果)"

    # CNN以外の手法。configのキーがまるごと違うので、CNNの言葉で説明すると
    # 嘘になる(「事前学習あり」など)。取り違えの元なので分けて書く。
    if summary.get("method") == "features":
        parts = ["検出した特徴量", str(config.get("model", "?"))]
        columns = summary.get("feature_columns")
        if columns:
            parts.append(f"{len(columns)}個")
        return " / ".join(parts)

    if str(summary.get("method", "")).startswith("blend"):
        return " / ".join([
            "天気図CNN + 検出した特徴量の混合",
            str(config.get("model", "?")),
            f"重みは検証データでラベルごとに選択({config.get('val_mode', '?')})",
        ])

    parts = [config.get("input_mode", "chart")]
    parts.append("事前学習なし" if config.get("no_pretrained") else "事前学習あり")
    if config.get("coordconv"):
        parts.append("coordconv")
    if config.get("era5_features"):
        parts.append("era5特徴量あり")
    return " / ".join(parts)



def support_signature(summary: dict) -> dict:
    """foldごとの陽性件数(support)を、ラベルごとに並べたもの。

    同じラベルファイルで走った実行なら、この値は一致する。ラベルを付け直したり
    ラベルファイルを差し替えたりすると変わるので、「別のラベルで測った数字を
    比べていないか」の判定に使える。summary.jsonにラベルファイルの記録が無い
    時期の結果でも、後から機械的に判定できるのが利点。
    """
    return {
        label: [fold["per_label"].get(label, {}).get("support") for fold in summary["folds"]]
        for label in LABELS
    }


def check_comparable(summaries: list) -> list:
    """比較可能かを確かめ、ラベルが食い違っているものを一覧で返す。

    実際にあった事故: 台風ラベルをベストトラックから付け直したあと、
    付け直す前の実行(runs/loyo_v2、陽性219件)と後の実行(247件)を並べて
    「+0.192の改善」と読んでしまった。差はモデルではなくラベルのものだった。
    """
    if len(summaries) < 2:
        return []
    (base_path, base_summary) = summaries[0]
    base = support_signature(base_summary)
    base_fingerprint = base_summary.get("labels_fingerprint")
    problems = []
    for path, summary in summaries[1:]:
        other = support_signature(summary)
        mismatched = [label for label in LABELS if base[label] != other[label]]
        # 指紋が両方にあって食い違うなら、supportがたまたま揃っていても別のラベル。
        # supportは陽性件数しか見ないので、陽性の枚数を変えずに中身を入れ替えた
        # 修正(付け間違いの差し替えなど)を見逃す。
        other_fingerprint = summary.get("labels_fingerprint")
        if base_fingerprint and other_fingerprint and base_fingerprint != other_fingerprint:
            mismatched = mismatched or ["(指紋不一致)"]
        if mismatched:
            problems.append((path, mismatched, base, other))
    return problems


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
    parser.add_argument(
        "--force", action="store_true",
        help="ラベルが食い違っていても比較する。差にラベルの影響が入ることを承知のうえで使う",
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

    problems = check_comparable(summaries)
    if problems and not args.force:
        base_path = summaries[0][0]
        lines = [
            "ラベルが食い違っています。このまま比べると、モデルの差とラベルの差が混ざります。",
            "",
        ]
        for path, mismatched, base, other in problems:
            lines.append(f"{path.name} と {base_path.name} で陽性件数が違うラベル:")
            for label in mismatched:
                if label not in LABEL_JA:
                    lines.append(f"  ラベルファイルの指紋が違います")
                    continue
                lines.append(
                    f"  {LABEL_JA[label]:<22} {base_path.name}={base[label]}  {path.name}={other[label]}"
                )
            lines.append("")
        lines += [
            "同じラベルファイルで両方を測り直してください。",
            "食い違っているラベルを --exclude で外すか、承知のうえなら --force で続行できます。",
        ]
        raise SystemExit("\n".join(lines))
    if problems and args.force:
        print("警告: ラベルが食い違ったまま比較しています(--force)。差にラベルの影響が入ります。\n")

    split_problems = check_same_split(summaries)
    if split_problems:
        base_name = summaries[0][0].name
        print("=" * 72)
        print("★分割の条件が違います。差にモデル以外の影響が入ります。")
        for path, differing in split_problems:
            for key, (base_value, other_value) in differing.items():
                print(f"  {key:<12} {base_name}={base_value}  {path.name}={other_value}")
        print()
        print("  val_mode が違うと学習データの量が変わる(実測で spread は tail より")
        print("  2割少ない)。同じ条件で測り直してから比べること。")
        print("=" * 72)
        print()

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
