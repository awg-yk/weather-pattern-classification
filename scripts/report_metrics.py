"""交差検証のまとめ(summary.json)から、発表で説明しやすい数字を出す。

macro F1 は「何%当たるか」ではないので、そのまま見せると誤解される。
このスクリプトは、意味の違う3つの数字を並べて出す:

    1位正解率   モデルが一番自信のあるラベルが、正解に含まれていた割合。
                「何%当たるか」に一番近い。人に説明するならこれ
    適合率      「このラベルだ」と言ったうちの、当たっていた割合(空振りの少なさ)
    再現率      実際にそのラベルだった日のうち、拾えた割合(見逃しの少なさ)

macro F1 は適合率と再現率の調和平均を、10ラベルで**均等に**平均したもの。
出現の少ないラベル(オホーツク海高気圧など)も1票として数えるので、
よく出るラベルだけ当てても上がらない。だから正解率より辛い数字になる。

使い方:
    python -m scripts.report_metrics --run runs/new_annot
    python -m scripts.report_metrics --run runs/new_annot --compare runs/new_baseline
    python -m scripts.report_metrics --run runs/new_annot --plot reports/precision_recall.png
    python -m scripts.report_metrics --run runs/new_annot --csv reports/label_metrics.csv
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.labels import LABELS, LABEL_JA


def load_summary(run_dir) -> dict:
    path = Path(run_dir)
    if path.is_dir():
        path = path / "summary.json"
    if not path.exists():
        raise SystemExit(
            f"まとめが見つかりません: {path}\n"
            "  交差検証を回すと runs/<名前>/summary.json ができます")
    return json.loads(path.read_text(encoding="utf-8"))


def mean_sd(values) -> tuple:
    """平均と標準偏差。foldが1つだけなら標準偏差は0にする。"""
    return (statistics.mean(values),
            statistics.stdev(values) if len(values) > 1 else 0.0)


def overall(summary: dict) -> dict:
    """全体の数字を、foldをまたいだ平均±標準偏差でまとめる。"""
    folds = summary["folds"]
    out = {}
    for key in ("top1_accuracy", "macro_f1_evaluable", "micro_f1",
                "weighted_f1", "trivial_macro_f1"):
        # 古い実行のまとめには入っていない項目がありうる
        values = [f[key] for f in folds if key in f]
        if values:
            out[key] = mean_sd(values)
    out["_folds"] = [f["test_year"] for f in folds]
    out["_n_eval"] = [f.get("n_eval") for f in folds]
    return out


def per_label(summary: dict) -> dict:
    """ラベルごとの適合率・再現率・F1・件数を、foldをまたいで平均する。"""
    folds = summary["folds"]
    table = {}
    for label in LABELS:
        rows = [f["per_label"][label] for f in folds]
        table[label] = {
            "precision": mean_sd([r["precision"] for r in rows]),
            "recall": mean_sd([r["recall"] for r in rows]),
            "f1": mean_sd([r["f1"] for r in rows]),
            "support": sum(r["support"] for r in rows),
        }
    return table


def top1_baselines(summary: dict) -> dict:
    """1位正解率を読むための下敷き。

    **83%が良いのかどうかは、何もしない場合と比べないと分からない。**
    このデータは1枚に複数のラベルが付くので、1位が正解の1つに当たれば
    「当たり」と数える。当たりやすさは1枚あたりのラベル数で決まる。
    """
    per_chart, chance, majority = [], [], []
    for fold in summary["folds"]:
        n = fold.get("n_eval")
        if not n:
            continue
        supports = [fold["per_label"][label]["support"] for label in LABELS]
        total = sum(supports)
        per_chart.append(total / n)
        # でたらめに1つ選んだとき、それが正解に含まれている確率
        chance.append(total / n / len(LABELS))
        # 一番多いラベルを毎回答えたときに当たる割合
        majority.append(max(supports) / n)
    if not per_chart:
        return {}
    return {"per_chart": mean_sd(per_chart), "chance": mean_sd(chance),
            "majority": mean_sd(majority)}


def print_overall(summary: dict, name: str) -> None:
    got = overall(summary)
    years = "・".join(str(y) for y in got["_folds"])
    total = sum(n for n in got["_n_eval"] if n)
    print(f"\n{'=' * 64}\n{name}\n{'=' * 64}")
    # 古いまとめには枚数が入っていない。0枚と言うより、黙っている方がよい
    counted = f"、評価した天気図 {total}枚" if total else ""
    print(f"  {len(got['_folds'])}fold(テスト年 {years}){counted}\n")

    base = top1_baselines(summary)
    if "top1_accuracy" in got:
        mean, sd = got["top1_accuracy"]
        print(f"  1位正解率        {mean * 100:5.1f}% ± {sd * 100:.1f}"
              f"   ← 「何%当たるか」に一番近い数字")
        if base:
            print(f"      1枚に付く正解ラベルは平均 {base['per_chart'][0]:.2f}個。"
                  "1位がそのどれかに当たれば正解と数える")
            print(f"      でたらめに1つ選んだ場合   {base['chance'][0] * 100:5.1f}%")
            print(f"      一番多いラベルを毎回答えた場合 {base['majority'][0] * 100:5.1f}%")
    for key, label, note in (
        ("macro_f1_evaluable", "macro F1", "10ラベルを均等に平均。出現の少ないラベルも1票"),
        ("micro_f1", "micro F1", "全ての判定をまとめて数えたF1。多いラベルの影響が大きい"),
        ("weighted_f1", "weighted F1", "出現件数で重みを付けた平均"),
    ):
        if key in got:
            mean, sd = got[key]
            print(f"  {label:<16} {mean:5.3f}  ± {sd:.3f}   ← {note}")
    if "trivial_macro_f1" in got:
        mean, _ = got["trivial_macro_f1"]
        macro = got["macro_f1_evaluable"][0]
        print(f"\n  自明な予測のmacro F1  {mean:.3f}"
              f"(全部『該当する』と答え続けた場合)")
        print(f"  そこからの上積み      +{macro - mean:.3f}")


def print_per_label(summary: dict) -> None:
    table = per_label(summary)
    print(f"\n  {'ラベル':<24}{'適合率':>10}{'再現率':>10}{'F1':>10}{'件数':>8}")
    print("  " + "-" * 62)
    # 件数の多い順。珍しいラベルほど下に来るので、成績の傾向が読み取りやすい
    for label in sorted(LABELS, key=lambda l: -table[l]["support"]):
        row = table[label]
        print(f"  {LABEL_JA[label]:<24}"
              f"{row['precision'][0] * 100:9.1f}%"
              f"{row['recall'][0] * 100:9.1f}%"
              f"{row['f1'][0]:10.3f}"
              f"{row['support']:8d}")
    print("\n  適合率 = 「このラベルだ」と言ったうちの当たり(空振りの少なさ)")
    print("  再現率 = 実際にそのラベルだった日のうち拾えた割合(見逃しの少なさ)")
    print("  件数   = 3foldを合わせた、実際にそのラベルだった天気図の枚数")


def print_comparison(a: dict, b: dict, name_a: str, name_b: str) -> None:
    """2つの実行を、同じ物差しで並べる。"""
    got_a, got_b = overall(a), overall(b)
    print(f"\n{'=' * 64}\n比較\n{'=' * 64}")
    print(f"  {'':<18}{name_b:>14}{name_a:>14}{'差':>10}")
    print("  " + "-" * 56)
    for key, label, scale in (("top1_accuracy", "1位正解率", 100),
                              ("macro_f1_evaluable", "macro F1", 1),
                              ("micro_f1", "micro F1", 1),
                              ("weighted_f1", "weighted F1", 1)):
        if key not in got_a or key not in got_b:
            continue
        va, vb = got_a[key][0] * scale, got_b[key][0] * scale
        unit = "%" if scale == 100 else ""
        fmt = ".1f" if scale == 100 else ".3f"
        print(f"  {label:<18}{vb:>13{fmt}}{unit}{va:>13{fmt}}{unit}"
              f"{va - vb:>+9{fmt}}{unit}")

    ta, tb = per_label(a), per_label(b)
    print("\n  ラベル別 F1")
    print(f"  {'ラベル':<24}{name_b:>10}{name_a:>10}{'差':>10}")
    print("  " + "-" * 54)
    for label in sorted(LABELS, key=lambda l: -ta[l]["support"]):
        va, vb = ta[label]["f1"][0], tb[label]["f1"][0]
        print(f"  {LABEL_JA[label]:<24}{vb:10.3f}{va:10.3f}{va - vb:>+10.3f}")


def write_csv(summary: dict, path: Path) -> None:
    import pandas as pd

    table = per_label(summary)
    rows = []
    for label in sorted(LABELS, key=lambda l: -table[l]["support"]):
        row = table[label]
        rows.append({
            "ラベル": LABEL_JA[label],
            "適合率": round(row["precision"][0], 3),
            "適合率_標準偏差": round(row["precision"][1], 3),
            "再現率": round(row["recall"][0], 3),
            "再現率_標準偏差": round(row["recall"][1], 3),
            "F1": round(row["f1"][0], 3),
            "F1_標準偏差": round(row["f1"][1], 3),
            "件数": row["support"],
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    # Excelでそのまま開けるようBOM付きUTF-8で書く
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n書き出しました: {path}")


def plot(summary: dict, path: Path, title: str) -> None:
    """適合率と再現率を横並びの棒で描く。**F1だけでは、空振りが多いのか
    見逃しが多いのかが分からない。**手を打つ先を決めるのはこの2つ。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from src.jp_font import missing_font_hint, register_matplotlib_cjk

    if not register_matplotlib_cjk():
        print("警告: 日本語フォントが見つかりません。" + missing_font_hint())

    table = per_label(summary)
    order = sorted(LABELS, key=lambda l: table[l]["support"])
    names = [f"{LABEL_JA[l]}\n({table[l]['support']}件)" for l in order]
    precision = [table[l]["precision"][0] for l in order]
    recall = [table[l]["recall"][0] for l in order]
    p_sd = [table[l]["precision"][1] for l in order]
    r_sd = [table[l]["recall"][1] for l in order]

    y = np.arange(len(order))
    height = 0.38
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(y + height / 2, precision, height, xerr=p_sd, label="適合率(空振りの少なさ)",
            color="#4C72B0", error_kw={"ecolor": "#333", "capsize": 3, "lw": 1})
    ax.barh(y - height / 2, recall, height, xerr=r_sd, label="再現率(見逃しの少なさ)",
            color="#DD8452", error_kw={"ecolor": "#333", "capsize": 3, "lw": 1})

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xlabel("割合")
    ax.set_title(title, pad=28)
    # 凡例は棒に重ねない。**重なると、短い棒がどちらの色かを読めなくなる**
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    print(f"\n図を書き出しました: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True,
                        help="交差検証の出力フォルダ(例 runs/new_annot)")
    parser.add_argument("--compare", default=None,
                        help="比べる相手のフォルダ(例 runs/new_baseline)")
    parser.add_argument("--name", default=None, help="--run の呼び名")
    parser.add_argument("--compare-name", default=None, help="--compare の呼び名")
    parser.add_argument("--csv", default=None, help="ラベル別の表をCSVで書き出す")
    parser.add_argument("--plot", default=None, help="適合率と再現率の図を書き出す")
    args = parser.parse_args()

    name = args.name or Path(args.run).name
    summary = load_summary(args.run)
    print_overall(summary, name)
    print_per_label(summary)

    if args.compare:
        other_name = args.compare_name or Path(args.compare).name
        other = load_summary(args.compare)
        print_overall(other, other_name)
        print_per_label(other)
        print_comparison(summary, other, name, other_name)

    if args.csv:
        write_csv(summary, Path(args.csv))
    if args.plot:
        plot(summary, Path(args.plot), f"ラベル別の適合率と再現率({name})")


if __name__ == "__main__":
    main()
