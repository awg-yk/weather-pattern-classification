"""ラベルの付き方を数える。件数と、どのラベルと同時に付いているか。

ラベルの定義が揺れていると、モデルは同じ気圧配置に違う答えを要求されることになり、
学習しようがない。それはF1が低いという形で現れるが、数字だけを見ても
「難しいラベル」なのか「基準が揺れているラベル」なのかは区別できない。

同時に付いているラベルの組み合わせを見ると、運用が透ける。例えば
futatsudama_low(二つ玉低気圧)は106件あるが、japan_sea_low と nankigan_low の
両方が同時に付いているものは0件だった -- 個別の2ラベルを足すのではなく、
置き換える運用になっている。そこから外れた13件が、判定の揺れを示す。

使い方:
    python -m scripts.label_stats --labels data/labels_v2.csv
    python -m scripts.label_stats --labels data/labels_v2.csv --label futatsudama_low \
        --implied-by japan_sea_low nankigan_low
"""

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

from src.labels import LABEL_JA, LABELS, parse_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--label", default=None, help="このラベルの共起だけを詳しく見る")
    parser.add_argument("--years", type=int, nargs="*", default=None)
    parser.add_argument("--top", type=int, default=12, help="共起の組み合わせを何位まで出すか")
    parser.add_argument(
        "--implied-by", nargs="+", default=None,
        help="これらが揃っていれば--labelも付くはず、という組み合わせ。"
        "気象の知識で決めるものなので指定式にしている"
        "(例: --label futatsudama_low --implied-by japan_sea_low nankigan_low)",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.labels)
    df["parsed_labels"] = df["label"].apply(parse_labels)
    df = df[df["parsed_labels"].apply(len) > 0].reset_index(drop=True)
    if args.years:
        stamps = df["filename"].str.extract(r"(\d{10})")[0]
        years = pd.to_datetime(stamps, format="%Y%m%d%H", errors="coerce").dt.year
        df = df[years.isin(args.years)].reset_index(drop=True)

    sets = [set(labels) for labels in df["parsed_labels"]]
    print(f"{args.labels}: {len(df)}件" + (f" (対象年 {args.years})" if args.years else ""))

    print("\n【ラベルごとの件数】")
    for label in LABELS:
        count = sum(label in s for s in sets)
        print(f"  {LABEL_JA[label]:<24}{count:>6}件  ({count / len(df):>5.1%})")

    print("\n【1枚に付くラベル数】")
    for size, count in sorted(Counter(len(s) for s in sets).items()):
        print(f"  {size}個: {count}件")

    if not args.label:
        return
    if args.label not in LABELS:
        raise SystemExit(f"知らないラベルです: {args.label}\n使えるのは: {LABELS}")

    target = [s for s in sets if args.label in s]
    print(f"\n【{LABEL_JA[args.label]} と同時に付いているラベル】({len(target)}件)")
    combos = Counter(tuple(sorted(s - {args.label})) for s in target)
    for combo, count in combos.most_common(args.top):
        name = " + ".join(LABEL_JA[l] for l in combo) if combo else "(このラベルだけ)"
        print(f"  {count:>5}件  {name}")

    if args.implied_by:
        unknown = [l for l in args.implied_by if l not in LABELS]
        if unknown:
            raise SystemExit(f"知らないラベルです: {unknown}")
        expected = set(args.implied_by)
        names = " と ".join(LABEL_JA[l] for l in args.implied_by)
        with_target = sum(expected <= s for s in target)
        without = sum(expected <= s for s in sets if args.label not in s)
        print(f"\n【{names} が揃っている天気図】")
        print(f"  {LABEL_JA[args.label]} あり: {with_target}件")
        print(f"  {LABEL_JA[args.label]} なし: {without}件")
        print("  どちらかに大きく偏っていれば運用は一貫している。"
              "混ざっているなら判定が揺れている。")

if __name__ == "__main__":
    main()
