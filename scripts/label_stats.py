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
    parser.add_argument(
        "--by-year", action="store_true",
        help="年ごとの出現率を並べる。ある年だけ大きく違えば、その年の判定基準が"
        "ずれている疑いがある",
    )
    parser.add_argument(
        "--out-csv", default=None,
        help="見直しの候補をCSVに書き出す。kind列で「判定が要るもの」と"
        "「規約を当てるだけのもの」を分ける",
    )
    parser.add_argument(
        "--list-exceptions", action="store_true",
        help="--implied-by の運用から外れている天気図のファイル名を並べる(見直しの対象)",
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

    if args.by_year:
        stamps = df["filename"].str.extract(r"(\d{10})")[0]
        year_of = pd.to_datetime(stamps, format="%Y%m%d%H", errors="coerce").dt.year
        years_present = sorted(year_of.dropna().unique())
        print("\n【年ごとの出現率】")
        header = "".join(f"{int(y):>10}" for y in years_present)
        print(f"  {'ラベル':<24}{header}      幅")
        print("  " + "-" * (24 + 10 * len(years_present) + 8))
        for label in LABELS:
            rates = []
            for year in years_present:
                rows = [s for s, y in zip(sets, year_of) if y == year]
                rates.append(sum(label in s for s in rows) / len(rows) if rows else 0.0)
            cells = "".join(f"{rate:>9.1%} " for rate in rates)
            spread = max(rates) - min(rates)
            mark = "  ←" if spread > 0.5 * max(rates) and max(rates) > 0.02 else ""
            print(f"  {LABEL_JA[label]:<24}{cells}{spread:>7.1%}{mark}")
        print("\n  幅 = 最大の年と最小の年の差。← は幅が最大値の半分を超えるもの。")
        print("  気候の年々変動でも動くが、極端なものは判定基準がその年だけ"
              "ずれている疑いがある。")

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

        if args.list_exceptions:
            # --implied-by は「これらが揃えば--labelも付くはず」という意味なので、
            # 揃っていて付いていない行が違反。多数派か少数派かは関係ない。
            print(f"\n【見直しの候補】{names}が揃っていて{LABEL_JA[args.label]}が"
                  f"付いていない{without}件")
            for filename, labels in zip(df["filename"], sets):
                if expected <= labels and args.label not in labels:
                    print(f"  {filename}  {'|'.join(sorted(labels))}")
            # 対象ラベルが付いていて、組み合わせの片方だけを伴う例も揺れている
            partial = [
                (filename, labels) for filename, labels in zip(df["filename"], sets)
                if args.label in labels and 0 < len(expected & labels) < len(expected)
            ]
            if partial:
                print(f"\n【同じく候補】{LABEL_JA[args.label]}に、"
                      f"{names}の片方だけを伴う{len(partial)}件")
                for filename, labels in partial:
                    print(f"  {filename}  {'|'.join(sorted(labels))}")

        if args.out_csv:
            # 2種類の候補は作業が違う。混ぜて渡すと、片方に不要な判定をさせてしまう。
            #   needs_judgement … そのラベルを付けるべきか、天気図を見て決める必要がある
            #   rule_only       … 付けるべきなのは確定していて、規約に合わせるだけ
            rows = [
                {"filename": filename, "label": "|".join(sorted(labels)),
                 "kind": "needs_judgement"}
                for filename, labels in zip(df["filename"], sets)
                if expected <= labels and args.label not in labels
            ] + [
                {"filename": filename, "label": "|".join(sorted(labels)), "kind": "rule_only"}
                for filename, labels in zip(df["filename"], sets)
                if args.label in labels and 0 < len(expected & labels) < len(expected)
            ]
            out = Path(args.out_csv)
            out.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8")
            print(f"\n見直しの候補{len(rows)}件を書き出しました: {out}")

if __name__ == "__main__":
    main()
