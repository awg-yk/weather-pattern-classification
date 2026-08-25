"""盲検レビュー用に、陽性と陰性を同数ずつ抜き出す。

■ なぜ無作為の1か月分では足りないか

人間の再現性(同じ人が同じ天気図を2回判定したときの一致)を測るには、
陽性がある程度の枚数が要る。ところが無作為に60枚見ても、二つ玉低気圧は
2.8枚、太平洋高気圧型は6.1枚しか含まれない(2432件中112件・248件)。
そこから出したκやF1は、1枚の判断が変わるだけで大きく振れる。

そこで陽性と陰性を同数ずつ抜く。60枚見る労力は同じまま、二つ玉低気圧の
陽性が3枚から30枚になる。

■ 出現率をいじった分は、集計側で戻す

層別抽出は出現率を人為的に変えるので、そのままのκ・F1は他のラベルと
比較できない(κもF1も出現率に依存する)。集計は
scripts/compare_review.py --stratified を使うこと。感度と特異度という
出現率に依存しない2つの量を測り、本来の出現率で組み直す。

■ 盲検は保たれる

書き出すのはファイル名だけで、どちらの層から来たかは記録しない。
どの層だったかは元のラベルCSVから分かるので、記録する必要がない。
scripts/review_cli.py はファイル名順(=日付順)に見せるため、
並び順から層が透けることもない。

使い方:
    python -m scripts.sample_review --labels data/labels_v2.csv \
        --label futatsudama_low --out data/review_futatsudama_sample.csv

    # そのまま判定に進む
    python -m scripts.review_cli --images-dir <画像> --label futatsudama_low \
        --candidates data/review_futatsudama_sample.csv \
        --out-csv data/review_futatsudama_2nd.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.labels import LABEL_JA, LABELS, parse_labels



def _print_precision_forecast(n_pos: int, n_neg: int, prevalence: float) -> None:
    """この枚数で、F1の上限をどれくらいの精度で出せるかを先に示す。

    判定を終えてから「枚数が足りなかった」と分かるのは無駄が大きい。
    感度0.85・特異度0.95を仮に置いて、区間の幅を見積もる(実際の値は判定するまで
    分からないので、あくまで見当をつけるためのもの)。
    """
    from scripts.compare_review import uncertainty

    def width(np_, nn_):
        u = uncertainty(round(np_ * 0.85), np_, round(nn_ * 0.95), nn_,
                        prevalence, draws=4000, seed=1)
        return u["f1_high"] - u["f1_low"]

    here = width(n_pos, n_neg)
    print(f"\n  この枚数で見込まれるF1の95%区間の幅: 約{here:.2f}"
          "(感度0.85・特異度0.95と仮定した見積もり)")
    if here <= 0.15:
        print("  → 十分です。")
        return

    print("  → 広すぎます。枚数を変えたときの見込み:")
    print(f"      {'陽性':>6}{'陰性':>6}{'区間幅':>10}{'合計':>8}")
    for np_, nn_ in ((n_pos, n_neg), (n_pos * 2, n_neg), (n_pos, n_neg * 2),
                     (n_pos, n_neg * 4), (n_pos * 2, n_neg * 4)):
        print(f"      {np_:>6}{nn_:>6}{width(np_, nn_):>10.3f}{np_ + nn_:>8}")
    if prevalence < 0.15:
        print("  出現率が低いラベルでは、陽性を増やしても区間はほとんど狭まりません。")
        print("  適合率が 偽陽性=(1-出現率)×(1-特異度) に支配されるためです。")
        print("  感度(再現率の上限)だけなら陽性30枚で十分に測れるので、"
              "F1の上限は諦めて感度だけ報告する、という判断もあります。")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--label", required=True, choices=list(LABELS))
    parser.add_argument("--out", required=True, help="候補の書き出し先CSV")
    parser.add_argument(
        "--n-positive", type=int, default=30,
        help="そのラベルが付いている行から何枚抜くか。少ないと感度の推定が粗くなる",
    )
    parser.add_argument(
        "--n-negative", type=int, default=30,
        help="付いていない行から何枚抜くか。特異度の推定に使う",
    )
    parser.add_argument("--years", type=int, nargs="+", default=None, help="対象の年")
    parser.add_argument("--months", type=int, nargs="+", default=None, help="対象の月")
    parser.add_argument(
        "--exclude", nargs="+", default=None,
        help="既存の見直し結果CSV。そこに出てくる画像は抜かない(同じ枚を2度見ないため)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.labels)
    df["parsed"] = df["label"].apply(parse_labels)
    df["positive"] = df["parsed"].apply(lambda ls: args.label in ls)

    stamps = pd.to_datetime(
        df["filename"].str.extract(r"(\d{10})")[0], format="%Y%m%d%H", errors="coerce"
    )
    keep = stamps.notna()
    if args.years:
        keep &= stamps.dt.year.isin(set(args.years))
    if args.months:
        keep &= stamps.dt.month.isin(set(args.months))
    pool = df[keep]

    # 出現率は「抽出した範囲」で測る。集計側が本来の出現率に戻すときに使う値なので、
    # 抽出範囲と食い違うと復元がずれる。
    prevalence = float(pool["positive"].mean())

    if args.exclude:
        seen = set()
        for path in args.exclude:
            seen |= set(pd.read_csv(path)["filename"])
        before = len(pool)
        pool = pool[~pool["filename"].isin(seen)]
        print(f"既に見た{before - len(pool)}枚を除外しました")

    positives = pool[pool["positive"]]
    negatives = pool[~pool["positive"]]
    n_pos = min(args.n_positive, len(positives))
    n_neg = min(args.n_negative, len(negatives))
    if n_pos < args.n_positive:
        print(f"⚠ 陽性が{len(positives)}枚しかないため{n_pos}枚にしました")
    if n_pos == 0:
        raise SystemExit(f"{LABEL_JA[args.label]} の陽性が範囲内にありません。")

    rng = np.random.default_rng(args.seed)
    picked = pd.concat([
        positives.sample(n=n_pos, random_state=rng.integers(2**31)),
        negatives.sample(n=n_neg, random_state=rng.integers(2**31)),
    ])

    # review_cli.py は kind 列で絞り込む。全行を対象にしたいので同じ値を入れる。
    out = pd.DataFrame({
        "filename": sorted(picked["filename"]),   # 日付順。層は並びから分からない
        "kind": "needs_judgement",
    })
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"\n{LABEL_JA[args.label]}: 陽性{n_pos}枚 + 陰性{n_neg}枚 = {len(out)}枚を抽出")
    print(f"  抽出範囲の出現率: {prevalence:.3%}"
          f"({int(pool['positive'].sum())}/{len(pool)}件)")
    print(f"  書き出し: {out_path}")

    _print_precision_forecast(n_pos, n_neg, prevalence)
    print("\n次の手順:")
    print(f"  1) python -m scripts.review_cli --images-dir <画像> --label {args.label} \\")
    print(f"         --candidates {out_path} --out-csv <判定結果.csv>")
    print(f"  2) python -m scripts.compare_review --review <判定結果.csv> \\")
    print(f"         --labels {args.labels} --label {args.label} --stratified")
    print("\n  2)で --stratified を付け忘れると、陽性を多く含む集合のまま集計され、")
    print("  他のラベルと比較できない数字になります。")


if __name__ == "__main__":
    main()
