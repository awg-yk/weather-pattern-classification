"""分割方式ごとの時間的リークの量を測る診断スクリプト。

天気図は隣り合う日どうしが極めて似ているため、日付を無視してランダムに分割すると
「検証セットの画像と1日違いの画像で学習している」状態になりやすい。それが実際に
どの程度起きているかを、分割方式ごとに数字で示す。

使い方:
    python -m scripts.check_leakage --labels data/labels.csv
"""

import argparse

import pandas as pd

from src.dataset import parse_datetime
from src.labels import LABEL_TO_INDEX
from src.split import make_splits, min_days_to_other_split


def summarize(df: pd.DataFrame, mode: str, val_ratio: float, test_ratio: float, seed: int) -> None:
    print(f"\n{'=' * 62}\n--split-mode {mode}\n{'=' * 62}")
    try:
        splits = make_splits(df, mode=mode, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)
    except ValueError as e:
        print(f"  分割できません: {e}")
        return

    for name, reference in (("val", "train"), ("test", "train+val")):
        rows = splits[name]
        others = splits["train"] if reference == "train" else splits["train"] + splits["val"]
        gaps = min_days_to_other_split(df, rows, others)
        if gaps.empty:
            print(f"  {name}: なし")
            continue
        print(
            f"  {name}({len(rows)}件) の各画像から、最も日付が近い{reference}画像までの日数:\n"
            f"      中央値 {gaps.median():>6.0f}日   最小 {gaps.min():>4.0f}日\n"
            f"      1日以内 {(gaps <= 1).mean():>6.1%}   3日以内 {(gaps <= 3).mean():>6.1%}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True, help="labels.csvのパス")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw = pd.read_csv(args.labels)
    # WeatherMapDatasetと同じ絞り込み(既知ラベルが1つも付いていない行は学習に使われない)
    known = raw["label"].apply(
        lambda s: any(l in LABEL_TO_INDEX for l in str(s).split("|"))
    )
    df = raw[known].reset_index(drop=True)
    df["parsed_datetime"] = [
        parse_datetime(fn, dt) for fn, dt in zip(df["filename"], df.get("date", [None] * len(df)))
    ]

    print(f"学習に使われる行: {len(df)}件(labels.csv全体は{len(raw)}件)")
    undated = df["parsed_datetime"].isna().sum()
    if undated:
        print(f"警告: 日付を取り出せない行が{undated}件あります")

    dated = df["parsed_datetime"].dropna()
    if not dated.empty:
        print(f"期間: {dated.min():%Y-%m-%d} 〜 {dated.max():%Y-%m-%d}")
        per_year = dated.dt.year.value_counts().sort_index()
        print("年ごとの件数: " + ", ".join(f"{int(y)}:{c}" for y, c in per_year.items()))

    for mode in ("random", "temporal", "by_year"):
        summarize(df, mode, args.val_ratio, args.test_ratio, args.seed)

    print(
        "\n読み方: randomで「1日以内」の割合が高いほど、検証・テストの画像とほぼ同じ日の"
        "\n画像で学習していることになり、報告スコアが実力より高く出る。"
    )


if __name__ == "__main__":
    main()
