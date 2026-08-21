"""交差検証を始める前に、その設定で本当に走るかを数秒で確かめる。

1foldに数十分かかるので、途中で落ちたり、落ちないまま無意味な結果が出たりすると
損失が大きい。ここで確かめるのは、学習を回さなくても分かることだけ:

  - 指定したラベルファイルが読め、どのラベルが何件あるか
  - 天気図の画像が全行そろっているか(ファイル名の表記ゆれを吸収して照合する)
  - ERA5のnetCDFが対象年ぶんあり、全行の時刻が含まれているか
  - foldごとの train/val/test の件数
  - **検証データに1件も出ないラベルがないか**

最後が特に効く。検証データが季節に偏っていると、そこに出ないラベルについては
モデル選択も閾値も重みも決められず、静かにでたらめになる。実際 okhotsk_high は
初夏の型で、既定の--val-mode tail では検証(8〜12月)にほとんど現れない。

使い方(cross_validate.py と同じ引数を渡す):
    python -m scripts.preflight --data-dir <画像> --labels data/labels_v2.csv \
        --years 2023 2024 2025
    python -m scripts.preflight --input-mode era5-grid --labels data/labels_v2.csv \
        --years 2023 2024 2025
"""

import argparse
from pathlib import Path

import pandas as pd

from src.era5 import open_era5
from src.labels import LABEL_JA, LABELS, parse_labels
from src.split import make_splits

NUM_CHANNELS_FILES = (("mslp", "msl"), ("t850", "t"))


def load_rows(labels_csv, years):
    df = pd.read_csv(labels_csv)
    for column in ("filename", "label"):
        if column not in df.columns:
            raise SystemExit(f"{labels_csv} に {column} 列がありません(列: {list(df.columns)})")
    df["parsed_labels"] = df["label"].apply(parse_labels)
    unrecognised = int((df["parsed_labels"].apply(len) == 0).sum())
    df = df[df["parsed_labels"].apply(len) > 0].reset_index(drop=True)
    stamps = df["filename"].str.extract(r"(\d{10})")[0]
    df["parsed_datetime"] = pd.to_datetime(stamps, format="%Y%m%d%H", errors="coerce")
    if df["parsed_datetime"].isna().any():
        bad = df.loc[df["parsed_datetime"].isna(), "filename"].head(3).tolist()
        raise SystemExit(f"ファイル名から日時を取り出せない行があります: {bad}")
    total = len(df)
    df = df[df["parsed_datetime"].dt.year.isin({int(y) for y in years})].reset_index(drop=True)
    return df, unrecognised, total


def check_charts(df, data_dir) -> list:
    from src.dataset import index_images_by_stamp

    directory = Path(data_dir)
    if not directory.exists():
        return [f"--data-dir {directory} がありません"]
    available = index_images_by_stamp(directory)
    missing = [
        row.filename for row in df.itertuples()
        if row.parsed_datetime.strftime("%Y%m%d%H") not in available
    ]
    print(f"  天気図: ディレクトリ内 {len(available)}件 / 必要 {len(df)}件")
    if missing:
        return [f"画像が{len(missing)}件足りません(例: {missing[:3]})"]
    return []


def check_era5(df, era5_dir) -> list:
    problems = []
    directory = Path(era5_dir)
    for year in sorted(df["parsed_datetime"].dt.year.unique()):
        needed = set(df.loc[df["parsed_datetime"].dt.year == year, "parsed_datetime"])
        for suffix, variable in NUM_CHANNELS_FILES:
            path = directory / f"era5_{suffix}_{year}.nc"
            if not path.exists():
                problems.append(f"{path} がありません")
                continue
            with open_era5(path) as ds:
                if variable not in ds:
                    problems.append(f"{path} に変数 {variable} がありません(あるのは {list(ds.data_vars)})")
                    continue
                name = "valid_time" if "valid_time" in ds.coords else "time"
                have = {pd.Timestamp(v) for v in ds[name].values}
            absent = needed - have
            print(f"  ERA5 {path.name}: 時刻{len(have)}件 / 必要{len(needed)}件"
                  + (f"  ← {len(absent)}件不足" if absent else ""))
            if absent:
                problems.append(
                    f"{path.name} に{len(absent)}件の時刻がありません"
                    f"(例: {sorted(absent)[:2]})"
                )
    return problems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--input-mode", default="chart", choices=["chart", "era5-grid"])
    parser.add_argument("--era5-grid-dir", default="data/raw/era5")
    parser.add_argument("--years", type=int, nargs="+", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--gap-days", type=int, default=3)
    parser.add_argument("--val-mode", default="tail", choices=["spread", "tail"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"設定: {args.input_mode} / labels={args.labels} / years={args.years}"
          f" / val-mode={args.val_mode}")

    if args.input_mode == "chart" and not args.data_dir:
        raise SystemExit("--input-mode chart には --data-dir が必要です")

    df, unrecognised, before = load_rows(args.labels, args.years)
    print(f"\n【ラベルファイル】{args.labels}")
    print(f"  認識できるラベルを持つ行 {before}件"
          + (f" (認識できないラベルの行を{unrecognised}件除外)" if unrecognised else ""))
    print(f"  対象年に絞ると {len(df)}件")
    if df.empty:
        raise SystemExit(f"対象年 {args.years} に該当する行がありません")

    counts = {label: sum(label in ls for ls in df["parsed_labels"]) for label in LABELS}
    print("\n【ラベルごとの件数】")
    for label in LABELS:
        share = counts[label] / len(df)
        print(f"  {LABEL_JA[label]:<24}{counts[label]:>6}件  ({share:>5.1%})")
    absent_labels = [l for l in LABELS if counts[l] == 0]

    print(f"\n【入力データ】")
    problems = (
        check_charts(df, args.data_dir) if args.input_mode == "chart"
        else check_era5(df, args.era5_grid_dir)
    )

    print(f"\n【foldごとの分割】")
    empty_in_val = []
    for test_year in args.years:
        splits = make_splits(
            df, mode="loyo", val_ratio=args.val_ratio, test_ratio=0.0,
            seed=args.seed, test_year=test_year, gap_days=args.gap_days, val_mode=args.val_mode,
        )
        sizes = {k: len(v) for k, v in splits.items()}
        months = sorted({d.month for d in df.loc[splits["val"], "parsed_datetime"]})
        print(f"  テスト={test_year}年: train {sizes['train']} / val {sizes['val']} / test {sizes['test']}"
              f"   検証が含む月 {months}")
        for label in LABELS:
            if counts[label] == 0:
                continue
            in_val = sum(label in ls for ls in df.loc[splits["val"], "parsed_labels"])
            in_test = sum(label in ls for ls in df.loc[splits["test"], "parsed_labels"])
            if in_val == 0:
                empty_in_val.append((test_year, label, in_test))

    if empty_in_val:
        print(f"\n  警告: 検証データに1件も現れないラベルがあります。"
              "そのラベルの閾値・モデル選択は決めようがありません。")
        for test_year, label, in_test in empty_in_val:
            print(f"    テスト={test_year}年: {LABEL_JA[label]}"
                  f" (検証0件だがテストには{in_test}件)")
        print("    --val-mode spread にすると検証が通年になります。")

    print(f"\n{'=' * 60}")
    if absent_labels:
        print(f"注意: 対象年に1件も出現しないラベル: "
              f"{', '.join(LABEL_JA[l] for l in absent_labels)}")
    if problems:
        print("実行できません:")
        for problem in problems:
            print(f"  - {problem}")
        raise SystemExit(1)
    print("この設定で交差検証を開始できます。")


if __name__ == "__main__":
    main()
