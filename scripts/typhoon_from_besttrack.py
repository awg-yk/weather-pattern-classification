"""気象庁のベストトラックから、台風ラベルを機械的に作る。

天気図を見た人の判定は揺れる。同じ2024年9月を1年以上あけて判定し直したところ、
既存の12日に対して30日が「あり」になり、κは0.400だった -- 難しいからではなく、
「どこまでを日本への影響とみなすか」が書かれていなかったから。

ベストトラックは中心位置と強度が6時間ごとに記録された数値データで、
**ERA5とも天気図とも独立している**。どちらのモデルにも不当な有利を与えずに、
台風だけは人手の揺れを完全に排除できる。1951年以降が公開されているので、
2000〜2022年の天気図にもそのまま展開できる。

判定は「台風の中心が、指定した緯度経度の範囲にあるか」だけ。
「日本に影響しているか」という主観的な問いを、範囲の指定で置き換える。

データの取得(この環境では取れないので手元で):
    https://www.jma.go.jp/jma/jma-eng/jma-center/rsmc-hp-pub-eg/Besttracks/bst_all.txt

使い方:
    # 既存ラベルとどれだけ一致するかを見る(書き換えない)
    python -m scripts.typhoon_from_besttrack --besttrack data/raw/bst_all.txt \\
        --labels data/labels_v2.csv

    # 範囲を変えて、どの範囲だと既存と最も合うかを探る
    python -m scripts.typhoon_from_besttrack --besttrack data/raw/bst_all.txt \\
        --labels data/labels_v2.csv --sweep

    # 反映する(元のファイルは .bak に残す)
    python -m scripts.typhoon_from_besttrack --besttrack data/raw/bst_all.txt \\
        --labels data/labels_v2.csv --apply
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd

from src.labels import parse_labels

# 気象庁ベストトラックのgrade。3=TS, 4=STS, 5=TY が「台風」。
# 2と9はTD(台風に満たない)、6は温帯低気圧化後、7は責任領域に入った直後。
TYPHOON_GRADES = {3, 4, 5}

# 既定の範囲: 本土・南西諸島・小笠原・南鳥島を含む
DEFAULT_REGION = (20.0, 46.0, 123.0, 154.0)  # 南, 北, 西, 東


def parse_besttrack(path: Path, grades=TYPHOON_GRADES) -> pd.DataFrame:
    """(時刻, 緯度, 経度)のDataFrameを返す。指定した階級の記録だけ。

    ヘッダ行は '66666' で始まる。データ行は空白区切りで
    YYMMDDHH / 002 / grade / 緯度(0.1度) / 経度(0.1度) / 中心気圧 / ...
    と並ぶ。桁位置ではなく空白で分けるのは、年代によって桁揃えが揺れるため。
    """
    records = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("66666") or not line.strip():
            continue
        parts = line.split()
        if len(parts) < 5 or len(parts[0]) != 8 or not parts[0].isdigit():
            continue
        try:
            grade = int(parts[2])
            latitude, longitude = int(parts[3]) / 10.0, int(parts[4]) / 10.0
        except ValueError:
            continue
        if grade not in grades:
            continue
        stamp = parts[0]
        # ベストトラックは1951年から。YYが51以上なら19xx、それ未満なら20xx。
        year = int(stamp[:2])
        year += 1900 if year >= 51 else 2000
        records.append({
            "datetime": pd.Timestamp(year=year, month=int(stamp[2:4]),
                                     day=int(stamp[4:6]), hour=int(stamp[6:8])),
            "lat": latitude, "lon": longitude, "grade": grade,
        })
    if not records:
        raise SystemExit(f"{path} から台風の記録を読み取れませんでした。形式を確認してください。")
    return pd.DataFrame(records)


def stamps_with_typhoon(track: pd.DataFrame, region) -> set:
    """範囲内に台風の中心があった時刻の集合。"""
    south, north, west, east = region
    inside = track[(track["lat"] >= south) & (track["lat"] <= north)
                   & (track["lon"] >= west) & (track["lon"] <= east)]
    return set(inside["datetime"])


def compare(df: pd.DataFrame, derived: pd.Series) -> dict:
    """既存ラベルと機械ラベルの一致を数える。"""
    existing = df["parsed_labels"].apply(lambda labels: "typhoon" in labels)
    both = int((existing & derived).sum())
    only_existing = int((existing & ~derived).sum())
    only_derived = int((~existing & derived).sum())
    neither = int((~existing & ~derived).sum())
    total = len(df)
    observed = (both + neither) / total
    expected = (((both + only_existing) / total) * ((both + only_derived) / total)
                + ((only_derived + neither) / total) * ((only_existing + neither) / total))
    kappa = (observed - expected) / (1 - expected) if expected < 1 else 1.0
    precision = both / (both + only_derived) if both + only_derived else 0.0
    recall = both / (both + only_existing) if both + only_existing else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"both": both, "only_existing": only_existing, "only_derived": only_derived,
            "neither": neither, "agreement": observed, "kappa": kappa, "f1": f1,
            "precision": precision, "recall": recall}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--besttrack", required=True, help="bst_all.txt のパス")
    parser.add_argument("--labels", required=True)
    parser.add_argument(
        "--region", type=float, nargs=4, default=list(DEFAULT_REGION),
        metavar=("SOUTH", "NORTH", "WEST", "EAST"),
        help=f"台風の中心がこの範囲にあれば「あり」。既定 {DEFAULT_REGION}"
        "(本土・南西諸島・小笠原・南鳥島を含む)",
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help="範囲をいくつか振って、既存ラベルとの一致を比べる。"
        "最初にどういう基準で付けていたかを逆算できる",
    )
    parser.add_argument("--apply", action="store_true", help="ラベルCSVを実際に書き換える")
    args = parser.parse_args()

    track = parse_besttrack(Path(args.besttrack))
    print(f"ベストトラック: 台風(TS以上)の記録 {len(track)}件"
          f" ({track['datetime'].min():%Y-%m-%d}〜{track['datetime'].max():%Y-%m-%d})")

    path = Path(args.labels)
    df = pd.read_csv(path)
    df["parsed_labels"] = df["label"].apply(parse_labels)
    stamps = df["filename"].str.extract(r"(\d{10})")[0]
    df["datetime"] = pd.to_datetime(stamps, format="%Y%m%d%H", errors="coerce")
    if df["datetime"].isna().any():
        raise SystemExit("ファイル名から日時を取り出せない行があります")
    print(f"ラベル: {len(df)}件 ({df['datetime'].min():%Y-%m-%d}〜{df['datetime'].max():%Y-%m-%d})")

    if args.sweep:
        # 南端と東端を振る。諸島をどこまで含めるかが判定を最も大きく変えるため。
        print(f"\n{'範囲(南,北,西,東)':<28}{'機械=あり':>10}{'一致率':>9}{'κ':>8}{'F1':>8}")
        print("-" * 63)
        for region in [(30, 46, 128, 146), (24, 46, 123, 149), (20, 46, 123, 154),
                       (20, 50, 120, 160), (15, 50, 115, 165)]:
            derived = df["datetime"].isin(stamps_with_typhoon(track, region))
            result = compare(df, derived)
            print(f"{str(region):<28}{int(derived.sum()):>10}{result['agreement']:>8.1%}"
                  f"{result['kappa']:>8.3f}{result['f1']:>8.3f}")
        print("\n  既存ラベルの「あり」は "
              f"{int(df['parsed_labels'].apply(lambda l: 'typhoon' in l).sum())}件。")
        print("  κが最も高い範囲が、最初に付けていた基準に最も近い。")
        return

    region = tuple(args.region)
    derived = df["datetime"].isin(stamps_with_typhoon(track, region))
    result = compare(df, derived)

    print(f"\n範囲: 北緯{region[0]}〜{region[1]}度 / 東経{region[2]}〜{region[3]}度")
    print(f"  機械が「あり」とした: {int(derived.sum())}件")
    print(f"  既存ラベルの「あり」: {int(df['parsed_labels'].apply(lambda l: 'typhoon' in l).sum())}件")
    print(f"\n                    機械 あり    機械 なし")
    print(f"  既存 あり      {result['both']:>10}{result['only_existing']:>13}")
    print(f"  既存 なし      {result['only_derived']:>10}{result['neither']:>13}")
    print(f"\n  一致率 {result['agreement']:.1%} / κ {result['kappa']:.3f} / F1 {result['f1']:.3f}")

    if not args.apply:
        print("\n何も書き換えていません。反映するには --apply を付けてください。")
        return

    changed = 0
    new_values = []
    for labels, has_typhoon in zip(df["parsed_labels"], derived):
        updated = sorted(set(labels) | {"typhoon"}) if has_typhoon else \
            sorted(set(labels) - {"typhoon"})
        if sorted(labels) != updated:
            changed += 1
        new_values.append("|".join(updated))

    backup = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, backup)
    df["label"] = new_values
    df.drop(columns=["parsed_labels", "datetime"]).to_csv(path, index=False, encoding="utf-8")
    print(f"\n{changed}件を書き換えました: {path}")
    print(f"元のファイルは {backup} に残しています。")
    print("この範囲の定義を src/labels.py に書き残すこと。")


if __name__ == "__main__":
    main()
