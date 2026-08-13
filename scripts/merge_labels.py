"""
labels.csvの中の特定ラベルを別のラベルに統合するマイグレーションスクリプト。

例: zonal_high(帯状高気圧) を migratory_high(移動性高気圧) に統合する場合、
「zonal_high|typhoon」は「migratory_high|typhoon」に、
「migratory_high|zonal_high」は重複を除いて「migratory_high」になる。

実行前に自動でバックアップ(.bak)を作成する。

使い方:
    python scripts/merge_labels.py --labels data/labels.csv --from zonal_high --to migratory_high
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd


def merge_label(label_field: str, from_label: str, to_label: str) -> str:
    labels = str(label_field).split("|")
    labels = [to_label if l == from_label else l for l in labels]
    # 統合の結果、同じラベルが重複したら1つにまとめる(順序は保持)
    seen = []
    for l in labels:
        if l not in seen:
            seen.append(l)
    return "|".join(seen)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True, help="labels.csvのパス")
    parser.add_argument("--from", dest="from_label", required=True, help="統合元のラベル")
    parser.add_argument("--to", dest="to_label", required=True, help="統合先のラベル")
    args = parser.parse_args()

    labels_path = Path(args.labels)
    backup_path = labels_path.with_suffix(labels_path.suffix + ".bak")
    shutil.copy(labels_path, backup_path)
    print(f"バックアップを作成しました: {backup_path}")

    df = pd.read_csv(labels_path)
    n_affected = df["label"].apply(lambda s: args.from_label in str(s).split("|")).sum()

    df["label"] = df["label"].apply(lambda s: merge_label(s, args.from_label, args.to_label))
    df.to_csv(labels_path, index=False)

    print(f"{n_affected}件の行で '{args.from_label}' を '{args.to_label}' に統合しました。")
    print(f"保存先: {labels_path}")


if __name__ == "__main__":
    main()
