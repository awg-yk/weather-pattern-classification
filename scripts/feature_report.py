"""どの特徴量がどのラベルの手がかりになっているかを測る。

`scripts/cv_features.py` の結果が期待と違ったとき、原因を特徴量まで
遡って見るための道具。macro F1 が下がったとき、それが「特徴量に信号が
無い」のか「信号はあるが分類器が使えていない」のかは、成績だけでは分からない。

指標は AUC。しきい値を決めずに順位だけで測るので、値の単位に依らない。
0.5 が当てずっぽう、1.0 が完全な分離、0.0 は逆向きの完全な分離で、
**0.5からの距離が信号の強さ**である。

学習はしない。特徴量そのものがラベルをどれだけ分けられるかだけを見る。
**この数字を報告用の成績として使ってはいけない。**

使い方:
    python -m scripts.feature_report --features data\\features.csv --top 6
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.chartfeatures import feature_names
from src.chartsymbols import label_auc
from src.labels import LABELS, LABEL_JA
from src.regions import load_regions

# これ未満しか0.5から離れていない特徴量は、手がかりとして扱わない。
WEAK_AUC = 0.05

KEY_COLUMNS = ("filename", "date", "parsed_datetime")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--features", required=True)
    parser.add_argument("--labels", default="data/labels_v2.csv")
    parser.add_argument("--top", type=int, default=5, help="ラベルごとに何個まで出すか")
    parser.add_argument("--min-positive", type=int, default=20,
                        help="これ未満の陽性しか無いラベルは飛ばす")
    args = parser.parse_args()

    features = pd.read_csv(args.features)
    labels = pd.read_csv(args.labels)
    merged = labels.merge(features, on="filename", how="inner")
    if merged.empty:
        raise SystemExit("特徴量とラベルが突き合いませんでした。")

    columns = [c for c in features.columns if c not in KEY_COLUMNS]
    print(f"{len(merged)}件、特徴量{len(columns)}個\n")

    print(f"{'ラベル':<24} {'陽性':>5}  手がかりになっている特徴量 (AUC)")
    print("-" * 88)
    weak_labels = []
    for label in LABELS:
        has = [label in str(t).split("|") for t in merged["label"].fillna("")]
        n_pos = sum(has)
        if n_pos < args.min_positive or n_pos == len(has):
            print(f"{LABEL_JA[label]:<24} {n_pos:>5}  (件数が足りず測れない)")
            continue

        scored = []
        for name in columns:
            values = merged[name].to_numpy(dtype=float)
            # 欠測は中央値で埋める。AUCは順位で測るので、欠測を極端な値に
            # すると「欠測かどうか」を見ているだけの偽の信号が出る
            if np.isnan(values).any():
                values = np.where(np.isnan(values), np.nanmedian(values), values)
            auc = label_auc(list(values), has)
            if auc == auc:
                scored.append((abs(auc - 0.5), auc, name))
        scored.sort(reverse=True)

        best = scored[:args.top]
        if not best or best[0][0] < WEAK_AUC:
            weak_labels.append(label)
            print(f"{LABEL_JA[label]:<24} {n_pos:>5}  ★手がかりが無い"
                  f"(最大でも {best[0][1]:.3f} など)" if best else "")
            continue
        detail = "  ".join(f"{name} {auc:.3f}" for _, auc, name in best)
        print(f"{LABEL_JA[label]:<24} {n_pos:>5}  {detail}")

    print()
    if weak_labels:
        print("★の付いたラベルは、今の特徴量に手がかりが無い。分類器を変えても上がらない。")
        print("  " + "、".join(LABEL_JA[l] for l in weak_labels))
    print("AUCは0.5からの距離が信号の強さ。0.5未満は逆向きの関係で、同じだけ手がかりになる。")
    print("学習をしていないので、報告用の成績として使ってはいけない。")


if __name__ == "__main__":
    main()
