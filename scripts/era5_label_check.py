"""ERA5の領域特徴だけでラベルをどこまで説明できるかを、CNNに組み込む前に確かめる。

いきなりCNNに組み込むと学習に1時間かかるうえ、効かなかったときに
「特徴量が悪いのか、組み込み方が悪いのか」を切り分けられない。
ここでロジスティック回帰(単純な線形モデル)に同じ特徴量を与えて測っておけば、
その特徴量がラベルの情報をどれだけ持っているかが数秒で分かる。

評価は交差検証と同じleave-one-year-outで行う。年をまたいで学習・評価するので、
隣接日のリークは起きない。

出力の見方:
    AUC  … その特徴量群で陽性と陰性をどれだけ順序づけられるか。0.5=無情報、1.0=完全
    F1   … 実際に閾値を切って分類したときの成績。CNNの数値と直接比較できる

使い方:
    python -m scripts.era5_label_check --features data/era5_features.csv \
        --labels data/labels.csv
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.dataset import parse_labels
from src.labels import LABEL_JA, LABELS

# 参考として並べる、CNN(CoordConvあり)のleave-one-year-out交差検証の結果
CNN_F1 = {
    "winter_pressure_pattern": 0.769, "nankigan_low": 0.633, "japan_sea_low": 0.668,
    "futatsudama_low": 0.497, "typhoon": 0.745, "migratory_high": 0.725,
    "pacific_high": 0.558, "front_passage": 0.740, "stationary_front": 0.726,
    "okhotsk_high": 0.240,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/era5_features.csv")
    parser.add_argument("--labels", default="data/labels.csv")
    parser.add_argument("--years", type=int, nargs="+", default=[2023, 2024, 2025])
    parser.add_argument(
        "--detail",
        default=None,
        help="1つのラベルについて、全特徴量の係数と単独AUCを表示する(例: okhotsk_high)",
    )
    args = parser.parse_args()

    feats = pd.read_csv(args.features)
    labels = pd.read_csv(args.labels)
    labels["parsed_labels"] = labels["label"].apply(parse_labels)
    labels = labels[labels["parsed_labels"].apply(len) > 0]

    df = labels.merge(feats, on="filename", how="inner")
    if df.empty:
        raise SystemExit(
            "ラベルとERA5特徴量が1件も結合できませんでした。\n"
            f"  labels.csvのfilename例  : {labels['filename'].iloc[0]!r}\n"
            f"  特徴量CSVのfilename例   : {feats['filename'].iloc[0]!r}\n"
            "  --filename-format を合わせて era5_features を作り直してください。"
        )
    df["year"] = pd.to_datetime(df["datetime"]).dt.year
    df = df[df["year"].isin(args.years)].reset_index(drop=True)
    print(f"ラベル{len(labels)}件 / 特徴量{len(feats)}件 → 結合{len(df)}件\n")

    feature_cols = [c for c in feats.columns if c not in ("filename", "datetime")]
    X = df[feature_cols].to_numpy(dtype="float64")

    print(f"  {'ラベル':<22}{'AUC':>8}{'F1':>8}{'CNNのF1':>10}{'差':>8}")
    print("  " + "-" * 56)
    rows = []
    for label in LABELS:
        y = df["parsed_labels"].apply(lambda ls, l=label: l in ls).to_numpy(dtype=int)

        preds = np.zeros(len(df))
        scores = np.zeros(len(df))
        for year in args.years:
            test = (df["year"] == year).to_numpy()
            if y[~test].sum() < 2 or y[test].sum() < 1:
                continue  # そのfoldに陽性がほとんど無い場合は測れない
            model = make_pipeline(
                StandardScaler(),
                # 陽性が少ないラベルでも無視されないよう重みを釣り合わせる
                LogisticRegression(max_iter=2000, class_weight="balanced"),
            ).fit(X[~test], y[~test])
            scores[test] = model.predict_proba(X[test])[:, 1]
            preds[test] = model.predict(X[test])

        auc = roc_auc_score(y, scores) if len(set(y)) > 1 else float("nan")
        f1 = f1_score(y, preds, zero_division=0)
        cnn = CNN_F1.get(label, float("nan"))
        print(f"  {LABEL_JA[label]:<22}{auc:>8.3f}{f1:>8.3f}{cnn:>10.3f}{f1 - cnn:>+8.3f}")
        rows.append({"label": label, "auc": auc, "f1": f1, "cnn_f1": cnn, "support": int(y.sum())})

    if args.detail:
        if args.detail not in LABELS:
            raise SystemExit(f"--detail は {LABELS} のいずれかを指定してください")
        _detail(args.detail, X, df, feature_cols)
        return

    summary = pd.DataFrame(rows)
    print(f"\n  {'macro F1(ERA5のみ)':<22}{summary['f1'].mean():>16.3f}")
    print(f"  {'macro F1(CNN)':<22}{summary['cnn_f1'].mean():>16.3f}")

    # どの特徴量がどのラベルに効いているかを見る(全データで学習した係数の大きさ)
    print("\n--- ラベルごとに効いている特徴量(上位3件) ---")
    for label in LABELS:
        y = df["parsed_labels"].apply(lambda ls, l=label: l in ls).to_numpy(dtype=int)
        if y.sum() < 5:
            continue
        model = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")
        ).fit(X, y)
        coef = model[-1].coef_[0]
        top = np.argsort(-np.abs(coef))[:3]
        parts = [f"{feature_cols[i]}({coef[i]:+.2f})" for i in top]
        print(f"  {LABEL_JA[label]:<22}{'  '.join(parts)}")


def _detail(label, X, df, feature_cols):
    """1ラベルについて、特徴量ごとの寄与を詳しく出す。

    係数は他の特徴量と一緒に使ったときの寄与、単独AUCはその特徴量だけを
    見たときの判別力。両方見ないと「他と組んで初めて効く量」を見落とす。
    """
    y = df["parsed_labels"].apply(lambda ls, l=label: l in ls).to_numpy(dtype=int)
    model = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")
    ).fit(X, y)
    coef = model[-1].coef_[0]

    print(f"\n{'=' * 62}\n{LABEL_JA[label]}({label}) の内訳  陽性{int(y.sum())}件\n{'=' * 62}")
    print(f"  {'特徴量':<34}{'係数':>10}{'単独AUC':>10}")
    print("  " + "-" * 54)
    for i in np.argsort(-np.abs(coef)):
        auc = roc_auc_score(y, X[:, i])
        # 向きが逆でも判別力は同じなので、0.5未満なら反転して読む
        print(f"  {feature_cols[i]:<34}{coef[i]:>+10.2f}{max(auc, 1 - auc):>10.3f}")


if __name__ == "__main__":
    main()
