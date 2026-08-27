"""Phase 4: 検出から作った特徴量で気圧配置を分類し、交差検証する。

`docs/2026-08-26-detection-plan.md` の Phase 4。Phase 3 の特徴量
(`scripts/build_features.py` が書き出すCSV)を入力に、10ラベルを出力する。

既存CNNとの比較について
-----------------------
計画が「Phase 4 の出力は今と同じ10ラベルなので、既存CNNとは競合手法の
関係になる。優劣の判断には、同じfold・同じラベル・同じ評価コードを使うこと」
と書いている。そのため:

  - 分割は `src/split.py` の make_splits(mode="loyo") をそのまま使う
    (LOYO・gap_days による時間的リーク防止)
  - 閾値は `src/evaluate.py` の find_best_thresholds を検証データで決める
  - `summary.json` は `scripts/cross_validate.py` と同じ形にする。
    そのまま `scripts/compare_runs.py` に渡せる

比較相手は `runs/cv_baseline`(macro F1 0.641)。
ラベル別の数字は `docs/2026-08-26-detection-prescreen.md`。

モデル
------
計画の第一候補は XGBoost。既定は scikit-learn の
HistGradientBoostingClassifier にしてある。依存を増やさずに済み、
**欠測(NaN)をそのまま扱える**ためで、これは Phase 3 の特徴量にとって
重要である(高気圧が1つも無い日は「高気圧の位置」が存在しない)。
`--model xgboost` で切り替えられる。

使い方:
    python -m scripts.cv_features --features data/features.csv `
        --labels data/labels_v2.csv --years 2023 2024 2025 --out runs/cv_features
"""

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from src.calibration import file_fingerprint
from src.metrics import find_best_thresholds, trivial_macro_f1
from src.labels import LABELS, LABEL_JA
from src.split import add_parsed_datetime, make_splits

# 特徴量CSVで、特徴量ではない列。
KEY_COLUMNS = ("filename", "date")


def load_dataset(features_path: Path, labels_path: Path) -> tuple:
    """特徴量とラベルを、行の順を揃えて読む。

    `src/dataset.py` が「ラベル・画像・ERA5特徴量が行番号でずれないこと」を
    守っているのと同じ理由で、ここでも filename で突き合わせる。
    行番号で結合すると、ずれても最後まで動いてそれらしい数字が出る。
    """
    features = pd.read_csv(features_path)
    labels = pd.read_csv(labels_path)
    if "filename" not in features.columns:
        raise SystemExit(f"{features_path} に filename 列がありません。")

    merged = labels.merge(features, on="filename", how="inner", suffixes=("", "_feat"))
    missing = len(labels) - len(merged)
    if missing:
        print(f"警告: ラベルのある{len(labels)}件のうち{missing}件に特徴量がありません。")
        print("      検出を全画像で走らせたか確認すること。")
    if merged.empty:
        raise SystemExit("特徴量とラベルが1件も突き合いませんでした。filename を確認すること。")

    # 分割が使う観測日時。src/split.py が持っているものを使う。別々に実装すると
    # 日付の解釈がずれ、分割ごとずれて比較が無効になる。
    merged = add_parsed_datetime(merged)
    feature_columns = [c for c in features.columns if c not in KEY_COLUMNS]
    X = merged[feature_columns].to_numpy(dtype=np.float64)
    y = np.zeros((len(merged), len(LABELS)), dtype=np.int64)
    for row, text in enumerate(merged["label"].fillna("")):
        for name in str(text).split("|"):
            if name in LABELS:
                y[row, LABELS.index(name)] = 1
    return merged, X, y, feature_columns


def make_model(kind: str, seed: int):
    """ラベル1つぶんの二値分類器を作る。"""
    if kind == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError:
            raise SystemExit(
                "xgboost が入っていません。pip install xgboost するか、"
                "--model hgb (既定) を使ってください。"
            )
        return XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9,
            eval_metric="logloss", random_state=seed,
        )
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_iter=300, max_depth=4, learning_rate=0.08, random_state=seed,
    )


def fit_predict(X_train, y_train, X_apply, kind: str, seed: int) -> np.ndarray:
    """ラベルごとに分類器を作り、確率を並べて返す。

    ラベルは併用される(1枚に複数付く)ので、多クラスではなくラベルごとの
    二値分類にする。`src/labels.py` の「ラベルの併用についての決まり」を参照。
    """
    probs = np.zeros((len(X_apply), len(LABELS)), dtype=np.float64)
    for i, label in enumerate(LABELS):
        column = y_train[:, i]
        if column.sum() == 0:
            continue          # この分割に陽性が無い。確率0のままにする
        if column.sum() == len(column):
            probs[:, i] = 1.0
            continue
        model = make_model(kind, seed)
        model.fit(X_train, column)
        probs[:, i] = model.predict_proba(X_apply)[:, 1]
    return probs


def evaluate_fold(y_true: np.ndarray, preds: np.ndarray, test_year) -> dict:
    """`src/evaluate.py` と同じ形の1foldぶんの結果を返す。"""
    report = classification_report(
        y_true, preds, target_names=LABELS, output_dict=True, zero_division=0,
    )
    supports = {label: int(y_true[:, i].sum()) for i, label in enumerate(LABELS)}
    evaluable = [label for label in LABELS if supports[label] > 0]
    macro_evaluable = (
        float(np.mean([report[label]["f1-score"] for label in evaluable]))
        if evaluable else 0.0
    )
    trivial, trivial_per_label = trivial_macro_f1(y_true)
    return {
        "test_year": test_year,
        "n_eval": int(len(y_true)),
        "macro_f1_all_labels": report["macro avg"]["f1-score"],
        "macro_f1_evaluable": macro_evaluable,
        "trivial_macro_f1": trivial,
        "macro_f1_over_trivial": macro_evaluable - trivial,
        "micro_f1": report["micro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "per_label": {
            label: {
                "f1": report[label]["f1-score"],
                "precision": report[label]["precision"],
                "recall": report[label]["recall"],
                "support": supports[label],
                "trivial_f1": float(trivial_per_label[LABELS.index(label)]),
            }
            for label in LABELS
        },
    }


def bootstrap_spread(X, y, train, test, thresholds, kind: str, seed: int,
                     repeats: int) -> dict:
    """学習データを重複ありで取り直し、成績がどれだけ動くかを返す。

    **特徴量を1つ2つ足したときの差が、この幅より小さいなら読めない。**

    seed を変えても意味が無いことに注意。HistGradientBoosting は既定で
    部分抽出をしないので、同じデータからは必ず同じ木ができる(確かめ済み:
    seed 1/2/3/42 で確率が完全に一致した)。動かすべきは乱数ではなく
    **学習データそのもの**である。
    """
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(repeats):
        picked = rng.choice(train, size=len(train), replace=True)
        probs = fit_predict(X[picked], y[picked], X[test], kind, seed)
        preds = (probs > thresholds).astype(int)
        fold = evaluate_fold(y[test], preds, None)
        scores.append(fold["macro_f1_evaluable"])
    arr = np.array(scores)
    return {
        "repeats": repeats,
        "min": float(arr.min()), "max": float(arr.max()),
        "mean": float(arr.mean()), "std": float(arr.std(ddof=1)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--features", required=True, help="scripts/build_features.py の出力")
    parser.add_argument("--labels", default="data/labels_v2.csv")
    parser.add_argument("--years", nargs="+", required=True, help="foldにするテスト年")
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="hgb", choices=("hgb", "xgboost"))
    parser.add_argument("--gap-days", type=int, default=3,
                        help="テスト年からこの日数以内の学習データを除く(リーク防止)")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--val-mode", default="tail", choices=("spread", "tail"),
                        help="検証データの取り方。しきい値はここで決まる。"
                             "tailは直近をまとめて取るので季節が偏る"
                             "(実測では1〜4月と11〜12月しか入らなかった)。"
                             "spreadは通年になるので、梅雨や台風のような季節性の"
                             "強いラベルでもしきい値が偏らない")
    parser.add_argument("--seed", type=int, default=42,
                        help="hgb では効かない(既定で部分抽出をしないため決定的)。"
                            "ばらつきを測るなら --bootstrap を使うこと")
    parser.add_argument("--bootstrap", type=int, default=0,
                        help="学習データを重複ありで取り直して、この回数だけ"
                             "余分に当てはめ、成績の幅を出す。特徴量を1つ2つ"
                             "足したときの差が、この幅より小さいなら読めない")
    args = parser.parse_args()

    merged, X, y, feature_columns = load_dataset(Path(args.features), Path(args.labels))
    print(f"{len(merged)}件、特徴量{len(feature_columns)}個、モデル {args.model}")

    results = []
    for test_year in args.years:
        splits = make_splits(
            merged, mode="loyo", test_year=test_year,
            val_ratio=args.val_ratio, gap_days=args.gap_days, seed=args.seed,
            val_mode=args.val_mode,
        )
        train, val, test = splits["train"], splits["val"], splits["test"]
        print(f"\n=== fold: テスト={test_year}年 "
              f"(学習{len(train)} / 検証{len(val)} / テスト{len(test)}) ===")

        # 閾値は検証データで決める。テストで決めるとテストに合わせたことになる。
        # **検証データの季節が偏ると、その季節にしか出ないラベルの閾値が
        # 見当違いになる。**実測では tail だと1〜4月と11〜12月しか入らず、
        # 停滞前線(梅雨・秋雨)はAUC 0.874の信号があるのにF1は0.516だった。
        val_probs = fit_predict(X[train], y[train], X[val], args.model, args.seed)
        thresholds = find_best_thresholds(val_probs, y[val])

        test_probs = fit_predict(X[train], y[train], X[test], args.model, args.seed)
        preds = (test_probs > thresholds).astype(int)

        fold = evaluate_fold(y[test], preds, test_year)
        results.append(fold)
        print(f"  macro F1(評価できたラベルのみ) {fold['macro_f1_evaluable']:.3f}"
              f"  (自明な予測 {fold['trivial_macro_f1']:.3f}、"
              f"上積み {fold['macro_f1_over_trivial']:+.3f})")

        if args.bootstrap:
            spread = bootstrap_spread(
                X, y, train, test, thresholds, args.model, args.seed, args.bootstrap)
            fold["bootstrap"] = spread
            print(f"  学習データを取り直した{args.bootstrap}回の幅: "
                  f"{spread['min']:.3f}〜{spread['max']:.3f} "
                  f"(標準偏差 {spread['std']:.3f})")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 72}\n交差検証のまとめ({len(results)}fold)\n{'=' * 72}")

    def summarize(values):
        mean = statistics.mean(values)
        sd = statistics.stdev(values) if len(values) > 1 else 0.0
        return f"{mean:.3f} ± {sd:.3f}"

    print("\n【全体】")
    for key, name in (
        ("macro_f1_evaluable", "macro F1(評価できたラベルのみ)"),
        ("macro_f1_all_labels", "macro F1(全ラベル)"),
        ("micro_f1", "micro F1"),
        ("weighted_f1", "weighted F1"),
    ):
        per_fold = ", ".join(f"{r[key]:.3f}" for r in results)
        print(f"  {name:<32} {summarize([r[key] for r in results])}   (各fold: {per_fold})")

    print("\n【ラベル別 F1】")
    for label in LABELS:
        f1s = [r["per_label"][label]["f1"] for r in results]
        supports = [r["per_label"][label]["support"] for r in results]
        detail = " ".join(f"{f:.2f}({s})" for f, s in zip(f1s, supports))
        print(f"  {LABEL_JA[label]:<22} {summarize(f1s):<18} {detail}")

    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                # どの手法で出た数字かを結果と一緒に残す。CNNのsummary.jsonとは
                # configのキーがまるごと違うので、compare_runs がCNNの言葉で
                # 説明してしまわないようにする。
                "method": "features",
                "config": vars(args),
                "feature_columns": feature_columns,
                # ラベルファイルの指紋。パスが同じでも中身が変わることがある
                "labels_fingerprint": file_fingerprint(args.labels),
                "labels_path": str(args.labels),
                "features_fingerprint": file_fingerprint(args.features),
                "folds": results,
                "mean": {
                    key: statistics.mean([r[key] for r in results])
                    for key in ("macro_f1_evaluable", "macro_f1_all_labels",
                                "micro_f1", "weighted_f1")
                },
            },
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nまとめを書き出しました: {summary_path}")
    print("既存CNNと比べるには:")
    print(f"  python -m scripts.compare_runs runs\\cv_baseline {args.out}")


if __name__ == "__main__":
    main()
