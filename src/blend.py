"""2つのモデルの確率を混ぜるための共通部品。

`scripts/ensemble_chart_grid.py`(天気図 + ERA5格子)と
`scripts/ensemble_chart_features.py`(天気図 + 検出した特徴量)が共有する。

**実装を1つにしてある理由**: 重みと閾値の選び方が2か所でずれると、
どちらの混合が良いかという比較そのものが無効になる。`src/split.py` に
日付の解釈をまとめてあるのと同じ理由である。

ここは torch を読まない。木のモデル側から使えるようにするため
(`src/metrics.py` を `src/evaluate.py` から切り出したのと同じ事情)。
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

THRESHOLDS = np.round(np.arange(0.05, 1.0, 0.05), 2)
WEIGHTS = np.round(np.arange(0.0, 1.01, 0.05), 2)


def align_by_filename(base_df: pd.DataFrame, other: pd.DataFrame,
                      columns: list, what: str = "特徴量") -> np.ndarray:
    """`other` を `base_df` の行順に並べ替えた行列を返す。

    **行番号で結合してはいけない。**ずれても最後まで動いて、それらしい数字が
    出る。天気図の予測と別の日付の特徴量を混ぜることになる。
    `src/dataset.py` が同じ理由で filename を使っている。

    `base_df` にあって `other` に無い行は NaN で埋める(木のモデルは欠測を
    扱えるため)。何件埋めたかは呼び出し側が知る必要があるので、警告を出す。
    """
    if "filename" not in other.columns:
        raise SystemExit(f"{what}に filename 列がありません。")
    indexed = other.set_index("filename")
    if indexed.index.has_duplicates:
        dupes = indexed.index[indexed.index.duplicated()].unique()[:3]
        raise SystemExit(
            f"{what}に同じ filename が複数あります(例: {', '.join(map(str, dupes))})。"
            "どの行を使うか決められません。"
        )
    aligned = indexed.reindex(base_df["filename"].values)
    missing = int(aligned[columns].isna().all(axis=1).sum())
    if missing:
        print(f"警告: {len(base_df)}件のうち{missing}件に{what}がありません。"
              "検出を全画像で走らせたか確認すること。")
    return aligned[columns].to_numpy(dtype=np.float64)


def per_label_weights(val_a: np.ndarray, val_b: np.ndarray, val_y: np.ndarray,
                      candidates=WEIGHTS):
    """ラベルごとに、検証データで最も良い混合の重みと閾値を選ぶ。

    重みを全ラベル共通の1つの値にすると、得意分野が逆のラベルどうしが妥協させ
    られる。実測では、天気図が強い台風(0.764対0.492)と格子が強いオホーツク海
    高気圧(0.131対0.345)が同じw=0.35前後を共有し、台風は0.711に落ち、オホーツクは
    0.200に落ちた -- 全体でも天気図単独(0.619)を下回る0.607になった。

    閾値は既にラベルごとに決めている。重みも同じ粒度で決めれば、そのラベルで
    弱いほうのモデルは検証データを見た時点で自然に外れる。

    陽性が1件も無いラベルは選びようがないので、混ぜない(A側のみ)に倒す。
    返り値の重みは「B側をどれだけ混ぜるか」(0=Aのみ, 1=Bのみ)。
    """
    weights = np.zeros(val_y.shape[1])
    thresholds = np.full(val_y.shape[1], 0.5)
    for i in range(val_y.shape[1]):
        if val_y[:, i].sum() == 0:
            continue
        best = None
        for w in candidates:
            blended = (1 - w) * val_a[:, i] + w * val_b[:, i]
            for threshold in THRESHOLDS:
                score = f1_score(val_y[:, i], (blended > threshold).astype(float),
                                 zero_division=0)
                if best is None or score > best[0]:
                    best = (score, w, threshold)
        _, weights[i], thresholds[i] = best
    return weights, thresholds


def blend(a: np.ndarray, b: np.ndarray, weights) -> np.ndarray:
    """確率を混ぜる。weights はスカラーでもラベルごとの配列でもよい。"""
    return (1 - weights) * a + weights * b


def macro_f1(probs: np.ndarray, targets: np.ndarray, thresholds) -> float:
    return f1_score(targets, (probs > thresholds).astype(float),
                    average="macro", zero_division=0)
