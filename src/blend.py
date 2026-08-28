"""2つのモデルの確率を混ぜるための共通部品。

`scripts/ensemble_chart_grid.py`(天気図 + ERA5格子)と
`scripts/ensemble_chart_features.py`(天気図 + 検出した特徴量)が共有する。

**実装を1つにしてある理由**: 重みと閾値の選び方が2か所でずれると、
どちらの混合が良いかという比較そのものが無効になる。`src/split.py` に
日付の解釈をまとめてあるのと同じ理由である。

ここは torch を読まない。木のモデル側から使えるようにするため
(`src/metrics.py` を `src/evaluate.py` から切り出したのと同じ事情)。
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.calibration import file_fingerprint

THRESHOLDS = np.round(np.arange(0.05, 1.0, 0.05), 2)
WEIGHTS = np.round(np.arange(0.0, 1.01, 0.05), 2)

# 重みの summary.json から引き継げなかったときに使う値。
# scripts/cross_validate.py の既定と揃えてある。
_DEFAULTS = {"val_mode": "tail", "seed": 42, "gap_days": 3, "val_ratio": 0.2}


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


def inherit_split_settings(weights_dir, args) -> None:
    """重みを作ったときの分割設定を読み、指定されなかった項目を埋める。

    **分割の条件が違うと、同じ重みでも閾値が別の検証データで決まる。**
    天気図単独の成績が変わるので、「混ぜて上積みが出た」の基準そのものが
    ずれる。実測で、val_mode を取り違えたまま回したとき天気図単独が
    0.609 になり、`runs/cv_baseline` の 0.641 と 0.032 も違った。
    その状態の「混合 0.641」は、低いほうの基準から測った数字だった。

    `docs/2026-08-26-stale-run-comparisons.md` の「別の条件で測った結果を
    並べてしまう」と同じ落とし穴なので、黙って進まないようにする。

    ラベルファイルが変わっている場合も止める。パスが同じでも中身が変わる
    ことがあり、実際に台風ラベルを付け直したあと、付け直す前の結果と並べて
    「改善した」と読んでしまったことがある。
    """
    summary_path = Path(weights_dir) / "summary.json"
    if not summary_path.exists():
        print(f"注意: {summary_path} がありません。分割設定を引き継げないので、"
              "指定した値(または既定値)で進みます。")
        print("      天気図単独の成績が元の実行と食い違う場合は、まずここを疑うこと。")
        _fill_defaults(args)
        return

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    config = summary.get("config") or {}

    fingerprint = summary.get("labels_fingerprint")
    if fingerprint and fingerprint != file_fingerprint(args.labels):
        raise SystemExit(
            f"ラベルファイルが、重みを作ったときと違います。\n"
            f"  重み側  : {summary.get('labels_path')} ({fingerprint})\n"
            f"  いま    : {args.labels} ({file_fingerprint(args.labels)})\n"
            "別のラベルで測った結果を混ぜることになります。"
        )

    inherited = []
    for name in ("val_mode", "seed", "gap_days", "val_ratio"):
        if getattr(args, name) is not None:
            continue
        if name in config and config[name] is not None:
            setattr(args, name, type(_DEFAULTS[name])(config[name]))
            inherited.append(f"{name}={getattr(args, name)}")
    _fill_defaults(args)
    if inherited:
        print(f"分割設定を {summary_path} から引き継ぎました: {', '.join(inherited)}")


def _fill_defaults(args) -> None:
    for name, value in _DEFAULTS.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
