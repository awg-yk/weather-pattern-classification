"""しきい値の最適化と、自明な予測の基準。学習の枠組みに依らない部分。

`src/evaluate.py` から切り出した。中身は変えていない。

切り出した理由: evaluate.py は冒頭で torch と Dataset を読み込むので、
この2つを使うだけでも torch が要る。Phase 4 の木のモデル
(`scripts/cv_features.py`)は torch を使わないのに、比較のために同じ評価
コードを通す必要がある。計画が「優劣の判断には、同じfold・同じラベル・
同じ評価コードを使うこと」と書いているとおりで、**この2つは学習の枠組みに
依らずに共有されるべきものである。**

`from src.evaluate import find_best_thresholds` は今までどおり動く
(evaluate.py が再輸出している)。
"""

import numpy as np
from sklearn.metrics import f1_score


def trivial_macro_f1(labels: np.ndarray) -> tuple:
    """「全部を陽性と予測する」だけで得られるmacro F1と、そのラベル別の値。

    出現率pのラベルは、常に陽性と答えるだけでF1 = 2p/(1+p) を得る。頻出ラベルでは
    これが0.5を超えるため、macro F1の絶対値は学習の成果を表さない。この下駄を
    引いて初めて、モデルが何を足したのかが分かる。

    実際、ERA5格子を224で学習した回はmacro F1 0.250で、この基準(約0.29)を
    下回っていた -- 数字だけ見ていると「低いが学習はできている」と読めてしまう。
    """
    prevalence = labels.mean(axis=0)
    per_label = np.where(prevalence > 0, 2 * prevalence / (1 + prevalence), 0.0)
    evaluable = prevalence > 0
    return (float(per_label[evaluable].mean()) if evaluable.any() else 0.0), per_label


def find_best_thresholds(probs: np.ndarray, labels: np.ndarray, steps: int = 19) -> np.ndarray:
    """ラベルごとにF1を最大化する閾値を探索する。

    マルチラベルではラベルごとに陽性/陰性の出現頻度が大きく異なり、一律0.5では
    少数派ラベルを取りこぼしやすい。0.05刻みでF1が最大になる点をラベルごとに選ぶ。
    """
    candidates = np.linspace(0.05, 0.95, steps)
    best_thresholds = np.full(labels.shape[1], 0.5)
    for i in range(labels.shape[1]):
        col_labels = labels[:, i]
        if col_labels.sum() == 0:
            continue
        best_f1 = -1.0
        for t in candidates:
            preds = (probs[:, i] > t).astype(float)
            f1 = f1_score(col_labels, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresholds[i] = t
    return best_thresholds
