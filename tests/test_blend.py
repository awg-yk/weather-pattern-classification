"""2つのモデルの確率を混ぜる部分のテスト。

守っているのは「**壊れても正常に見えるもの**」(tests/README.md)。
行がずれても混合は最後まで走り、それらしいmacro F1を出す。
"""

import numpy as np
import pandas as pd
import pytest

from src.blend import WEIGHTS, align_by_filename, blend, macro_f1, per_label_weights


# --- 行の突き合わせ ------------------------------------------------------

def test_features_are_reordered_to_match_the_charts():
    """特徴量は filename で並べ替えること。行番号で結合してはいけない。

    **ずれても最後まで動いて、それらしい数字が出る。**天気図の予測と
    別の日付の特徴量を混ぜることになる。
    """
    charts = pd.DataFrame({"filename": ["c.png", "a.png", "b.png"]})
    feats = pd.DataFrame({"filename": ["a.png", "b.png", "c.png"],
                          "n_high": [1.0, 2.0, 3.0]})
    got = align_by_filename(charts, feats, ["n_high"])
    assert got.ravel().tolist() == [3.0, 1.0, 2.0]


def test_a_chart_without_features_becomes_nan(capsys):
    """特徴量の無い天気図は NaN にして、何件あったかを知らせること。

    黙って落とすと行がずれる。黙って0で埋めると嘘の値を教えることになる。
    """
    charts = pd.DataFrame({"filename": ["a.png", "b.png"]})
    feats = pd.DataFrame({"filename": ["a.png"], "n_high": [1.0]})
    got = align_by_filename(charts, feats, ["n_high"])
    assert got[0, 0] == 1.0 and np.isnan(got[1, 0])
    assert "1件" in capsys.readouterr().out


def test_duplicate_filenames_are_refused():
    """同じ filename が2行あると、どちらを使うか決められないので止める。"""
    charts = pd.DataFrame({"filename": ["a.png"]})
    feats = pd.DataFrame({"filename": ["a.png", "a.png"], "n_high": [1.0, 9.0]})
    with pytest.raises(SystemExit):
        align_by_filename(charts, feats, ["n_high"])


# --- 重みの選び方 --------------------------------------------------------

def test_the_weight_is_chosen_per_label():
    """得意分野が逆の2ラベルが、別々の重みを取れること。

    共通の重み1つにすると、実測で両方が妥協させられて全体が下がった
    (`src/blend.py` の説明を参照)。
    """
    rng = np.random.default_rng(0)
    n = 200
    y = np.zeros((n, 2), dtype=int)
    y[:100, 0] = 1          # ラベル0
    y[100:, 1] = 1          # ラベル1
    a = rng.uniform(0.3, 0.7, (n, 2))
    b = rng.uniform(0.3, 0.7, (n, 2))
    # ラベル0はA(天気図)が当て、Bは逆を答える。ラベル1はその反対にする。
    # **片方が逆を答える**ようにしないと、少し混ぜただけでも正解できてしまい、
    # 重みが選べているのかどうかが見えない
    a[:, 0] = np.where(y[:, 0] == 1, 0.9, 0.1)
    b[:, 0] = np.where(y[:, 0] == 1, 0.1, 0.9)
    b[:, 1] = np.where(y[:, 1] == 1, 0.9, 0.1)
    a[:, 1] = np.where(y[:, 1] == 1, 0.1, 0.9)

    weights, _ = per_label_weights(a, b, y, WEIGHTS)
    assert weights[0] < 0.5 < weights[1], f"ラベルごとに分かれていない: {weights}"


def test_a_tie_keeps_the_smaller_weight():
    """同じ成績になる重みが複数あるなら、小さいほう(天気図寄り)を選ぶこと。

    混合は天気図単独に対する上積みとして報告する。同点でわざわざ特徴量side
    に倒すと、上積みが無いのに「特徴量が効いた」と読めてしまう。
    """
    y = np.array([[1], [1], [0], [0]])
    same = np.array([[0.9], [0.9], [0.1], [0.1]], dtype=float)
    weights, _ = per_label_weights(same, same.copy(), y, WEIGHTS)
    assert weights[0] == 0.0


def test_a_label_with_no_positives_falls_back_to_the_first_model():
    """検証データに陽性が無いラベルは、選びようがないので混ぜない。"""
    y = np.zeros((20, 1), dtype=int)
    weights, _ = per_label_weights(np.full((20, 1), 0.7), np.full((20, 1), 0.2), y)
    assert weights[0] == 0.0


def test_blending_never_loses_to_both_sources_on_the_validation_set():
    """検証データの上では、混合は少なくとも片方と同じ成績になること。

    重みの候補に0と1が入っているので、混ぜて悪くなる選択は選ばれない。
    ここが崩れると「混ぜたら下がった」の原因が重み選びなのかテスト側なのか
    切り分けられなくなる。
    """
    rng = np.random.default_rng(1)
    n = 300
    y = (rng.uniform(size=(n, 3)) > 0.6).astype(int)
    a = np.clip(y * 0.6 + rng.normal(0, 0.25, (n, 3)) + 0.2, 0, 1)
    b = np.clip(y * 0.4 + rng.normal(0, 0.35, (n, 3)) + 0.2, 0, 1)

    weights, thresholds = per_label_weights(a, b, y, WEIGHTS)
    mixed = macro_f1(blend(a, b, weights), y, thresholds)
    only_a = macro_f1(a, y, per_label_weights(a, b, y, [0.0])[1])
    only_b = macro_f1(b, y, per_label_weights(a, b, y, [1.0])[1])
    assert mixed >= max(only_a, only_b) - 1e-9


def test_blend_accepts_a_scalar_and_a_vector():
    a = np.array([[0.8, 0.2]])
    b = np.array([[0.0, 1.0]])
    assert blend(a, b, 0.5).ravel().tolist() == [0.4, 0.6]
    assert blend(a, b, np.array([0.0, 1.0])).ravel().tolist() == [0.8, 1.0]
