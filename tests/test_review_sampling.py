"""層別抽出した見直し結果を、本来の出現率に戻す計算のテスト。

守っているのは「出現率をいじった分を戻し忘れない」こと。層別抽出は陽性を
人為的に増やすので、そのまま集計すると人間の再現性(モデルの上限の目安)を
大きく過大評価する。それは数字として自然に見えてしまい、気づきにくい。
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.compare_review import cohen_kappa, reconstruct_at_prevalence, uncertainty


def _confusion_at(sensitivity, specificity, prevalence, n=2_000_000):
    """指定した感度・特異度・出現率で実際に標本を作り、そこから直接測る。"""
    rng = np.random.default_rng(0)
    original = (rng.random(n) < prevalence).astype(int)
    review = np.where(
        original == 1,
        (rng.random(n) < sensitivity).astype(int),
        (rng.random(n) >= specificity).astype(int),
    )
    both = int(((original == 1) & (review == 1)).sum())
    only_first = int(((original == 1) & (review == 0)).sum())
    only_second = int(((original == 0) & (review == 1)).sum())
    precision = both / (both + only_second)
    recall = both / (both + only_first)
    return {
        "f1": 2 * precision * recall / (precision + recall),
        "kappa": cohen_kappa(original, review),
    }


@pytest.mark.parametrize("prevalence", [0.046, 0.102, 0.398])
def test_reconstruction_matches_a_directly_measured_sample(prevalence):
    """組み直したF1とκが、その出現率で実際に測った値と一致する。"""
    got = reconstruct_at_prevalence(0.80, 0.95, prevalence)
    want = _confusion_at(0.80, 0.95, prevalence)
    assert got["f1"] == pytest.approx(want["f1"], abs=0.01)
    assert got["kappa"] == pytest.approx(want["kappa"], abs=0.01)


def test_ignoring_prevalence_overstates_the_ceiling_for_rare_labels():
    """層別のまま集計すると、少数ラベルの上限を大きく過大評価する。

    この差が、--stratified を用意した理由そのもの。
    """
    naive = reconstruct_at_prevalence(0.80, 0.95, 0.5)      # 陽性50%の集合で読んだ場合
    honest = reconstruct_at_prevalence(0.80, 0.95, 0.046)   # 本来の出現率に戻した場合
    assert naive["f1"] - honest["f1"] > 0.2, "過大評価が再現していない"
    # 感度(再現率)は出現率に依らない —— だから測る価値がある
    assert naive["recall"] == pytest.approx(honest["recall"])


def test_sensitivity_and_specificity_do_not_depend_on_prevalence():
    """組み直しの入力である2つの量が、出現率に依存しないこと。"""
    for p in (0.05, 0.3, 0.7):
        r = reconstruct_at_prevalence(0.9, 0.8, p)
        assert r["recall"] == pytest.approx(0.9)
        assert r["neither"] / (1 - p) == pytest.approx(0.8)


def test_more_negatives_help_rare_labels_more_than_more_positives():
    """少数ラベルでは、陰性を増やすほうがF1の区間を狭める。

    ツールが「どちらの層を増やすべきか」を案内する根拠。適合率は
    偽陽性 =(1-出現率)×(1-特異度) に支配されるため、陽性を増やしても効かない。
    """
    base = uncertainty(24, 30, 28, 30, 0.046, draws=4000, seed=1)
    more_pos = uncertainty(48, 60, 28, 30, 0.046, draws=4000, seed=1)
    more_neg = uncertainty(24, 30, 56, 60, 0.046, draws=4000, seed=1)
    w = lambda u: u["f1_high"] - u["f1_low"]
    assert w(more_neg) < w(base)
    assert w(more_neg) < w(more_pos), "少数ラベルで陰性が効かないなら案内が誤り"


def test_uncertainty_narrows_as_the_sample_grows():
    small = uncertainty(24, 30, 28, 30, 0.3, draws=4000, seed=1)
    large = uncertainty(240, 300, 280, 300, 0.3, draws=4000, seed=1)
    assert (large["f1_high"] - large["f1_low"]) < (small["f1_high"] - small["f1_low"]) / 2
