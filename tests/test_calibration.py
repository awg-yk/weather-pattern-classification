"""確信度の校正(src/calibration.py)のテスト。

実行:
    python -m pytest tests/test_calibration.py -q
    python tests/test_calibration.py          # pytestが無い環境でも動く
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import calibration as calib
from src.labels import LABELS


def _weighted_bce_optimum(p, w):
    """pos_weight=w の重み付きBCEを最小にする出力。学習後のモデルが出す値の理論値。"""
    return w * p / (w * p + (1 - p))


def _synthetic(n=4000, w=8.0, seed=0):
    """pos_weightでかさ上げされた出力と、その正解ラベルを作る。"""
    rng = np.random.default_rng(seed)
    p = rng.beta(0.6, 3.0, size=n)                   # 少数ラベルらしい分布
    y = (rng.random(n) < p).astype(float)
    z = calib.logit(_weighted_bce_optimum(p, w))
    return z, y


def test_platt_recovers_pos_weight_shift():
    """pos_weightのかさ上げだけがある場合、当てはめは b = -log w を復元する。"""
    w = 8.0
    z, y = _synthetic(w=w)
    a, b = calib.fit_platt(z, y, b_init=0.0)
    assert abs(a - 1.0) < 0.05
    assert abs(b - (-np.log(w))) < 0.15


def test_calibration_reduces_ece():
    """校正すると、表示%と実際の的中率のズレ(ECE)が大きく減る。"""
    w = 8.0
    z, y = _synthetic(w=w)
    raw_ece = calib.expected_calibration_error(calib.sigmoid(z), y)
    a, b = calib.fit_platt(z, y, b_init=-np.log(w))
    cal_ece = calib.expected_calibration_error(calib.sigmoid(a * z + b), y)
    assert raw_ece > 0.2      # 未校正では3割近くずれている
    assert cal_ece < 0.05
    assert cal_ece < raw_ece / 5


def test_remove_class_weight_bias_is_exact_inverse():
    """重み付き学習の確率から、重み無しの確率へ厳密に戻せる。"""
    w = np.array([8.0, 3.0, 1.0])
    p = np.array([0.2, 0.05, 0.7])
    inflated = _weighted_bce_optimum(p, w)
    np.testing.assert_allclose(calib.remove_class_weight_bias(inflated, w), p, atol=1e-9)


def test_display_60_percent_is_really_16_percent():
    """報告された症状の再現: pos_weight=8 での表示60%の中身は約16%。"""
    recovered = calib.remove_class_weight_bias(np.array([0.60]), np.array([8.0]))[0]
    assert 0.15 < recovered < 0.17


def test_fit_uses_prior_only_when_few_positives():
    """検証データに陽性がほとんど無いラベルは、当てはめずpos_weightの補正だけを使う。"""
    rng = np.random.default_rng(1)
    n = 300
    logits = rng.normal(size=(n, len(LABELS)))
    targets = np.zeros((n, len(LABELS)))
    targets[:50, 0] = 1.0    # 1つ目のラベルだけ十分な陽性がある
    targets[:2, 1] = 1.0     # 2つ目は2件しかない
    pos_weight = np.full(len(LABELS), 8.0)

    calibration = calib.fit(logits, targets, pos_weight=pos_weight)
    assert calibration[LABELS[0]].method == "platt"
    assert calibration[LABELS[1]].method == "prior"
    assert abs(calibration[LABELS[1]].b - (-np.log(8.0))) < 1e-9
    assert calibration[LABELS[1]].a == 1.0


def test_fit_without_pos_weight_falls_back_to_no_correction():
    """pos_weightが分からない古い重みでも落ちず、当てはめられない分は素通しになる。"""
    logits = np.random.default_rng(2).normal(size=(50, len(LABELS)))
    targets = np.zeros((50, len(LABELS)))
    calibration = calib.fit(logits, targets, pos_weight=None)
    assert all(c.method == "none" for c in calibration.per_label.values())
    assert not calibration.is_fitted


def test_identity_calibration_matches_plain_sigmoid():
    """校正ファイルが無いときの挙動は、これまで通りの生のsigmoidと一致する。"""
    logits = np.random.default_rng(3).normal(size=(7, len(LABELS)))
    np.testing.assert_allclose(
        calib.Calibration.identity().probabilities(logits), calib.sigmoid(logits), atol=1e-12
    )


def test_from_probabilities_matches_probabilities():
    """確率しか手元に無い経路でも、ロジットから校正した場合と同じ結果になる。"""
    logits = np.random.default_rng(4).normal(size=(20, len(LABELS)))
    calibration = calib.Calibration(
        per_label={l: calib.LabelCalibration(a=0.8, b=-1.5, threshold=0.3, method="platt")
                   for l in LABELS}
    )
    np.testing.assert_allclose(
        calibration.from_probabilities(calib.sigmoid(logits)),
        calibration.probabilities(logits),
        atol=1e-6,
    )


def test_save_load_roundtrip(tmp_path=None):
    """保存して読み直しても、同じ確率・同じしきい値を返す。"""
    tmp_path = Path(tmp_path) if tmp_path else Path("/tmp")
    logits = np.random.default_rng(5).normal(size=(30, len(LABELS)))
    targets = (np.random.default_rng(6).random((30, len(LABELS))) < 0.3).astype(float)
    original = calib.fit(logits, targets, pos_weight=np.full(len(LABELS), 4.0))

    path = tmp_path / "roundtrip.calib.json"
    original.save(path)
    loaded = calib.Calibration.load(path)

    np.testing.assert_allclose(loaded.probabilities(logits), original.probabilities(logits))
    np.testing.assert_allclose(loaded.thresholds(), original.thresholds())


def test_default_path_sits_next_to_weights():
    assert calib.default_path("weights/model.pt").name == "model.calib.json"
    assert calib.default_path("runs/loyo/model_test2023.pt").name == "model_test2023.calib.json"


def test_load_for_weights_without_file_is_identity(tmp_path=None):
    """校正ファイルが無くても例外にせず、素通しの校正を返す(既存の重みが動き続ける)。"""
    tmp_path = Path(tmp_path) if tmp_path else Path("/tmp")
    calibration = calib.load_for_weights(tmp_path / "does_not_exist.pt", verbose=False)
    assert not calibration.is_fitted
    assert calibration.thresholds().tolist() == [0.5] * len(LABELS)


def test_predicted_labels_uses_per_label_thresholds():
    calibration = calib.Calibration(
        per_label={l: calib.LabelCalibration(threshold=0.2 if i == 0 else 0.9)
                   for i, l in enumerate(LABELS)}
    )
    probs = np.full(len(LABELS), 0.5)
    assert calibration.predicted_labels(probs) == [LABELS[0]]


def test_sigmoid_is_stable_at_extremes():
    """極端なロジットでもオーバーフロー警告や NaN を出さない。"""
    extreme = np.array([-800.0, -50.0, 0.0, 50.0, 800.0])
    out = calib.sigmoid(extreme)
    assert np.all(np.isfinite(out))
    assert out[0] == 0.0 and out[-1] == 1.0


def test_top1_confidence_and_correctness():
    probs = np.array([[0.1, 0.8, 0.3], [0.6, 0.2, 0.1]])
    targets = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    conf, hit = calib.top1_confidence_and_correctness(probs, targets)
    np.testing.assert_allclose(conf, [0.8, 0.6])
    np.testing.assert_allclose(hit, [1.0, 0.0])


def test_ece_is_zero_for_perfectly_calibrated_scores():
    """確信度どおりの割合で当たっていれば ECE はほぼ0になる。"""
    rng = np.random.default_rng(7)
    conf = rng.uniform(0.05, 0.95, size=20000)
    correct = (rng.random(20000) < conf).astype(float)
    assert calib.expected_calibration_error(conf, correct) < 0.02


if __name__ == "__main__":
    import inspect
    import tempfile

    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            if "tmp_path" in inspect.signature(fn).parameters:
                with tempfile.TemporaryDirectory() as d:
                    fn(d)
            else:
                fn()
            print(f"ok   {name}")
        except Exception as e:
            failures += 1
            print(f"FAIL {name}: {e}")
    print(f"\n{'すべて成功' if not failures else f'{failures}件失敗'}")
    sys.exit(1 if failures else 0)
