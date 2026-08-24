"""確信度の校正(src/calibration.py)のテスト。

実行:
    python -m pytest tests/test_calibration.py -q
    python tests/test_calibration.py          # pytestが無い環境でも動く
"""

import sys
from pathlib import Path

import numpy as np
import pytest

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



# ---- 校正ファイルの取り違え対策 ----


def _make_weights_and_calibration(tmp_path, weights_name="model.pt"):
    """指紋つきの校正ファイルと、その元になった「重み」を作る。

    中身は指紋を取るだけなので、テストでは本物の重みである必要はない。
    """
    weights = Path(tmp_path) / weights_name
    weights.write_bytes(b"weights-content-v1")

    calibration = calib.Calibration(
        per_label={l: calib.LabelCalibration(a=1.0, b=-2.0, threshold=0.3, method="prior")
                   for l in LABELS}
    )
    calibration.source = calib.build_source(
        weights_path=weights,
        labels_csv=None,
        image_size=224,
        pos_weight=np.full(len(LABELS), 8.0),
        pos_weight_source="checkpoint",
        pos_weight_cap=8.0,
        split={"mode": "temporal", "seed": 42},
        fitted_on="val",
        n_fit=100,
    )
    calibration.save(calib.default_path(weights))
    return weights, calibration


def test_fingerprint_changes_with_content(tmp_path=None):
    tmp_path = Path(tmp_path) if tmp_path else Path("/tmp")
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    a.write_bytes(b"same")
    b.write_bytes(b"same")
    assert calib.file_fingerprint(a) == calib.file_fingerprint(b)
    b.write_bytes(b"different")
    assert calib.file_fingerprint(a) != calib.file_fingerprint(b)


def test_matching_calibration_loads(tmp_path=None):
    tmp_path = Path(tmp_path) if tmp_path else Path("/tmp")
    weights, original = _make_weights_and_calibration(tmp_path)
    loaded = calib.load_for_weights(weights, verbose=False)
    assert loaded.is_fitted
    np.testing.assert_allclose(loaded.thresholds(), original.thresholds())


def test_stale_calibration_is_rejected(tmp_path=None):
    """重みを学習し直したのに古い校正ファイルが残っている状態を検出する。"""
    tmp_path = Path(tmp_path) if tmp_path else Path("/tmp")
    weights, _ = _make_weights_and_calibration(tmp_path)
    weights.write_bytes(b"weights-content-v2")   # 学習し直した

    try:
        calib.load_for_weights(weights, verbose=False)
    except calib.StaleCalibrationError as e:
        assert "作り直す" in str(e)
        return
    raise AssertionError("古い校正ファイルが検出されずに読み込まれた")


def test_calibration_without_fingerprint_still_loads(tmp_path=None):
    """指紋を記録する前に作られた校正ファイルは、警告つきで読めること。"""
    tmp_path = Path(tmp_path) if tmp_path else Path("/tmp")
    weights = Path(tmp_path) / "old.pt"
    weights.write_bytes(b"weights")
    calibration = calib.Calibration(
        per_label={l: calib.LabelCalibration(a=1.0, b=-1.0, method="prior") for l in LABELS}
    )
    calibration.source = {"weights": str(weights)}    # 指紋なし
    calibration.save(calib.default_path(weights))

    loaded = calib.load_for_weights(weights, verbose=False)
    assert loaded.is_fitted


def test_source_records_what_it_was_built_from(tmp_path=None):
    tmp_path = Path(tmp_path) if tmp_path else Path("/tmp")
    weights, calibration = _make_weights_and_calibration(tmp_path)
    source = calibration.source
    for key in ("weights_sha256", "image_size", "pos_weight", "pos_weight_source",
                "pos_weight_cap", "split", "fitted_on", "n_fit", "created_at"):
        assert key in source, f"{key} が記録されていない"
    assert source["weights_sha256"] == calib.file_fingerprint(weights)



# ---- 温度スケーリングから移したテスト ----
# もとは tests/test_era5.py にあり、scripts/calibrate.py の温度スケーリング
# (共通のT一つ)を対象にしていた。手法をラベルごとの当てはめに変えたので、
# 同じ意図をこちらで引き継ぐ。


def test_platt_recovers_a_known_distortion():
    """確信度を3倍に膨らませたモデルから、その3倍を推定できるか。

    温度スケーリングが T≈3 を復元したのと同じ状況。Platt では a≈1/3 になる
    (sigmoid(a·z) で z が3倍なら、a を 1/3 にすれば元に戻る)。
    """
    rng = np.random.default_rng(0)
    z = rng.normal(scale=1.2, size=4000)
    y = (rng.random(4000) < calib.sigmoid(z)).astype(float)
    a, b = calib.fit_platt(z * 3.0, y, b_init=0.0)
    assert 0.28 < a < 0.40
    assert abs(b) < 0.2


def test_calibration_preserves_the_ranking_within_a_label():
    """較正はラベル内の順位を変えない。

    a > 0 なので sigmoid(a·z + b) は z の単調増加関数。よって、そのラベルの
    確率で並べた順序は較正前後で一致する。順位だけで決まる macro AP は不変で、
    F1 もラベルごとにしきい値を選び直すかぎり変わらない
    ——「較正は目盛りを直すだけ」と説明できる根拠。
    """
    rng = np.random.default_rng(0)
    z = rng.normal(scale=2.0, size=(2000, len(LABELS)))
    calibration = calib.Calibration(
        per_label={l: calib.LabelCalibration(a=0.6, b=-2.0, method="platt") for l in LABELS}
    )
    calibrated = calibration.probabilities(z)
    for i in range(len(LABELS)):
        before = np.argsort(calib.sigmoid(z[:, i]))
        after = np.argsort(calibrated[:, i])
        np.testing.assert_array_equal(before, after)


def test_calibration_reduces_the_error_it_is_meant_to_reduce():
    """膨らませた確信度のECEが、較正で半分以下になる。"""
    rng = np.random.default_rng(0)
    z = rng.normal(scale=1.2, size=4000)
    y = (rng.random(4000) < calib.sigmoid(z)).astype(float)
    inflated = z * 3.0

    a, b = calib.fit_platt(inflated, y, b_init=0.0)
    before = calib.expected_calibration_error(calib.sigmoid(inflated), y)
    after = calib.expected_calibration_error(calib.sigmoid(a * inflated + b), y)
    assert after < before / 2


def test_temperature_scaling_cannot_remove_the_pos_weight_shift():
    """共通の温度では pos_weight のかさ上げを消せない —— 手法を変えた理由。

    sigmoid(z/T) は T が何であっても z=0 を 0.5 に写す。しかし pos_weight=w で
    学習した出力の 0.5 は、真の確率 1/(1+w) に当たる。割り算では、この
    「ロジットを log w ぶん平行移動する」歪みを表現できない。
    """
    w = 8.0
    # 生の出力がちょうど 0.5 になる点。実体は 1/(1+8) = 0.111
    assert abs(calib.remove_class_weight_bias(np.array([0.5]), np.array([w]))[0] - 1 / (1 + w)) < 1e-9

    # どんな温度を掛けても、この点は 0.5 のまま動かない
    for temperature in (0.1, 0.5, 1.0, 2.0, 10.0):
        assert calib.sigmoid(np.array([0.0]) / temperature)[0] == 0.5

    # Platt の b はこれを表現できる
    a, b = 1.0, -np.log(w)
    assert abs(calib.sigmoid(np.array([a * 0.0 + b]))[0] - 1 / (1 + w)) < 1e-9



def _best_achievable_f1(scores: np.ndarray, y: np.ndarray) -> float:
    """しきい値を全通り試したときの、そのラベルの最大F1。

    固定の刻み幅ではなくデータ上の値そのものを候補にする。刻み幅を固定すると、
    較正で確率がずれたときに「最適点が候補から外れた」だけの差が出てしまい、
    較正そのものの影響と区別できなくなる。
    """
    from sklearn.metrics import f1_score

    return max(
        f1_score(y, (scores >= t).astype(float), zero_division=0)
        for t in np.unique(scores)
    )


def test_optimized_f1_is_unchanged_by_calibration():
    """しきい値を選び直すかぎり、較正は macro F1 を変えない。

    scripts/cross_validate.py は --optimize-thresholds で走っているため、
    過去に報告した交差検証の数値は較正を入れても変化しない。その根拠。
    """
    rng = np.random.default_rng(11)
    n = 400
    z = rng.normal(scale=1.5, size=(n, len(LABELS)))
    y = (rng.random((n, len(LABELS))) < calib.sigmoid(z)).astype(float)

    # pos_weight のかさ上げと自信過剰の両方が入った校正
    calibration = calib.Calibration(
        per_label={
            l: calib.LabelCalibration(a=0.7, b=-2.1 + 0.1 * i, method="platt")
            for i, l in enumerate(LABELS)
        }
    )
    raw = calib.sigmoid(z)
    calibrated = calibration.probabilities(z)

    for i in range(len(LABELS)):
        assert _best_achievable_f1(raw[:, i], y[:, i]) == pytest.approx(
            _best_achievable_f1(calibrated[:, i], y[:, i])
        ), f"{LABELS[i]} のF1が較正で変わった"

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
