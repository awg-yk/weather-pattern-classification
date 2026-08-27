"""検出結果から Phase 4 に渡す特徴量を作る部分のテスト。

守っているのは「**壊れても正常に見えるもの**」(tests/README.md)。
特徴量がずれても学習は最後まで走り、それらしい数字を出す。
"""

import math
from pathlib import Path

import numpy as np
import pytest

from src.chartfeatures import (
    EDGE_MARGIN,
    ChartDetections,
    build_features,
    feature_names,
    split_by_edge,
    to_row,
)
from src.chartsymbols import Segment
from src.regions import load_regions

REGIONS = load_regions()


def seg(kind: str, elongation: float, length: float, pixels: int = 500) -> Segment:
    return Segment(kind=kind, pixels=pixels, length=length,
                   elongation=elongation, cx=0.5, cy=0.5)


def test_counts_and_positions():
    det = ChartDetections(highs=[(0.2, 0.3), (0.4, 0.5)], lows=[(0.8, 0.6)])
    f = build_features(det, REGIONS)
    assert f["n_high"] == 2 and f["n_low"] == 1
    assert f["high_cx"] == pytest.approx(0.3)
    assert f["high_cy"] == pytest.approx(0.4)
    assert f["low_cx"] == pytest.approx(0.8)


def test_missing_positions_stay_nan_not_zero():
    """高気圧が無い日を0で埋めると、図の左上に高気圧があることになる。

    0埋めは嘘の位置を教える。NaNのまま渡し、木のほうで「値が無い」枝として
    扱わせる(HistGradientBoostingClassifier はNaNを直接扱える)。
    """
    f = build_features(ChartDetections(lows=[(0.5, 0.5)]), REGIONS)
    assert math.isnan(f["high_cx"]) and math.isnan(f["high_cy"])
    assert f["n_high"] == 0                      # 個数は0でよい
    assert math.isnan(f["high_low_distance"])    # 片方が無ければ距離も無い


def test_distance_is_the_nearest_pair():
    det = ChartDetections(highs=[(0.0, 0.0), (0.5, 0.5)], lows=[(0.6, 0.5)])
    f = build_features(det, REGIONS)
    assert f["high_low_distance"] == pytest.approx(0.1)


def test_spread_captures_two_separated_lows():
    """二つ玉低気圧は「低気圧が2つ離れて存在する」構造そのもの。

    個数だけでは、近くに2つある場合と離れて2つある場合が区別できない。
    """
    close = build_features(ChartDetections(lows=[(0.40, 0.5), (0.45, 0.5)]), REGIONS)
    apart = build_features(ChartDetections(lows=[(0.30, 0.5), (0.70, 0.5)]), REGIONS)
    assert close["n_low"] == apart["n_low"] == 2
    assert apart["low_spread"] > close["low_spread"]
    assert math.isnan(build_features(ChartDetections(lows=[(0.5, 0.5)]), REGIONS)["low_spread"])


# --- 前線 ---------------------------------------------------------------

def test_only_frontlike_segments_are_counted():
    """細長くない塊は前線ではない。文字や凡例を本数に数えない。"""
    det = ChartDetections(front_segments={
        "warm_front": [seg("warm_front", 10.0, 200.0), seg("warm_front", 1.2, 30.0)],
    })
    f = build_features(det, REGIONS)
    assert f["n_warm"] == 1
    assert f["warm_length"] == pytest.approx(200.0)


def test_absent_front_types_are_zero_not_nan():
    """前線が無いのは「0本」であって欠測ではない。位置とは扱いが違う。"""
    f = build_features(ChartDetections(), REGIONS)
    for name in ("n_warm", "n_cold", "n_occluded",
                 "warm_length", "cold_length", "occluded_length", "stationary_px"):
        assert f[name] == 0


# --- 地域ごとの存在有無 --------------------------------------------------

def test_region_membership_uses_the_same_rectangles_as_gradcam():
    """計画が「src/regions.py は Phase 3 に直接使える」と書いていた部分。"""
    okhotsk = REGIONS["okhotsk_high"]
    inside = ((okhotsk.x0 + okhotsk.x1) / 2, (okhotsk.y0 + okhotsk.y1) / 2)
    f = build_features(ChartDetections(highs=[inside]), REGIONS)
    assert f["high_in_okhotsk_high"] == 1
    assert f["low_in_okhotsk_high"] == 0

    far = build_features(ChartDetections(highs=[(0.05, 0.95)]), REGIONS)
    assert far["high_in_okhotsk_high"] == 0


# --- 中心が枠外の系 ------------------------------------------------------

def test_edge_symbols_are_separated_from_centres():
    """中心が枠外の系には×が描かれず文字だけになる。

    文字の位置は中心ではないので、矩形の内外判定に使うと嘘になる。
    「図の縁に記号がある」ことだけを伝える。
    """
    points = [(0.5, 0.5), (0.02, 0.4), (0.5, 0.98)]
    inside, edge = split_by_edge(points)
    assert inside == [(0.5, 0.5)]
    assert len(edge) == 2

    det = ChartDetections(highs=inside, edge_highs=edge)
    f = build_features(det, REGIONS)
    assert f["n_high"] == 1 and f["n_edge_high"] == 2
    # 縁の記号は位置の平均に混ぜない
    assert f["high_cx"] == pytest.approx(0.5)


def test_edge_margin_is_symmetric():
    just_inside = EDGE_MARGIN + 0.01
    inside, edge = split_by_edge([(just_inside, 0.5), (1 - just_inside, 0.5)])
    assert len(inside) == 2 and edge == []


# --- CSVの行 -------------------------------------------------------------

def test_row_matches_the_declared_column_order():
    """列の順がずれると、学習は通るのに別の特徴量として読まれる。"""
    det = ChartDetections(highs=[(0.2, 0.3)], lows=[(0.7, 0.6)],
                          front_segments={"cold_front": [seg("cold_front", 8.0, 150.0)]},
                          stationary_pixels=1234)
    names = feature_names(REGIONS)
    row = to_row(det, REGIONS)
    assert len(row) == len(names)
    values = build_features(det, REGIONS)
    for name, value in zip(names, row):
        assert (math.isnan(value) and math.isnan(values[name])) or value == values[name]


def test_every_feature_is_a_number():
    """文字列や None が混ざると、学習時に分かりにくい形で落ちる。"""
    row = to_row(ChartDetections(), REGIONS)
    assert all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in row)
    assert np.isfinite([v for v in row if not math.isnan(v)]).all()


# --- Phase 4 の評価が既存CNNと同じ土俵に乗ること --------------------------

def test_metrics_do_not_need_torch():
    """しきい値と自明な予測の基準は、学習の枠組みに依らず共有される。

    計画が「優劣の判断には、同じfold・同じラベル・同じ評価コードを使うこと」
    と書いている。evaluate.py は冒頭で torch を読むので、木のモデルからは
    使えなかった。src/metrics.py に切り出してある。
    """
    import subprocess
    import sys
    code = (
        "import sys\n"
        "from src.metrics import find_best_thresholds, trivial_macro_f1\n"
        "assert 'torch' not in sys.modules, 'torch が読み込まれている'\n"
        "print('ok')"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=Path(__file__).resolve().parent.parent)
    assert out.returncode == 0 and "ok" in out.stdout, out.stderr


def test_date_parsing_has_one_implementation():
    """日付の解釈がずれると分割ごとずれ、比較が無効になる。

    src/dataset.py が持っていたものを src/split.py に移し、dataset.py は
    再輸出している。2つの実装が並ぶことがないようにする。
    """
    pytest.importorskip("torch", reason="src/dataset.py が torch を読み込むため")
    from src.dataset import parse_datetime as from_dataset
    from src.split import parse_datetime as from_split
    assert from_dataset is from_split


def test_add_parsed_datetime_does_not_touch_the_original():
    import pandas as pd

    from src.split import add_parsed_datetime
    df = pd.DataFrame({"filename": ["Js_2024070100.png"], "date": ["20240701"]})
    out = add_parsed_datetime(df)
    assert "parsed_datetime" not in df.columns
    assert out["parsed_datetime"].iloc[0] == pd.Timestamp("2024-07-01 00:00")


def test_compare_runs_does_not_describe_a_tree_run_as_a_cnn(tmp_path):
    """CNNのconfigのキーが無いだけで「事前学習あり」と出ると、取り違える。"""
    from scripts.compare_runs import describe
    tree = {"method": "features", "config": {"model": "hgb"},
            "feature_columns": ["a", "b", "c"]}
    text = describe(tree)
    assert "特徴量" in text and "hgb" in text and "3個" in text
    assert "事前学習" not in text

    cnn = {"config": {"input_mode": "chart", "no_pretrained": False}}
    assert "事前学習あり" in describe(cnn)
