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
from src.labels import LABELS
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


# --- 天気図から特徴量CSVまで ---------------------------------------------

def synthetic_chart_with(n_high: int, n_low: int, stationary: bool):
    """記号と前線を指定した数だけ置いた天気図。"""
    import cv2
    black, warm, cold, coast = (4, 4, 4), (252, 4, 4), (4, 4, 252), (164, 44, 44)
    img = np.full((600, 600, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (590, 590), coast, 3)
    for i in range(n_high):
        x = 60 + i * 200
        cv2.rectangle(img, (x, 60), (x + 20, 160), black, 6)
        cv2.rectangle(img, (x + 60, 60), (x + 80, 160), black, 6)
        cv2.line(img, (x, 110), (x + 80, 110), black, 6)
    for i in range(n_low):
        x = 60 + i * 200
        cv2.line(img, (x, 300), (x, 420), black, 8)
        cv2.line(img, (x, 420), (x + 80, 420), black, 8)
    if stationary:
        for k in range(8):
            colour = warm if k % 2 == 0 else cold
            cv2.line(img, (60 + k * 60, 520), (60 + k * 60 + 50, 522), colour, 6)
    return img


def glyph_template(kind: str):
    import cv2

    from src.chartsymbols import crop_template, glyph_candidates
    tile = np.full((300, 300, 3), 255, dtype=np.uint8)
    black = (4, 4, 4)
    if kind == "H":
        cv2.rectangle(tile, (60, 60), (80, 160), black, 6)
        cv2.rectangle(tile, (120, 60), (140, 160), black, 6)
        cv2.line(tile, (60, 110), (140, 110), black, 6)
    else:
        cv2.line(tile, (60, 60), (60, 180), black, 8)
        cv2.line(tile, (60, 180), (140, 180), black, 8)
    c = glyph_candidates(tile, max_side=250)[0]
    return crop_template(tile, (c.x0, c.y0, c.x1, c.y1))


def test_analyse_chart_recovers_what_was_drawn(tmp_path):
    """天気図に置いた記号と前線の数を、検出が取り戻せること。

    ここが狂うと、特徴量は正常な形のまま中身だけが嘘になり、学習は最後まで
    走ってそれらしい数字を出す。
    """
    from PIL import Image

    from scripts.build_features import analyse_chart

    chart = tmp_path / "Js_2024070100.png"
    Image.fromarray(synthetic_chart_with(2, 1, stationary=True)).save(chart)
    templates = {"H": glyph_template("H"), "L": glyph_template("L")}

    det, _ = analyse_chart(chart, templates, {}, scale=1.0, threshold=0.65,
                           angle_range=20, angle_step=5)
    assert len(det.highs) + len(det.edge_highs) == 2
    assert len(det.lows) + len(det.edge_lows) == 1
    assert det.stationary_pixels > 0


def test_shrinking_happens_on_the_binary_mask_not_the_colours():
    """RGBのまま縮めると海岸線の赤茶と黒が混ざり、色の切り分けが崩れる。"""
    import cv2

    from scripts.build_features import ink_image, shrink
    img = synthetic_chart_with(1, 1, stationary=False)
    ink = ink_image(img)
    assert set(np.unique(ink)) <= {0, 255}          # 2値になっている
    # 海岸線(赤茶)はインクに含まれない
    assert ink[12, 300].tolist() == [255, 255, 255]
    small = shrink(ink, 0.5)
    assert small.shape[0] == img.shape[0] // 2


def test_marks_supply_the_position_and_letters_supply_the_type(tmp_path):
    """印は位置を、文字は種別を担うこと。

    ×は中心そのものなので位置として正しく、文字は形が安定していて種別として
    正しい。**印の形(丸の有無)で高低を見分けようとすると壊れる**ので、
    種別はいちばん近い文字から取る。
    """
    from PIL import Image

    from scripts.build_features import analyse_chart

    chart = tmp_path / "Js_2024070100.png"
    Image.fromarray(synthetic_chart_with(1, 1, stationary=False)).save(chart)
    letters = {"H": glyph_template("H"), "L": glyph_template("L")}

    # 印が無いときは文字の位置が中心として使われる
    without, _ = analyse_chart(chart, letters, {}, 1.0, 0.65, 20, 5)
    assert len(without.highs) == 1

    # 印が高気圧の場所に当たると、種別は近くの H から取られる
    marks = {"mark": glyph_template("H")}
    with_marks, report = analyse_chart(chart, letters, marks, 1.0, 0.65, 20, 5)
    assert len(with_marks.highs) == 1
    assert with_marks.edge_highs == []       # H の文字は印と組になったので余らない
    assert report["marks"] == 1 and report["orphan_marks"] == 0


def test_marks_are_matched_at_their_own_scale(tmp_path):
    """印は文字と別の倍率で当てられること。

    印は約31x31、H/L の文字は約95x117。文字に効く 0.7 は印には粗すぎて
    22x22 になり、丸と×の細部が潰れる。ここが繋がっていないと、印を渡した
    ときだけ位置の情報が消え、成績が静かに落ちる(実測 0.403 -> 0.321)。
    """
    from PIL import Image

    from scripts.build_features import analyse_chart

    chart = tmp_path / "Js_2024070100.png"
    Image.fromarray(synthetic_chart_with(1, 1, stationary=False)).save(chart)
    letters = {"H": glyph_template("H"), "L": glyph_template("L")}
    marks = {"mark": glyph_template("H")}

    # 文字の倍率を変えても、印の当たり方は変わらない(別々に照合している)
    _, full = analyse_chart(chart, letters, marks, scale=1.0, threshold=0.65,
                            angle_range=20, angle_step=5, mark_scale=1.0)
    _, mixed = analyse_chart(chart, letters, marks, scale=0.7, threshold=0.65,
                             angle_range=20, angle_step=5, mark_scale=1.0)
    assert full["marks"] == mixed["marks"] == 1

    # 印を縮めると当たらなくなる。**これが 0.403 -> 0.321 の正体**
    _, shrunk = analyse_chart(chart, letters, marks, scale=1.0, threshold=0.65,
                              angle_range=20, angle_step=5, mark_scale=0.3)
    assert shrunk["marks"] == 0


def test_build_features_counts_what_it_found(tmp_path):
    """検出数を数えて返すこと。

    印を入れて成績が落ちたときに、それが「印が当たっていない」せいなのかを
    切り分けられるようにする。**数字が無いと、悪化の原因を推測で語ることになる。**
    """
    from PIL import Image

    from scripts import build_features

    chart = tmp_path / "Js_2024070100.png"
    Image.fromarray(synthetic_chart_with(2, 1, stationary=False)).save(chart)
    build_features._WORKER.update(
        letters={"H": glyph_template("H"), "L": glyph_template("L")},
        marks={}, regions=load_regions(), scale=1.0, mark_scale=1.0,
        mark_radius=0.08, overlay=None,
    )
    _, _, report = build_features._run_one((str(chart), 0.65, 20, 5))
    assert {"high", "low", "edge_high", "edge_low", "marks", "orphan_marks",
            "letters_H", "letters_L"} <= set(report)
    assert report["high"] + report["edge_high"] == 2


def test_overlay_is_written_so_the_numbers_can_be_checked(tmp_path):
    """重ね描きが書き出せること。

    **数字だけでは「印が出すぎ」と「文字が足りない」を区別できない。**
    印7.9に対して種別が付いたのが3.1、という数字はどちらでも起こりうる。
    """
    from PIL import Image

    from scripts.build_features import analyse_chart

    chart = tmp_path / "Js_2024070100.png"
    Image.fromarray(synthetic_chart_with(1, 1, stationary=False)).save(chart)
    out = tmp_path / "overlay"
    analyse_chart(chart, {"H": glyph_template("H")}, {"mark": glyph_template("H")},
                  1.0, 0.65, 20, 5, 1.0, 0.08, overlay_dir=out)
    assert (out / "Js_2024070100_marks.png").exists()


# --- 退化した特徴量に対する守り ------------------------------------------

def test_all_nan_column_is_dropped_before_fitting():
    """1つも値が無い列を落とすこと。定数列は落とさないこと。

    HistGradientBoosting は欠測を扱えるが、全部NaNの列だけは扱えず
    `window shape cannot be larger than input array shape` で落ちる。
    ブートストラップで学習データを取り直すと、希な特徴量がたまたま全滅する。
    """
    from scripts.cv_features import drop_empty_columns

    train = np.array([[1.0, np.nan, 5.0], [2.0, np.nan, 5.0]])
    apply = np.array([[3.0, 9.0, 5.0]])
    kept_train, kept_apply = drop_empty_columns(train, apply)
    assert kept_train.shape[1] == 2 and kept_apply.shape[1] == 2
    assert kept_train[0].tolist() == [1.0, 5.0]     # 定数列(5.0)は残る
    assert kept_apply[0].tolist() == [3.0, 5.0]     # 検証側も同じ列を落とす


def test_fit_predict_survives_an_all_nan_feature():
    """全部NaNの列があっても学習が通ること(落ちていた経路そのもの)。"""
    from scripts.cv_features import fit_predict

    rng = np.random.default_rng(0)
    X = np.column_stack([rng.normal(size=40), np.full(40, np.nan)])
    y = np.zeros((40, len(LABELS)), dtype=int)
    y[X[:, 0] > 0, 0] = 1
    probs = fit_predict(X, y, X, "hgb", seed=0)
    assert probs.shape == (40, len(LABELS))
    assert np.isfinite(probs).all()


def test_fit_predict_survives_when_every_feature_is_empty():
    """使える特徴量が1つも無いときは、落ちずに確率0を返すこと。"""
    from scripts.cv_features import fit_predict

    X = np.full((10, 3), np.nan)
    y = np.zeros((10, len(LABELS)), dtype=int)
    y[:5, 0] = 1
    probs = fit_predict(X, y, X, "hgb", seed=0)
    assert probs.shape == (10, len(LABELS))
    assert (probs == 0).all()


# --- しきい値を決める検証データの季節 ------------------------------------

def test_val_mode_tail_misses_the_summer(tmp_path):
    """直近をまとめて検証にすると、季節が偏る。

    実測で、検証データに1〜4月と11〜12月しか入らなかった。梅雨や秋雨の
    停滞前線はAUC 0.874の信号があったのにF1は0.516に留まった。しきい値を
    その季節のデータを見ずに決めていたためである。
    """
    import pandas as pd

    from src.split import add_parsed_datetime, make_splits

    days = pd.date_range("2023-01-01", "2025-12-31", freq="D")
    df = add_parsed_datetime(pd.DataFrame({
        "filename": [f"Js_{d.strftime('%Y%m%d')}00.png" for d in days],
        "date": [d.strftime("%Y%m%d") for d in days],
    }))

    tail = make_splits(df, mode="loyo", test_year="2024", val_mode="tail")
    spread = make_splits(df, mode="loyo", test_year="2024", val_mode="spread")

    def months(rows):
        return set(df.loc[rows, "parsed_datetime"].dt.month)

    assert 7 not in months(tail["val"]), "tailで7月が入るなら、この前提が変わっている"
    assert months(spread["val"]) >= set(range(1, 13)), "spreadは通年になるはず"


def test_cv_features_passes_val_mode_through():
    """--val-mode を受け取って make_splits に渡していること。

    渡し忘れると既定のtailになり、季節性の強いラベルのしきい値が偏る。
    黙って悪い結果が出るので、繋がっていることをここで縛る。
    """
    source = Path(__file__).resolve().parent.parent / "scripts" / "cv_features.py"
    text = source.read_text(encoding="utf-8")
    assert '"--val-mode"' in text
    assert "val_mode=args.val_mode" in text


# --- 定義そのものを表す特徴量 --------------------------------------------

def test_futatsudama_needs_lows_in_both_regions():
    """二つ玉低気圧の定義は「日本海側に1つかつ南岸に1つ」。

    src/labels.py の規約が「japan_sea_low と nankigan_low を置き換える」と
    定めている。実測では low_in_japan_sea_low が0.693、low_in_nankigan_low が
    0.605あったのに、計画が「数えるだけの問題」と書いていた n_low は0.622
    しかなかった。定義は数ではなく配置である。
    """
    js, nk = REGIONS["japan_sea_low"], REGIONS["nankigan_low"]
    in_js = ((js.x0 + js.x1) / 2, (js.y0 + js.y1) / 2)
    in_nk = ((nk.x0 + nk.x1) / 2, (nk.y0 + nk.y1) / 2)

    both = build_features(ChartDetections(lows=[in_js, in_nk]), REGIONS)
    assert both["low_in_japan_sea_and_nankigan"] == 1

    # 片方だけ、あるいは同じ領域に2つでは立たない
    for lows in ([in_js], [in_nk], [in_js, in_js]):
        f = build_features(ChartDetections(lows=lows), REGIONS)
        assert f["low_in_japan_sea_and_nankigan"] == 0
        assert f["n_low"] == len(lows)      # 数は数えている


def test_west_high_east_low_is_negative_when_the_high_is_west():
    """西高東低はラベル名がそのまま配置を表す。"""
    west = build_features(
        ChartDetections(highs=[(0.2, 0.5)], lows=[(0.8, 0.5)]), REGIONS)
    east = build_features(
        ChartDetections(highs=[(0.8, 0.5)], lows=[(0.2, 0.5)]), REGIONS)
    assert west["west_high_east_low"] < 0 < east["west_high_east_low"]
    # 片方が無ければ配置は決まらない
    lonely = build_features(ChartDetections(highs=[(0.2, 0.5)]), REGIONS)
    assert math.isnan(lonely["west_high_east_low"])


# --- ばらつきの測り方 ----------------------------------------------------

def test_hist_gradient_boosting_ignores_the_seed():
    """seed を変えても結果は変わらない。ばらつきは seed では測れない。

    HistGradientBoosting は既定で部分抽出をしないので、同じデータからは
    必ず同じ木ができる。binningの部分抽出はn>10000のときだけ、early stopping
    も n<10000 では既定で無効。**seedを3つ回して幅を見るのは無意味である。**
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 8))
    y = (X[:, 0] + rng.normal(scale=0.5, size=400) > 0).astype(int)
    X_test = rng.normal(size=(100, 8))

    probs = []
    for seed in (1, 2, 3):
        model = HistGradientBoostingClassifier(max_iter=50, max_depth=3,
                                               random_state=seed)
        model.fit(X, y)
        probs.append(model.predict_proba(X_test)[:, 1])
    assert np.array_equal(probs[0], probs[1])
    assert np.array_equal(probs[0], probs[2])


def test_bootstrap_moves_the_training_data_not_the_seed():
    """ばらつきは学習データを取り直して測る。幅が返ること。"""
    from scripts.cv_features import bootstrap_spread
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 6))
    y = np.zeros((300, len(LABELS)), dtype=int)
    y[:, 0] = (X[:, 0] > 0).astype(int)
    y[:, 1] = (X[:, 1] > 0).astype(int)
    train, test = list(range(200)), list(range(200, 300))
    thresholds = np.full(len(LABELS), 0.5)

    spread = bootstrap_spread(X, y, train, test, thresholds, "hgb", 42, repeats=4)
    assert spread["repeats"] == 4
    assert spread["min"] <= spread["mean"] <= spread["max"]
    assert spread["std"] >= 0.0
    # ラベル別も返す。macroの幅では、あるラベルの勝ち負けが本物かを言えない
    assert set(spread["per_label_std"]) == set(LABELS)
    assert all(v >= 0.0 for v in spread["per_label_std"].values())


# --- 中心の印の種別 ------------------------------------------------------

def test_the_ring_around_a_cross_cannot_be_templated():
    """丸で囲んだ×を丸ごとテンプレートで拾おうとすると失敗すること。

    **この失敗が「印の形では見分けない」設計の理由**なので、前提が変わって
    いないかをここで縛る。図の側の丸は大きさも太さもまちまちなので、丸ごと
    当てるテンプレートはしきい値を割る。一方、丸の内側の×はいつでも完璧に
    当たるので、低気圧がすべて高気圧として数えられる(実測 高7.70 / 低0.20)。
    """
    import cv2

    from src.chartsymbols import match_templates

    def template(circled: bool):
        tile = np.zeros((80, 80), np.uint8)
        cv2.line(tile, (25, 25), (55, 55), 1, 5)
        cv2.line(tile, (55, 25), (25, 55), 1, 5)
        if circled:
            cv2.circle(tile, (40, 40), 28, 1, 4)
        ys, xs = np.nonzero(tile)
        return tile[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(bool)

    templates = {"cross": template(False), "circle_cross": template(True)}
    labels = []
    for radius, thickness in ((28, 4), (24, 3), (33, 5), (22, 6)):
        chart = np.full((400, 400, 3), 255, np.uint8)
        cv2.line(chart, (185, 185), (215, 215), (4, 4, 4), 5)
        cv2.line(chart, (215, 185), (185, 215), (4, 4, 4), 5)
        cv2.circle(chart, (200, 200), radius, (4, 4, 4), thickness)
        hits = match_templates(chart, templates, threshold=0.65, angles=(0.0,))
        labels.append(hits[0].label if hits else "なし")

    # 丸の大きさがテンプレートと合う1つ以外は、ただの×として拾われてしまう
    assert labels.count("cross") >= 2, labels


def test_marks_take_their_type_from_the_nearest_letter():
    """印の種別は、いちばん近い H / L の文字から取ること。"""
    from src.chartfeatures import assign_marks_to_letters

    marks = [(0.20, 0.30), (0.60, 0.40)]
    letters = {"H": [(0.22, 0.32)], "L": [(0.63, 0.41)]}
    typed, spare, orphans = assign_marks_to_letters(marks, letters, radius=0.08)
    assert typed["H"] == [(0.20, 0.30)]
    assert typed["L"] == [(0.60, 0.40)]
    assert spare == {"H": [], "L": []} and orphans == []


def test_a_letter_serves_only_one_mark():
    """1つの文字が2つの印の種別になってはいけない。

    近い順に1対1で組む。余った印は種別が決まらないので位置として使わない。
    """
    from src.chartfeatures import assign_marks_to_letters

    marks = [(0.50, 0.50), (0.52, 0.50)]
    typed, spare, orphans = assign_marks_to_letters(
        marks, {"H": [(0.51, 0.51)], "L": []}, radius=0.08)
    assert len(typed["H"]) == 1
    assert len(orphans) == 1


def test_letters_without_a_mark_are_the_off_frame_systems():
    """組にならなかった文字は、中心が枠外の系として扱えること。

    中心が図郭の外にある系には×が描かれず、文字だけになる。
    """
    from src.chartfeatures import assign_marks_to_letters

    typed, spare, orphans = assign_marks_to_letters(
        [(0.50, 0.50)], {"H": [(0.51, 0.50)], "L": [(0.02, 0.90)]}, radius=0.08)
    assert typed["H"] == [(0.50, 0.50)]
    assert spare["L"] == [(0.02, 0.90)]      # 印の無い L = 枠外の低気圧


def test_nearest_letter_distances_measures_the_radius():
    """半径は当てずっぽうではなく、実測の分布から決める。"""
    from src.chartfeatures import nearest_letter_distances

    got = nearest_letter_distances([(0.10, 0.10), (0.90, 0.90)],
                                   {"H": [(0.13, 0.14)], "L": []})
    assert got[0] == pytest.approx(0.05, abs=1e-6)
    assert got[1] > 0.5
    assert nearest_letter_distances([(0.1, 0.1)], {"H": [], "L": []}) == []


def test_marks_left_in_the_templates_folder_are_reported(capsys):
    """印を --templates に置くと H/L しか見ないので黙って無視される。"""
    from scripts.build_features import warn_about_misplaced_marks

    warn_about_misplaced_marks({"H": None, "L2": None,
                                "cross": None, "circle_cross": None})
    out = capsys.readouterr().out
    assert "★" in out and "cross" in out and "--marks" in out

    warn_about_misplaced_marks({"H": None, "H2": None, "L": None})
    assert capsys.readouterr().out == ""
