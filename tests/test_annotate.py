"""天気図に検出結果を描き込む部分のテスト。

守っているのは「**壊れても正常に見えるもの**」(tests/README.md)。
描き込みが効いていなくても学習は最後まで走り、それらしい数字が出る。
元画像をそのままコピーしただけでも気づけない。
"""

import numpy as np
import pytest

from scripts.annotate_charts import (
    BOX_H,
    BOX_W,
    FRONT_COLORS,
    HIGH_COLOR,
    LOW_COLOR,
    draw_annotations,
)
from src.chartfeatures import ChartDetections


def blank(h=200, w=200):
    return np.full((h, w, 3), 255, dtype=np.uint8)


def detections(highs=(), lows=()):
    return ChartDetections(highs=list(highs), lows=list(lows),
                           edge_highs=[], edge_lows=[],
                           front_segments={}, stationary_pixels=0)


def test_the_original_image_is_not_modified():
    """元の配列を書き換えないこと。

    並列処理で同じ配列を使い回すので、その場で書き換えると別の画像に
    前の検出結果が混ざる。
    """
    rgb = blank()
    before = rgb.copy()
    draw_annotations(rgb, {"masks": None}, detections(highs=[(0.5, 0.5)]))
    assert np.array_equal(rgb, before)


def test_a_high_gets_a_box_in_its_own_colour():
    rgb = blank()
    out = draw_annotations(rgb, {"masks": None}, detections(highs=[(0.5, 0.5)]))
    painted = np.unique(out.reshape(-1, 3), axis=0).tolist()
    assert list(HIGH_COLOR) in painted
    assert list(LOW_COLOR) not in painted


def test_high_and_low_are_told_apart():
    """高気圧と低気圧が同じ色になっていないこと。

    同じ色だと、人が見ても区別できず、CNNにも区別を渡せない。
    """
    assert HIGH_COLOR != LOW_COLOR
    out = draw_annotations(blank(), {"masks": None},
                           detections(highs=[(0.3, 0.3)], lows=[(0.7, 0.7)]))
    painted = np.unique(out.reshape(-1, 3), axis=0).tolist()
    assert list(HIGH_COLOR) in painted and list(LOW_COLOR) in painted


def test_the_box_is_drawn_as_an_outline_not_filled():
    """枠は輪郭だけにすること。

    **塗りつぶすと下の天気図が消える。**強調のつもりが情報の削除になる。
    """
    out = draw_annotations(blank(), {"masks": None}, detections(highs=[(0.5, 0.5)]))
    h, w = out.shape[:2]
    centre = out[h // 2, w // 2].tolist()
    assert centre == [255, 255, 255], "枠の内側が塗りつぶされている"


def test_the_box_lands_where_the_detection_says():
    """枠が検出位置に描かれること。位置がずれたら人が確かめる意味がなくなる。"""
    out = draw_annotations(blank(400, 400), {"masks": None},
                           detections(highs=[(0.25, 0.75)]))
    ys, xs = np.nonzero(np.all(out == HIGH_COLOR, axis=2))
    assert 0.25 == pytest.approx(xs.mean() / 400, abs=0.02)
    assert 0.75 == pytest.approx(ys.mean() / 400, abs=0.02)


def test_fronts_are_painted_in_their_own_colours():
    mask = np.zeros((200, 200), dtype=bool)
    mask[100, 50:150] = True
    out = draw_annotations(blank(), {"masks": {"stationary_front": mask}}, detections())
    painted = np.unique(out.reshape(-1, 3), axis=0).tolist()
    assert list(FRONT_COLORS["stationary_front"]) in painted


def test_thin_fronts_are_thickened():
    """1画素の線は224x224に縮めた時点で消える。太らせること。"""
    mask = np.zeros((200, 200), dtype=bool)
    mask[100, 50:150] = True
    out = draw_annotations(blank(), {"masks": {"stationary_front": mask}},
                           detections(), thickness=5)
    painted = np.all(out == FRONT_COLORS["stationary_front"], axis=2)
    assert painted.sum() > mask.sum() * 2, "太らせていない"


def test_the_front_pixels_themselves_are_left_alone():
    """前線そのものは塗り替えず、周りに縁取りだけを描くこと。

    **塗り替えると、天気図が元から持っている赤(温暖)・青(寒冷)の区別が
    消える。**強調のつもりが情報の削除になる。しかも学習は完走するので
    数字を見ても気づけない。
    """
    rgb = blank()
    mask = np.zeros((200, 200), dtype=bool)
    mask[100, 50:150] = True
    rgb[mask] = (252, 4, 4)          # 元の温暖前線(赤)
    out = draw_annotations(rgb, {"masks": {"warm_front": mask}}, detections())

    assert np.all(out[mask] == (252, 4, 4)), "前線の画素が塗り替えられている"
    halo = np.all(out == FRONT_COLORS["warm_front"], axis=2)
    assert halo.any(), "縁取りが描かれていない"
    assert not (halo & mask).any(), "縁取りが前線の上に乗っている"


def test_switches_turn_each_layer_off():
    report = {"masks": {"stationary_front": np.ones((200, 200), dtype=bool)}}
    only_boxes = draw_annotations(blank(), report, detections(highs=[(0.5, 0.5)]),
                                  fronts=False)
    assert list(FRONT_COLORS["stationary_front"]) not in \
        np.unique(only_boxes.reshape(-1, 3), axis=0).tolist()
    only_fronts = draw_annotations(blank(), report, detections(highs=[(0.5, 0.5)]),
                                   boxes=False)
    assert list(HIGH_COLOR) not in np.unique(only_fronts.reshape(-1, 3), axis=0).tolist()


def test_annotation_colours_do_not_collide_with_the_chart():
    """描き込みの色が、天気図に既にある色と重ならないこと。

    重なると、色マスクによる検出(方法③)がこの画像に対して使えなくなり、
    人が見ても元からある線か描き込みか区別できない。
    """
    from src.chartsymbols import DEFAULT_BANDS, to_hsv

    used = [HIGH_COLOR, LOW_COLOR, *FRONT_COLORS.values()]
    patch = np.array(used, dtype=np.uint8).reshape(1, -1, 3)
    hsv = to_hsv(patch)
    for name, band in DEFAULT_BANDS.items():
        hit = band.mask(hsv)[0]
        clashing = [used[i] for i in np.nonzero(hit)[0]]
        assert not clashing, f"{name} の色帯に入る描き込み色がある: {clashing}"


def test_one_fronts_halo_does_not_paint_over_another_front():
    """あとから描く縁取りが、別の前線の元画素を塗らないこと。

    停滞前線は赤と青が交互に並んだ形なので、その縁取りは温暖前線・寒冷前線の
    すぐ隣を通る。自分のマスクだけ避けると、隣の前線を1割ほど塗りつぶした。
    """
    rgb = blank()
    warm = np.zeros((200, 200), dtype=bool)
    warm[100, 40:60] = True
    stationary = np.zeros((200, 200), dtype=bool)
    stationary[100, 62:80] = True       # 温暖前線のすぐ隣
    rgb[warm] = (252, 4, 4)

    out = draw_annotations(rgb, {"masks": {"warm_front": warm,
                                           "stationary_front": stationary}},
                           detections(), thickness=5)
    assert np.all(out[warm] == (252, 4, 4)), "停滞前線の縁取りが温暖前線を塗っている"
