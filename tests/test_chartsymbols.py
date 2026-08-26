"""色マスクと記号拾いのテスト。

本物の天気図はこのリポジトリに入っていない(`data/raw/` は追跡していない)ので、
気象庁の配色どおりに描いた合成天気図で確かめる。**確かめているのはコードが
仕様どおり動くことだけで、本物の天気図で前線が取れるかどうかではない。**
本物での可否は `scripts/chart_palette.py` で色を測るところから始まる。
"""

import cv2
import numpy as np
import pytest

from src.chartsymbols import (
    Candidate,
    ColorBand,
    band_overlap,
    color_masks,
    crop_template,
    dominant_colors,
    glyph_candidates,
    match_templates,
    segments,
    stationary_mask,
)

# 気象庁 JSMAP の配色に寄せた色(RGB)。実測値ではなく、仕様どおりの色。
BLACK = (20, 20, 20)          # 等圧線
COASTLINE = (130, 70, 45)     # 海岸線・経緯度線=赤茶色
WARM = (220, 30, 30)          # 温暖前線=赤
COLD = (30, 60, 220)          # 寒冷前線=青
OCCLUDED = (230, 90, 200)     # 閉塞前線=ピンク

SIZE = 400


def blank() -> np.ndarray:
    return np.full((SIZE, SIZE, 3), 255, dtype=np.uint8)


def line(img, p0, p1, color, thickness=3):
    cv2.line(img, p0, p1, color, thickness, lineType=cv2.LINE_8)


def synthetic_chart() -> np.ndarray:
    """前線4種・海岸線・等圧線・記号を1枚に詰めた合成天気図。"""
    img = blank()
    # 等圧線: 端から端まで伸びる黒い曲線。連結成分としては巨大になる
    for offset in (60, 140, 220, 300):
        points = np.array([[x, offset + int(18 * np.sin(x / 40.0))] for x in range(0, SIZE)])
        cv2.polylines(img, [points], False, BLACK, 2, lineType=cv2.LINE_8)
    # 海岸線: 赤茶色。温暖前線の赤と混ざってはいけない
    line(img, (20, 20), (20, SIZE - 20), COASTLINE, 2)
    line(img, (20, 20), (SIZE - 20, 20), COASTLINE, 2)
    # 温暖前線(赤)・寒冷前線(青)・閉塞前線(ピンク): 互いに離して描く
    line(img, (40, 360), (200, 360), WARM)
    line(img, (240, 380), (380, 380), COLD)
    line(img, (40, 100), (180, 130), OCCLUDED)
    # 停滞前線: 同じ線の上で赤と青を交互に
    for i in range(8):
        x0 = 40 + i * 40
        color = WARM if i % 2 == 0 else COLD
        line(img, (x0, 250), (x0 + 38, 250), color)
    return img


def put_glyph(img, text, org, scale=0.9, thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK,
                thickness, lineType=cv2.LINE_8)


# --- ColorBand ---------------------------------------------------------

def test_hue_band_wraps_around_zero():
    """赤は色相0をまたぐ。h_min > h_max を巻き戻しとして解釈すること。"""
    band = ColorBand("red", h_min=172, h_max=8, s_min=0, v_min=0)
    hsv = np.array([[[0, 255, 255], [175, 255, 255], [90, 255, 255]]], dtype=np.uint8)
    assert band.mask(hsv).tolist() == [[True, True, False]]


def test_plain_hue_band_does_not_wrap():
    band = ColorBand("blue", h_min=100, h_max=130)
    hsv = np.array([[[115, 255, 255], [0, 255, 255]]], dtype=np.uint8)
    assert band.mask(hsv).tolist() == [[True, False]]


# --- 前線の色分け -------------------------------------------------------

def test_front_colors_are_separated():
    """4色がそれぞれ自分のマスクにだけ入ること。"""
    img = blank()
    line(img, (10, 50), (390, 50), WARM)
    line(img, (10, 150), (390, 150), COLD)
    line(img, (10, 250), (390, 250), OCCLUDED)
    line(img, (10, 350), (390, 350), COASTLINE)
    masks = color_masks(img)
    for name, row in (("warm_front", 50), ("cold_front", 150),
                      ("occluded_front", 250), ("coastline", 350)):
        assert masks[name][row].any(), f"{name} が自分の線を拾えていない"
        for other in ("warm_front", "cold_front", "occluded_front", "coastline"):
            if other != name:
                assert not masks[other][row].any(), f"{other} が {name} の線を拾っている"


def test_warm_front_does_not_pick_up_coastline():
    """この方式の成否を決める1点。赤(温暖前線)と赤茶色(海岸線)を分けられること。"""
    masks = color_masks(synthetic_chart())
    overlap = band_overlap(masks)
    assert overlap.get(("coastline", "warm_front"), 0) == 0


def test_isobars_are_black_not_front():
    masks = color_masks(synthetic_chart())
    assert masks["isobar"].any()
    for name in ("warm_front", "cold_front", "occluded_front"):
        assert not (masks[name] & masks["isobar"]).any()


# --- 前線の形 -----------------------------------------------------------

def test_segments_report_elongated_shapes():
    img = blank()
    line(img, (20, 200), (380, 200), WARM, 3)
    found = segments(color_masks(img)["warm_front"], "warm_front")
    assert len(found) == 1
    assert found[0].is_frontlike
    assert found[0].length == pytest.approx(360, abs=30)
    assert found[0].cx == pytest.approx(0.5, abs=0.05)


def test_round_blob_is_not_frontlike():
    """凡例の四角や文字は細長くないので、前線と数えない。"""
    img = blank()
    cv2.circle(img, (200, 200), 12, WARM, -1)
    found = segments(color_masks(img)["warm_front"], "warm_front")
    assert len(found) == 1
    assert not found[0].is_frontlike


# --- 停滞前線 -----------------------------------------------------------

def test_stationary_fires_on_alternating_red_and_blue():
    masks = color_masks(synthetic_chart())
    stationary = stationary_mask(masks["warm_front"], masks["cold_front"])
    rows = np.nonzero(stationary.any(axis=1))[0]
    assert rows.size > 0
    assert 245 <= rows.mean() <= 255, "交互に描いた行(250)で反応していない"


def test_stationary_ignores_lone_warm_front():
    """赤だけの線(温暖前線)は停滞前線にしない。"""
    img = blank()
    line(img, (20, 200), (380, 200), WARM)
    masks = color_masks(img)
    assert not stationary_mask(masks["warm_front"], masks["cold_front"]).any()


# --- 記号の候補 ---------------------------------------------------------

def test_glyph_candidates_find_letters_and_skip_isobars():
    img = synthetic_chart()
    put_glyph(img, "H", (100, 200))
    put_glyph(img, "L", (300, 200))
    found = glyph_candidates(img)
    # 描いた2文字が、その位置で拾えていること
    centers = [(round(c.cx, 2), round(c.cy, 2)) for c in found]
    assert any(abs(x - 0.27) < 0.03 and abs(y - 0.48) < 0.03 for x, y in centers), "Hを拾えていない"
    assert any(abs(x - 0.77) < 0.03 and abs(y - 0.48) < 0.03 for x, y in centers), "Lを拾えていない"
    # 等圧線は図の端から端まで繋がった巨大な成分なので、まるごと候補に入ることはない
    assert all(c.width <= 48 and c.height <= 48 for c in found)


def test_colored_line_across_an_isobar_leaves_a_false_candidate():
    """色の線が等圧線を上書きすると、切れた等圧線の断片が候補に混じる。

    合成天気図では海岸線が等圧線を横切り、その左側の短い切れ端が4つ残る。
    候補拾いは「記号だけを拾う」ものではなく、「テンプレートを当てる先を
    1枚あたり数十個に絞る」ものだと分かる。取捨はテンプレート側の仕事。
    """
    found = glyph_candidates(synthetic_chart())
    stubs = [c for c in found if c.x0 == 0]
    assert len(stubs) == 4


def test_glyph_candidates_ignore_long_curves():
    img = blank()
    points = np.array([[x, 200 + int(30 * np.sin(x / 30.0))] for x in range(0, SIZE)])
    cv2.polylines(img, [points], False, BLACK, 2, lineType=cv2.LINE_8)
    assert glyph_candidates(img) == []


# --- テンプレートマッチング ----------------------------------------------

def test_template_matching_locates_the_same_glyph_elsewhere():
    """1つの記号から切り出したテンプレートで、別の場所の同じ記号を見つけられること。"""
    source = blank()
    put_glyph(source, "H", (100, 100))
    box = glyph_candidates(source)[0]
    template = crop_template(source, (box.x0, box.y0, box.x1, box.y1))

    target = blank()
    put_glyph(target, "H", (250, 300))
    put_glyph(target, "L", (60, 300))
    hits = match_templates(target, {"H": template}, threshold=0.7)
    assert [h.label for h in hits] == ["H"], "Hが1つだけ当たるはず(Lに当たってはいけない)"
    assert hits[0].cx == pytest.approx(0.66, abs=0.06)
    assert hits[0].cy == pytest.approx(0.73, abs=0.06)


def test_overlapping_hits_are_suppressed():
    source = blank()
    put_glyph(source, "H", (100, 100))
    box = glyph_candidates(source)[0]
    template = crop_template(source, (box.x0, box.y0, box.x1, box.y1))
    # 同じテンプレートを2つの名前で当てても、同じ場所は1つに畳まれる
    hits = match_templates(source, {"H": template, "H_dup": template}, threshold=0.7)
    assert len(hits) == 1


def test_nested_hit_inside_a_bigger_glyph_is_suppressed():
    """"H" の右の縦棒は "L" によく似る。入れ子の誤検出を残さないこと。

    IoUで抑えると、小さいLの箱は大きいHの箱と重なりが小さく出るので生き残る。
    小さいほうの面積を基準にすれば落ちる。
    """
    source = blank()
    put_glyph(source, "H", (100, 100))
    put_glyph(source, "L", (250, 100))
    boxes = {c.width * c.height: c for c in glyph_candidates(source)}
    h_box, l_box = boxes[max(boxes)], boxes[min(boxes)]
    templates = {
        "H": crop_template(source, (h_box.x0, h_box.y0, h_box.x1, h_box.y1)),
        "L": crop_template(source, (l_box.x0, l_box.y0, l_box.x1, l_box.y1)),
    }
    # 閾値を下げると L が H の中にも当たるが、抑制で1つに落ちる
    hits = match_templates(source, templates, threshold=0.7)
    assert sorted(h.label for h in hits) == ["H", "L"]


def test_default_threshold_is_strict_enough_for_nested_glyphs():
    source = blank()
    put_glyph(source, "H", (100, 100))
    box = glyph_candidates(source)[0]
    template = crop_template(source, (box.x0, box.y0, box.x1, box.y1))
    # 既定の閾値では、Hのテンプレートは自分自身にしか当たらない
    assert len(match_templates(source, {"H": template})) == 1


# --- 色の実測 -----------------------------------------------------------

def test_dominant_colors_reports_the_drawn_colors():
    img = blank()
    line(img, (10, 50), (390, 50), WARM, 5)
    line(img, (10, 150), (390, 150), COLD, 3)
    top = dominant_colors(img, top=4)
    assert top[0]["rgb"][0] > 200 and top[0]["rgb"][2] < 60, "一番多いのは赤のはず"
    assert top[0]["pixels"] > top[1]["pixels"]


def test_dominant_colors_skips_the_paper_white():
    top = dominant_colors(blank(), top=4)
    assert top == []
