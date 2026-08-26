"""色マスクと記号拾いのテスト。

本物の天気図はこのリポジトリに入っていない(`data/raw/` は追跡していない)ので、
気象庁の配色どおりに描いた合成天気図で確かめる。**確かめているのはコードが
仕様どおり動くことだけで、本物の天気図で前線が取れるかどうかではない。**
本物での可否は `scripts/chart_palette.py` で色を測るところから始まる。
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.chartsymbols import (
    Candidate,
    ColorBand,
    band_overlap,
    band_variation,
    FURNITURE_STABILITY,
    label_auc,
    MaskAccumulator,
    color_masks,
    crop_template,
    dominant_colors,
    glyph_candidates,
    match_templates,
    segments,
    stationary_mask,
)

# 2023年1月の天気図8枚から実測した色(RGB)。docs/2026-08-26-detection-prescreen.md 参照。
# WARM と OCCLUDED は真冬の8枚に出てこなかったので、純青からの類推。
BLACK = (4, 4, 4)             # 等圧線
COASTLINE = (164, 44, 44)     # 海岸線・経緯度線=くすんだ赤茶 HSV(0,187,164)
COASTLINE_EDGE = (172, 60, 60)  # そのにじみ HSV(0,166,172)
WARM = (252, 4, 4)            # 温暖前線=純赤 HSV(0,251,252)
COLD = (4, 4, 252)            # 寒冷前線=純青 HSV(120,251,252)
OCCLUDED = (252, 4, 252)      # 閉塞前線=マゼンタ HSV(150,251,252)

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


# --- 実測した色の切り分け ---------------------------------------------

@pytest.mark.parametrize("rgb, expected", [
    ((164, 44, 44), "coastline"),        # 海岸線の芯
    ((172, 60, 60), "coastline"),        # 海岸線のにじみ
    ((204, 132, 132), None),             # 白へのにじみ。前線に数えてはいけない
    ((236, 204, 204), None),
    ((4, 4, 252), "cold_front"),         # 純青
    ((252, 4, 4), "warm_front"),         # 純赤
    ((252, 4, 252), "occluded_front"),   # マゼンタ
    ((4, 4, 4), "isobar"),
    ((164, 164, 164), None),             # 黒のにじみ(灰)。彩度が無いので色ではない
])
def test_measured_colors_land_in_one_band(rgb, expected):
    """実測した色が、それぞれ狙った帯にちょうど1つだけ入ること。

    海岸線(164,44,44)と温暖前線(252,4,4)はどちらも色相0で、**色相では
    分けられない**。明度で分けている(164 対 252)ので、その線引きが
    崩れていないかをここで守る。
    """
    from src.chartsymbols import DEFAULT_BANDS
    hsv = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)
    hit = [name for name, band in DEFAULT_BANDS.items() if band.mask(hsv)[0, 0]]
    assert hit == ([expected] if expected else [])


def test_coastline_is_not_a_warm_front():
    """最初の暫定値で起きた取り違えそのもの。海岸線を温暖前線に数えないこと。"""
    img = blank()
    line(img, (10, 100), (390, 100), COASTLINE, 3)
    line(img, (10, 200), (390, 200), COASTLINE_EDGE, 3)
    masks = color_masks(img)
    assert not masks["warm_front"].any()
    assert masks["coastline"].any()


# --- 地図の備品と気象の切り分け -----------------------------------------

def test_variation_flags_a_band_that_never_changes():
    """毎回ほぼ同じ画素数の帯は、気象ではなく地図の備品を掴んでいる。

    数値は2023年1月の8枚の実測値。暫定値のとき warm_front は海岸線を
    掴んでいて、画素数がほとんど動かなかった。
    """
    stats = band_variation({
        "warm_front": [59567, 60093, 58245, 59206, 58532, 56322, 56891, 57485],
        "cold_front": [3136, 4022, 0, 3692, 0, 0, 0, 0],
    })
    assert stats["warm_front"]["looks_like_furniture"]
    assert not stats["cold_front"]["looks_like_furniture"]
    assert stats["warm_front"]["cv"] < 0.05
    assert stats["cold_front"]["cv"] > 1.0


def test_variation_does_not_flag_an_empty_band():
    """空の帯は「備品」ではない。指摘の文言が変わるので分けておく。"""
    stats = band_variation({"occluded_front": [0, 0, 0, 0]})
    assert not stats["occluded_front"]["looks_like_furniture"]
    assert stats["occluded_front"]["mean"] == 0


def test_overlap_alone_cannot_catch_the_mixup():
    """重なりが0でも安心できないことを、実際に0になる形で示す。

    片方の帯が空なら重なりは必ず0になる。だから band_overlap の0を
    「分離できている」と読んではいけない。band_variation と併せて見る。
    """
    img = blank()
    line(img, (10, 100), (390, 100), COASTLINE, 3)
    empty = np.zeros(img.shape[:2], dtype=bool)
    everything = color_masks(img)["coastline"]
    assert band_overlap({"a": everything, "b": empty}) == {}


# --- 位置で備品と気象を分ける -------------------------------------------

def test_stability_separates_furniture_from_weather():
    """毎回同じ画素なら備品、動くなら気象。等圧線の誤判定を直した判定。

    画素数の変動係数だけでは等圧線も備品に見えてしまう。常に図全体を覆う
    ので総量が動かないからで、実測でも1月0.036・7月0.117と、海岸線
    (0.010)と見分けが付かなかった。位置を見れば分かれる。
    """
    acc = MaskAccumulator()
    coastline = np.zeros((20, 20), dtype=bool)
    coastline[0, :] = True                      # 毎回まったく同じ場所
    for i in range(5):
        isobar = np.zeros((20, 20), dtype=bool)
        isobar[5 + i, :] = True                 # 総量は同じだが場所が動く
        acc.add({"coastline": coastline, "isobar": isobar})

    stability = acc.stability()
    assert stability["coastline"]["typical"] == pytest.approx(1.0)
    assert stability["isobar"]["typical"] == pytest.approx(0.2)

    # 総量だけでは両者が同じに見えることも示しておく
    stats = band_variation({"coastline": [20] * 5, "isobar": [20] * 5})
    assert stats["coastline"]["looks_like_furniture"]
    assert stats["isobar"]["looks_like_furniture"]


def test_accumulator_refuses_to_mix_sizes():
    """大きさの違う画像は足し合わせない。実測で1画素違う組があった。"""
    acc = MaskAccumulator()
    acc.add({"a": np.ones((10, 10), dtype=bool)})
    acc.add({"a": np.ones((10, 9), dtype=bool)})
    assert acc.size_mismatch
    assert acc.n_images == 1


# --- 測定値とラベルの対応 -----------------------------------------------

def test_auc_is_one_when_the_measure_separates_perfectly():
    assert label_auc([9, 8, 7, 3, 2, 1], [True, True, True, False, False, False]) == 1.0


def test_auc_is_half_when_the_measure_is_useless():
    assert label_auc([5, 5, 5, 5], [True, True, False, False]) == 0.5


def test_auc_below_half_means_inverted_but_still_informative():
    """低いほどラベルが付く関係も信号である。実測の japan_sea_low が0.101だった。"""
    assert label_auc([1, 2, 8, 9], [True, True, False, False]) == 0.0


def test_auc_reproduces_the_measured_stationary_front_result():
    """2024年7月の20枚の実測値。停滞pxで stationary_front を分けたときのAUC。"""
    stationary_px = [2382, 2303, 2807, 2816, 4032, 3924, 1868, 2448, 1697, 126,
                     120, 1084, 1453, 2447, 2792, 2890, 2734, 2400, 1724, 2164]
    has_label = [True, True, True, True, True, True, False, False, False, False,
                 False, False, False, True, True, True, True, True, True, True]
    assert label_auc(stationary_px, has_label) == pytest.approx(0.923, abs=0.001)


def test_auc_is_nan_without_both_classes():
    assert np.isnan(label_auc([1, 2, 3], [True, True, True]))


# --- Windowsの既定コンソール(cp932)で出せること ---------------------------

def test_scripts_print_only_characters_cp932_can_show():
    """日本語Windowsの既定コンソールはcp932。出せない文字があるとその場で落ちる。

    em-dash(U+2014)を使っていて、表を出し終えた最後の1行で
    UnicodeEncodeError になった。cp932 には全角ダッシュ(U+2015)があるので
    そちらを使う。絵文字や✓・⚠も同じ理由で使えない。
    """
    root = Path(__file__).resolve().parent.parent
    targets = [
        root / "src" / "chartsymbols.py",
        root / "scripts" / "chart_palette.py",
        root / "scripts" / "extract_fronts.py",
        root / "scripts" / "extract_symbols.py",
    ]
    offenders = []
    for path in targets:
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for ch in line:
                try:
                    ch.encode("cp932")
                except UnicodeEncodeError:
                    offenders.append(f"{path.name}:{lineno} {ch!r} U+{ord(ch):04X}")
    assert not offenders, "cp932で出せない文字がある: " + ", ".join(offenders)


def test_occlusion_does_not_hide_furniture_from_the_metric():
    """上書きで日ごとに違う所が隠れても、備品は備品と判定できること。

    海岸線は毎日同じ場所にあるが、等圧線や前線がその上に描かれる。実測では
    1枚あたり5.1%が隠れ、12枚での「毎回点灯」は0.533まで落ちた。毎日そこに
    あるのに毎回点灯しないので、毎回点灯を判定に使うと備品を取り逃がす。
    中央値なら隠れは効かない。
    """
    rng = np.random.default_rng(0)
    acc = MaskAccumulator()
    coastline = np.zeros((200, 200), dtype=bool)
    coastline[::7, :] = True
    for _ in range(12):
        acc.add({"coastline": coastline & (rng.random(coastline.shape) > 0.051)})

    detail = acc.stability()["coastline"]
    assert detail["typical"] == pytest.approx(1.0)
    assert detail["always"] == pytest.approx(0.533, abs=0.03)   # 実測と同じ値になる
    assert detail["typical"] >= FURNITURE_STABILITY             # 備品と判定できる
    assert detail["always"] < FURNITURE_STABILITY               # 旧指標では取り逃がす
