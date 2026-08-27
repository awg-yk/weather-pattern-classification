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
from PIL import Image

from src.chartsymbols import (
    Candidate,
    ColorBand,
    band_overlap,
    band_variation,
    cluster_patches,
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


# --- 大きさのまとめ -----------------------------------------------------

def test_size_cluster_groups_nearby_sizes():
    """記号は線の太さの丸めで1〜2画素ゆれる。34x57 と 36x56 は同じ記号。"""
    from collections import Counter

    from scripts.extract_symbols import size_cluster

    sizes = Counter({(34, 57): 5, (36, 56): 5, (37, 56): 3, (39, 55): 4,
                     (17, 24): 4, (14, 26): 4})
    center, votes = size_cluster(sizes)
    assert abs(center[0] - 36) <= 3 and abs(center[1] - 56) <= 3
    assert votes == 17          # 背の高い4種類がまとまる


# --- 形でまとめる -------------------------------------------------------

def glyph_patches(img, size=(24, 32)):
    from src.chartsymbols import DEFAULT_BANDS, patch_of, to_hsv
    mask = DEFAULT_BANDS["isobar"].mask(to_hsv(img))
    return [patch_of(mask, c, size) for c in glyph_candidates(img)]


def test_clustering_separates_different_letters():
    """同じ記号どうしがまとまり、違う記号は別の山になること。

    大きな天気図から番号で箱を探す代わりに、山ごとの代表を見て名前を付ける。
    どの山がHでどれがLかは機械には分からないので、そこは人の仕事。
    """
    img = blank()
    for x, y in [(60, 80), (200, 80), (340, 80), (60, 180), (200, 180)]:
        put_glyph(img, "H", (x, y))
    for x, y in [(60, 280), (160, 280), (260, 280), (360, 280)]:
        put_glyph(img, "L", (x, y))
    for x, y in [(60, 380), (200, 380), (340, 380)]:
        put_glyph(img, "T", (x, y))

    clusters = cluster_patches(glyph_patches(img), threshold=0.7)
    assert [c["size"] for c in clusters] == [5, 4, 3]
    # 同じ山の members が同じ記号であること(描いた順に0-4=H, 5-8=L, 9-11=T)
    assert set(clusters[0]["members"]) == {0, 1, 2, 3, 4}
    assert set(clusters[1]["members"]) == {5, 6, 7, 8}
    assert set(clusters[2]["members"]) == {9, 10, 11}


def test_clustering_of_one_repeated_glyph_gives_one_cluster():
    img = blank()
    for x in (60, 160, 260, 360):
        put_glyph(img, "H", (x, 200))
    clusters = cluster_patches(glyph_patches(img), threshold=0.7)
    assert len(clusters) == 1 and clusters[0]["size"] == 4


def test_clustering_handles_no_candidates():
    assert cluster_patches([]) == []


def test_patch_is_resized_to_the_common_size():
    """線の太さの丸めで1〜2画素ゆれるので、比べる前に大きさを揃える。"""
    from src.chartsymbols import DEFAULT_BANDS, patch_of, to_hsv
    img = blank()
    put_glyph(img, "H", (100, 100))
    mask = DEFAULT_BANDS["isobar"].mask(to_hsv(img))
    patch = patch_of(mask, glyph_candidates(img)[0], (24, 32))
    assert patch.shape == (32, 24)


def test_correlation_tells_the_same_glyph_from_a_different_one():
    """山どうしが同じ記号か別の記号かを見分ける根拠。"""
    from src.chartsymbols import correlation
    img = blank()
    put_glyph(img, "H", (100, 100))
    put_glyph(img, "H", (250, 100))
    put_glyph(img, "L", (100, 250))
    a, b, c = glyph_patches(img)[:3]
    assert correlation(a, b) > 0.9     # 同じ記号
    assert correlation(a, c) < 0.5     # 違う記号


def test_threshold_sweep_prefers_the_stricter_value_on_a_tie():
    """同じ山数で覆えるなら、しきい値は高いほうを採る(別の記号を混ぜにくい)。"""
    import io
    from contextlib import redirect_stdout

    from scripts.extract_symbols import report_threshold_sweep

    img = blank()
    for x in (60, 160, 260):
        put_glyph(img, "H", (x, 200))
    for x in (60, 160, 260):
        put_glyph(img, "L", (x, 320))

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        report_threshold_sweep(glyph_patches(img), 0.7)
    out = buffer.getvalue()
    # どのしきい値でも同じ山数になるので、一番高い0.8が選ばれる
    assert "しきい値 0.8" in out


# --- コピペで進めても壊れないこと ---------------------------------------

def test_match_refuses_when_nothing_was_ever_named(tmp_path):
    """clusterNN しか無いなら、名前を付ける手順が抜けているので止める。"""
    from scripts.extract_symbols import load_templates
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(tmp_path / "cluster00.png")
    with pytest.raises(SystemExit, match="名前を付けていない"):
        load_templates(tmp_path)


def test_named_templates_win_over_leftover_clusters(tmp_path):
    """名前を付けたものが1つでもあれば、番号のままの山は使わず先へ進む。

    cut で H.png と L.png を作っても、前の cluster の山が残っているだけで
    止まっていた。名前を付けた側があるなら、それを使えばよい。
    """
    from scripts.extract_symbols import load_templates
    white = Image.fromarray(np.full((8, 8), 255, dtype=np.uint8))
    for name in ("H", "L", "cluster00", "cluster07"):
        white.save(tmp_path / f"{name}.png")
    assert sorted(load_templates(tmp_path)) == ["H", "L"]


def test_the_contact_sheet_is_not_a_template(tmp_path):
    """一覧(clusters.png)は見るための画像で、テンプレートではない。

    名前が cluster で始まるので、番号のままの山と一緒に扱われていた。
    """
    from scripts.extract_symbols import load_templates
    white = Image.fromarray(np.full((8, 8), 255, dtype=np.uint8))
    white.save(tmp_path / "clusters.png")
    white.save(tmp_path / "H.png")
    assert sorted(load_templates(tmp_path)) == ["H"]

    # 一覧しか無いときは「テンプレートが無い」になること
    (tmp_path / "H.png").unlink()
    with pytest.raises(SystemExit, match="テンプレートがありません"):
        load_templates(tmp_path)


def test_match_accepts_named_templates(tmp_path):
    from scripts.extract_symbols import load_templates
    Image.fromarray(np.full((8, 8), 255, dtype=np.uint8)).save(tmp_path / "H.png")
    assert list(load_templates(tmp_path)) == ["H"]


def test_cluster_removes_the_previous_run(tmp_path):
    """条件を変えて実行し直すと山の数が減ることがある。前回の余りを残さない。

    残すと match がディレクトリの中を全部読むので、条件の違う山が混ざった
    まま照合してしまう。実際にそれで無効な結果が出た。
    """
    import argparse

    from scripts.extract_symbols import cmd_cluster

    charts = tmp_path / "charts"
    charts.mkdir()
    img = blank()
    for x in (60, 160, 260):
        put_glyph(img, "H", (x, 200))
    Image.fromarray(img).save(charts / "Js_2024070100.png")

    out = tmp_path / "templates"
    out.mkdir()
    (out / "cluster07.png").write_bytes(b"")      # 前回の余り
    (out / "H.png").write_bytes(b"")              # 人が付けた名前は残す

    cmd_cluster(argparse.Namespace(
        in_dir=charts, limit=1, out=out, size=None, tolerance=3, threshold=0.5,
        min_cluster=1, min_side=6, max_side=64, patch_width=24, patch_height=32,
        band="isobar",
    ))
    assert not (out / "cluster07.png").exists()
    assert (out / "H.png").exists()


# --- 色付きの記号 -------------------------------------------------------

def put_colored_glyph(img, text, org, color, scale=1.2, thickness=3):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thickness, lineType=cv2.LINE_8)


def test_glyph_search_can_look_in_a_colored_band():
    """高気圧・低気圧の記号は色付きのことがある。黒だけ見ていると拾えない。

    実測では、黒の候補は数字と等圧線の切れ端と×印だけで、HもLも出てこな
    かった。色を指定して探せるようにした。
    """
    img = blank()
    put_colored_glyph(img, "L", (100, 150), WARM)
    put_colored_glyph(img, "H", (250, 150), COLD)

    assert glyph_candidates(img, band="isobar") == []
    assert len(glyph_candidates(img, band="warm_front")) == 1
    assert len(glyph_candidates(img, band="cold_front")) == 1


def test_compact_share_flags_letters_drawn_in_a_front_color():
    """前線と同じ色の文字は前線の画素数に混じる。その量を測れること。"""
    from src.chartsymbols import compact_share

    # 前線だけ: 細長いので混入は0
    only_front = blank()
    line(only_front, (20, 200), (380, 200), WARM, 3)
    assert compact_share(color_masks(only_front)["warm_front"]) == pytest.approx(0.0)

    # 文字だけ: 丸いので全部が混入
    only_text = blank()
    put_colored_glyph(only_text, "L", (100, 150), WARM)
    assert compact_share(color_masks(only_text)["warm_front"]) == pytest.approx(1.0)

    # 両方: あいだの値になる
    both = blank()
    line(both, (20, 300), (380, 300), WARM, 3)
    put_colored_glyph(both, "L", (100, 150), WARM)
    share = compact_share(color_masks(both)["warm_front"])
    assert 0.0 < share < 1.0


# --- 中抜きの記号を等圧線から切り離す -----------------------------------

def outlined_glyph_crossed_by_an_isobar():
    """太い中抜きの記号を、細い等圧線が横切っている図。実際の天気図の形。"""
    img = np.full((300, 300, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (160, 200), BLACK, 5)      # 中抜きの記号
    cv2.line(img, (0, 150), (299, 150), BLACK, 1, cv2.LINE_8)  # 細い等圧線
    return img


def test_crossing_isobar_hides_an_outlined_glyph():
    """細らせないと、記号は等圧線と繋がって巨大な成分の一部になり落ちる。

    実測でこれが起きていた。黒の候補に H も L も出てこず、数字と等圧線の
    切れ端と×印しか残らなかった。
    """
    assert glyph_candidates(outlined_glyph_crossed_by_an_isobar(), max_side=120) == []


def test_eroding_recovers_the_outlined_glyph():
    """記号の線は等圧線より太いので、細らせると細い線だけが先に消える。"""
    found = glyph_candidates(outlined_glyph_crossed_by_an_isobar(), max_side=120, erode=1)
    assert len(found) == 1
    # 細らせたぶんを戻して、元の大きさに近い枠が返ること
    assert found[0].width == pytest.approx(65, abs=6)
    assert found[0].height == pytest.approx(105, abs=6)


def test_eroding_too_much_breaks_the_glyph_apart():
    """やりすぎると記号自体が切れて、線ごとにばらばらになる。"""
    found = glyph_candidates(outlined_glyph_crossed_by_an_isobar(), max_side=120, erode=3)
    assert len(found) > 1


# --- 等圧線と繋がった記号はテンプレートでしか取れない --------------------

def outlined_glyphs(with_thick_isobar: bool):
    """中抜きの記号3個。太い等圧線が横切る版と、横切らない版。"""
    img = np.full((400, 600, 3), 255, dtype=np.uint8)
    for x in (80, 300, 480):
        cv2.rectangle(img, (x, 150), (x + 60, 250), BLACK, 6)
    if with_thick_isobar:
        cv2.line(img, (0, 200), (599, 200), BLACK, 4, lineType=cv2.LINE_8)
    return img


def test_connected_components_cannot_find_a_glyph_joined_to_a_thick_isobar():
    """太い等圧線は細らせても残り、記号と繋がったままになる。

    実測でこれが起きていた。--erode 1 でも H と L は候補に出ず、山として
    出たのは数字と等圧線の断片と×印だけだった。
    """
    merged = outlined_glyphs(with_thick_isobar=True)
    assert glyph_candidates(merged, max_side=150, erode=1) == []
    # 横切っていなければ普通に取れる。取れない原因が合流だと分かる
    assert len(glyph_candidates(outlined_glyphs(False), max_side=150, erode=1)) == 3


def test_template_matching_finds_glyphs_that_are_joined_to_an_isobar():
    """テンプレートさえあれば当たる。match は連結成分を使わないため。

    これが Phase 1 の道になる。候補拾いで取れない記号でも、1つ切り出して
    テンプレートにすれば、残りは全部見つかる。
    """
    clean = outlined_glyphs(with_thick_isobar=False)
    c = glyph_candidates(clean, max_side=150)[0]
    template = crop_template(clean, (c.x0, c.y0, c.x1, c.y1))

    merged = outlined_glyphs(with_thick_isobar=True)
    for threshold in (0.6, 0.8):
        assert len(match_templates(merged, {"L": template}, threshold=threshold)) == 3


def test_cut_by_box_works_without_any_candidate(tmp_path):
    """候補にならない場所からでもテンプレートを切り出せること。"""
    import argparse

    from scripts.extract_symbols import cmd_cut

    chart = tmp_path / "Js_2024070100.png"
    Image.fromarray(outlined_glyphs(with_thick_isobar=True)).save(chart)
    assert glyph_candidates(np.array(Image.open(chart)), max_side=150, erode=1) == []

    cmd_cut(argparse.Namespace(
        image=chart, index=-1, box=[75, 145, 145, 255], name="L", pad=0,
        fit=False, clean=False, out=tmp_path, band="isobar", erode=0, max_side=64,
    ))
    saved = np.array(Image.open(tmp_path / "L.png").convert("L")) > 127
    assert saved.any() and saved.shape == (110, 70)


# --- 傾きの違う記号 -----------------------------------------------------

def tilted_glyphs(angles=(0, 20, -25)):
    """同じ記号を、傾きを変えて並べた図。天気図の記号は傾きが揃っていない。"""
    img = np.full((500, 900, 3), 255, dtype=np.uint8)
    for x, angle in zip((100, 400, 700), angles):
        tile = np.full((200, 200, 3), 255, dtype=np.uint8)
        cv2.rectangle(tile, (60, 40), (140, 160), BLACK, 7)
        matrix = cv2.getRotationMatrix2D((100, 100), angle, 1.0)
        img[150:350, x:x + 200] = cv2.warpAffine(
            tile, matrix, (200, 200), borderValue=(255, 255, 255))
    return img


def one_template_from(img):
    c = glyph_candidates(img, max_side=200)[0]
    return crop_template(img, (c.x0, c.y0, c.x1, c.y1))


def test_matching_without_rotation_only_finds_the_one_it_was_cut_from():
    """実測でこれが起きた。切り出した記号だけがスコア1.00で当たり、他は
    しきい値を0.5まで下げても当たらなかった。"""
    img = tilted_glyphs()
    hits = match_templates(img, {"L": one_template_from(img)}, threshold=0.5)
    assert len(hits) == 1
    assert hits[0].score == pytest.approx(1.0)


def test_rotating_the_template_finds_the_tilted_ones():
    img = tilted_glyphs()
    hits = match_templates(img, {"L": one_template_from(img)}, threshold=0.6,
                           angles=range(-50, 55, 5))
    assert len(hits) == 3
    assert min(h.score for h in hits) > 0.9


def test_a_ten_degree_step_is_too_coarse():
    """5度ずれると一致スコアが0.72まで落ちる。刻みは5度が目安。"""
    img = tilted_glyphs()
    coarse = match_templates(img, {"L": one_template_from(img)}, threshold=0.6,
                             angles=range(-50, 60, 10))
    fine = match_templates(img, {"L": one_template_from(img)}, threshold=0.6,
                           angles=range(-50, 55, 5))
    assert len(coarse) == len(fine) == 3
    assert min(h.score for h in coarse) < min(h.score for h in fine)


def test_rotate_template_keeps_every_pixel():
    """回しても角が切れないこと(画布を広げている)。"""
    from src.chartsymbols import rotate_template
    template = np.zeros((20, 30), dtype=bool)
    template[5:15, 5:25] = True
    turned = rotate_template(template, 45)
    assert turned.shape[0] > 20 and turned.shape[1] > 30
    assert turned.sum() == pytest.approx(template.sum(), rel=0.2)


def test_extra_templates_of_the_same_symbol_are_counted_together():
    """1個体から作ったテンプレートで外れるとき、別個体を足して両方当てる。

    H2.png や L_b.png のように増やしても、数えるときは H と L にまとまる。
    """
    from scripts.extract_symbols import symbol_of
    assert symbol_of("H") == "H"
    assert symbol_of("H2") == "H"
    assert symbol_of("L_b") == "L"
    assert symbol_of("TD") == "TD"      # 数字も_も無い名前はそのまま
    assert symbol_of("TD2") == "TD"


def test_angle_report_flags_a_real_pileup_at_the_edge(capsys):
    """まとまった数が端に張り付いていれば、外にまだ記号がある。"""
    from scripts.extract_symbols import report_angles
    report_angles([-50.0] * 5 + [0.0, 10.0], 50.0)
    out = capsys.readouterr().out
    assert "★" in out and "広げる" in out

    report_angles([0.0, 10.0, -20.0], 50.0)
    assert "範囲の端" not in capsys.readouterr().out


# --- 固定の誤検出 -------------------------------------------------------

def test_fixed_detections_are_flagged(capsys):
    """毎回同じ画素・同じ傾きで出る検出は気象ではない。

    実測で H+20度 が20枚中9枚に出て、しかも毎回リストの最後(画像の下端)に
    あった。高気圧は日ごとに動くので、画素まで一致し続けるのは不自然。
    """
    from scripts.extract_symbols import report_fixed_detections
    placed = []
    for i in range(10):
        placed.append(("H", 0.30 + 0.01 * i, 0.20 + 0.005 * i, 0.0))   # 動く
        placed.append(("H", 0.812, 0.905, 20.0))                        # 動かない
    report_fixed_detections(placed, 10)
    out = capsys.readouterr().out
    assert "0.812" in out and "+20度" in out and "10枚" in out
    assert "0.300" not in out          # 動くほうは指摘しない


def test_moving_detections_are_not_flagged(capsys):
    from scripts.extract_symbols import report_fixed_detections
    placed = [("L", 0.2 + 0.05 * i, 0.3 + 0.02 * i, 0.0) for i in range(10)]
    report_fixed_detections(placed, 10)
    assert "画素まで一致し続ける検出はない" in capsys.readouterr().out


def test_same_place_but_different_tilt_is_not_fixed(capsys):
    """同じ場所でも傾きが変われば、描き直された記号とみなす。"""
    from scripts.extract_symbols import report_fixed_detections
    placed = [("H", 0.5, 0.5, float(5 * i)) for i in range(10)]
    report_fixed_detections(placed, 10)
    assert "画素まで一致し続ける検出はない" in capsys.readouterr().out


def test_score_report_flags_a_pileup_at_the_threshold(capsys):
    """しきい値ぎわに溜まっているときだけ指摘する。

    最低スコアを見てはいけない。検出が増えれば必ずしきい値に張り付くので、
    常に警告が出てしまっていた。1個の外れ値ではなく分布を見る。
    """
    from scripts.extract_symbols import report_scores
    report_scores([0.65, 0.66, 0.67, 0.66, 0.65, 0.90, 0.95], 0.65)
    assert "★" in capsys.readouterr().out

    # 1個だけしきい値ちょうどでも、全体が上にあれば指摘しない
    report_scores([0.65] + [0.80 + 0.01 * i for i in range(40)], 0.65)
    assert "★" not in capsys.readouterr().out


def test_angle_report_ignores_a_lone_detection_at_the_edge(capsys):
    """端の1個で範囲を広げる理由にはならない。検出が増えればほぼ必ず出る。"""
    from scripts.extract_symbols import report_angles
    report_angles([60.0] + [float(a) for a in range(-50, 55, 5)], 60)
    out = capsys.readouterr().out
    assert "★" not in out
    assert "広げても増えない" in out

    report_angles([60.0] * 8 + [0.0, 10.0, -5.0], 60)
    assert "★" in capsys.readouterr().out


# --- テンプレートの見切れ -----------------------------------------------

def thick_glyph_with_a_thin_isobar():
    img = np.full((400, 400, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (150, 150), (250, 280), BLACK, 7)          # 太い記号
    cv2.line(img, (0, 215), (399, 215), BLACK, 1, lineType=cv2.LINE_8)  # 細い等圧線
    return img


def test_clipped_template_is_detected():
    """枠が記号を切っていれば、細らせても縁に太い線が残る。

    実測で、切り出したテンプレートが見切れていた。部分だけのテンプレートは
    完全な記号にうまく当たらず、これが取りこぼしの主因だった。
    """
    from src.chartsymbols import touches_border
    img = thick_glyph_with_a_thin_isobar()
    assert touches_border(crop_template(img, (160, 160, 300, 330))) > 0.01
    assert touches_border(crop_template(img, (100, 100, 240, 270))) > 0.01


def test_a_template_that_fits_does_not_look_clipped():
    from src.chartsymbols import touches_border
    img = thick_glyph_with_a_thin_isobar()
    assert touches_border(crop_template(img, (140, 140, 262, 292))) == 0.0


def test_a_crossing_thin_isobar_is_not_mistaken_for_clipping():
    """等圧線は細いので、細らせれば縁から消える。余白を取っても誤判定しない。"""
    from src.chartsymbols import touches_border
    img = thick_glyph_with_a_thin_isobar()
    assert touches_border(crop_template(img, (110, 110, 292, 322))) == 0.0


# --- 枠を記号に合わせる -------------------------------------------------

def test_fitting_converges_from_a_too_small_and_a_too_large_box():
    """狭すぎる枠も広すぎる枠も、同じ「記号がちょうど収まる枠」に落ち着く。

    枠が狭ければ記号が見切れ、広ければ周りの等圧線がテンプレートの大半を
    占める。どちらも一致スコアを下げる。実測では余白35画素を足した
    テンプレートのほうが当たりが悪かった(5.3個/枚 -> 3.4個/枚)。
    """
    from src.chartsymbols import DEFAULT_BANDS, fit_glyph_box, to_hsv
    img = thick_glyph_with_a_thin_isobar()
    mask = DEFAULT_BANDS["isobar"].mask(to_hsv(img))
    tight = fit_glyph_box(mask, (170, 170, 230, 260))
    loose = fit_glyph_box(mask, (100, 100, 300, 330))
    assert tight == loose
    # 記号の真の範囲(146,146)-(254,284) を、わずかな余白付きで囲むこと
    assert tight[0] == pytest.approx(146, abs=8)
    assert tight[3] == pytest.approx(284, abs=8)


def test_a_fitted_box_does_not_look_clipped():
    """合わせた枠には余白が残るので、見切れの判定が誤反応しない。"""
    from src.chartsymbols import DEFAULT_BANDS, fit_glyph_box, to_hsv, touches_border
    img = thick_glyph_with_a_thin_isobar()
    mask = DEFAULT_BANDS["isobar"].mask(to_hsv(img))
    fitted = fit_glyph_box(mask, (170, 170, 230, 260))
    assert touches_border(crop_template(img, fitted)) == 0.0


def test_fitting_falls_back_when_the_box_holds_no_thick_ink():
    """太い塊が無ければ、指定された枠をそのまま返す。"""
    from src.chartsymbols import DEFAULT_BANDS, fit_glyph_box, to_hsv
    blank_img = blank()
    mask = DEFAULT_BANDS["isobar"].mask(to_hsv(blank_img))
    assert fit_glyph_box(mask, (10, 10, 60, 60)) == (10, 10, 60, 60)


def test_uneven_template_sizes_are_flagged(capsys):
    """同じ記号なのに大きさが揃わなければ、切り出しが失敗している。

    実測で L のテンプレートが4枚中1枚だけ84x131(他は160x192前後)になり、
    L の検出数が61個から82個へ不自然に増えた。半端なテンプレートは本物に
    当たらないうえ、記号の一部に似た形に当たって誤検出を増やす。
    """
    from scripts.extract_symbols import warn_uneven_sizes
    tile = lambda w, h: np.ones((h, w), dtype=bool)
    warn_uneven_sizes({"L": tile(84, 131), "L2": tile(160, 192),
                       "L3": tile(157, 210), "L4": tile(160, 210)})
    out = capsys.readouterr().out
    assert "★L" in out and "84x131" in out


def test_similar_template_sizes_are_not_flagged(capsys):
    from scripts.extract_symbols import warn_uneven_sizes
    tile = lambda w, h: np.ones((h, w), dtype=bool)
    warn_uneven_sizes({"L": tile(158, 195), "L2": tile(160, 192),
                       "L3": tile(157, 210)})
    assert capsys.readouterr().out == ""


def test_a_lone_template_is_never_flagged(capsys):
    from scripts.extract_symbols import warn_uneven_sizes
    warn_uneven_sizes({"H": np.ones((100, 90), dtype=bool)})
    assert capsys.readouterr().out == ""


# --- テンプレートから等圧線を消す ---------------------------------------

def glyphs_with_their_own_isobars():
    """同じ記号3個に、場所ごとに違う等圧線が掛かる図。実物に近い形。"""
    rng = np.random.default_rng(3)
    img = np.full((500, 900, 3), 255, dtype=np.uint8)
    spots = [(80, 150), (340, 150), (620, 150)]
    for x, y in spots:
        cv2.rectangle(img, (x, y), (x + 90, y + 140), BLACK, 7)
    for x, y in spots:
        for _ in range(3):
            slope = int(rng.integers(-60, 60))
            off = int(rng.integers(-40, 180))
            cv2.line(img, (x - 60, y + off), (x + 160, y + off + slope),
                     BLACK, 2, lineType=cv2.LINE_8)
    return img


def test_removing_isobars_helps_matching_the_other_instances():
    """テンプレートの等圧線を消すと、他の個体への当たりが上がる。

    枠の大きさをいくら調整しても、記号に掛かる等圧線はテンプレートに入る。
    等圧線の模様は場所ごとに違うので、そのぶんスコアが下がる。実測でも枠を
    広げるほど当たりが悪くなった(5.3個 -> 3.4個 -> 2.6個/枚)。
    """
    from src.chartsymbols import DEFAULT_BANDS, glyph_only_template, to_hsv
    img = glyphs_with_their_own_isobars()
    mask = DEFAULT_BANDS["isobar"].mask(to_hsv(img))
    plain = crop_template(img, (70, 140, 180, 300))
    cleaned, _, isolated = glyph_only_template(mask, (70, 140, 180, 300))
    assert isolated

    plain_hits = match_templates(img, {"L": plain}, threshold=0.5, angles=(0,))
    clean_hits = match_templates(img, {"L": cleaned}, threshold=0.5, angles=(0,))
    assert len(plain_hits) == len(clean_hits) == 3
    # 自分自身は1.00から落ちるが、他の個体への当たりは上がる
    assert min(h.score for h in clean_hits) > min(h.score for h in plain_hits)


def test_cleaned_template_holds_fewer_pixels_than_the_raw_crop():
    from src.chartsymbols import DEFAULT_BANDS, glyph_only_template, to_hsv
    img = glyphs_with_their_own_isobars()
    mask = DEFAULT_BANDS["isobar"].mask(to_hsv(img))
    plain = crop_template(img, (70, 140, 180, 300))
    cleaned, _, isolated = glyph_only_template(mask, (70, 140, 180, 300))
    assert isolated
    assert 0 < cleaned.sum() < plain.sum()


def test_cleaning_falls_back_when_there_is_no_thick_ink():
    from src.chartsymbols import DEFAULT_BANDS, glyph_only_template, to_hsv
    mask = DEFAULT_BANDS["isobar"].mask(to_hsv(blank()))
    template, box, isolated = glyph_only_template(mask, (10, 10, 60, 60))
    assert box == (10, 10, 60, 60) and not template.any()
    assert not isolated


def test_isolation_fails_loudly_when_a_bold_isobar_crosses_the_glyph():
    """太い等圧線が記号を横切ると、細らせても1つの塊のままで分けられない。

    黙って別の枠を返すと、見切れの判定が誤って反応して原因を見誤る。実測で
    それが起き、記号は見切れていないのに「見切れている」と報告していた。
    """
    from src.chartsymbols import DEFAULT_BANDS, glyph_only_template, to_hsv

    def scene(bold_crosses):
        img = np.full((600, 900, 3), 255, dtype=np.uint8)
        cv2.rectangle(img, (300, 200), (390, 340), BLACK, 7)
        if bold_crosses:
            cv2.line(img, (0, 260), (899, 300), BLACK, 5, lineType=cv2.LINE_8)
        return img

    box = (296, 196, 394, 344)
    mask_clear = DEFAULT_BANDS["isobar"].mask(to_hsv(scene(False)))
    _, _, ok_clear = glyph_only_template(mask_clear, box)
    assert ok_clear

    mask_crossed = DEFAULT_BANDS["isobar"].mask(to_hsv(scene(True)))
    template, fallback_box, ok_crossed = glyph_only_template(mask_crossed, box)
    assert not ok_crossed
    assert fallback_box == box          # 指定された枠をそのまま返す
    assert template.any()


def test_a_long_bold_isobar_is_not_taken_as_the_glyph():
    """図を横切る太い等圧線は枠より遥かに長いので、記号の塊として採らない。"""
    from src.chartsymbols import DEFAULT_BANDS, glyph_only_template, to_hsv
    img = np.full((600, 900, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (300, 200), (390, 340), BLACK, 7)
    cv2.line(img, (0, 500), (899, 520), BLACK, 5, lineType=cv2.LINE_8)  # 離れた所
    mask = DEFAULT_BANDS["isobar"].mask(to_hsv(img))
    template, box, ok = glyph_only_template(mask, (296, 196, 394, 344))
    assert ok
    assert box[3] < 400                 # 下の等圧線まで枠が伸びていない


def test_border_contact_is_not_called_clipping_when_isolation_failed(capsys):
    """等圧線を切り離せなかったなら、縁に線が掛かるのは当たり前。

    実測で、記号は見切れていないのに「見切れている」と報告していた。
    縁に掛かっていたのは記号を横切る等圧線だった。
    """
    import argparse

    from scripts.extract_symbols import cmd_cut

    img = np.full((600, 900, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (300, 200), (390, 340), BLACK, 7)
    cv2.line(img, (0, 260), (899, 300), BLACK, 5, lineType=cv2.LINE_8)
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        chart = Path(tmp) / "Js_2024070100.png"
        Image.fromarray(img).save(chart)
        cmd_cut(argparse.Namespace(
            image=chart, index=-1, box=[296, 196, 394, 344], name="L", pad=40,
            fit=True, clean=True, out=tmp, band="isobar", erode=0, max_side=64,
        ))
    out = capsys.readouterr().out
    assert "切り離せなかった" in out
    assert "見切れではない" in out
    assert "★" not in out


# --- 人が作ったテンプレート ---------------------------------------------

def test_a_hand_made_template_works_in_either_polarity():
    """手で作ったテンプレートは白地に黒になりやすい。どちらでも受け付ける。

    太い等圧線が記号を横切ると連結成分では分けられないが、人なら消せるし、
    重なっていない個体を選べる。手作りのほうが確実な場合がある。
    """
    from scripts.extract_symbols import as_template
    glyph = np.zeros((40, 30), dtype=np.uint8)
    glyph[5:35, 5:12] = 255
    glyph[28:35, 5:25] = 255

    white_on_black = as_template(glyph)
    black_on_white = as_template(255 - glyph)
    assert white_on_black.mean() == pytest.approx(black_on_white.mean())
    assert (white_on_black == black_on_white).all()
    assert 0.1 < white_on_black.mean() < 0.5     # 記号は画像の一部でしかない


def test_hand_made_template_with_an_odd_ink_share_is_flagged(capsys):
    """記号の割合が極端なら、切り出しの範囲か白黒の閾値を疑う。"""
    from scripts.extract_symbols import as_template
    as_template(np.zeros((40, 30), dtype=np.uint8), "H4.png")     # 真っ黒
    assert "★" in capsys.readouterr().out

    glyph = np.zeros((40, 30), dtype=np.uint8)
    glyph[5:35, 5:12] = 255
    glyph[28:35, 5:25] = 255
    as_template(glyph, "H.png")
    assert capsys.readouterr().out == ""


def test_templates_can_be_other_image_formats(tmp_path):
    """画像編集ソフトからだと png 以外で保存されることがある。"""
    from scripts.extract_symbols import load_templates
    glyph = np.zeros((40, 30), dtype=np.uint8)
    glyph[5:35, 5:12] = 255
    Image.fromarray(glyph).save(tmp_path / "H.bmp")
    Image.fromarray(glyph).convert("RGB").save(tmp_path / "L.jpg")
    assert sorted(load_templates(tmp_path)) == ["H", "L"]
