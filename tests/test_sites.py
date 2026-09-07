"""地点とその周辺だけを見る仕組みを確かめる。

**この機能は学習し直しを伴わない。**検出結果を距離で絞り込むだけなので、
macro F1 や1位正解率は変わらない。ここで守りたいのは、絞り込みの判定が
相対座標のまま正しく行われること、そして位置がずれていることに
利用者が気づける作りになっていること。
"""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.sites import DEFAULT_RADIUS, Site, get_site, load_sites, nearby, summarize


class FakeDetections:
    """ChartDetections の必要な部分だけを持つ入れ物。"""

    def __init__(self, highs=(), lows=(), edge_highs=(), edge_lows=()):
        self.highs = list(highs)
        self.lows = list(lows)
        self.edge_highs = list(edge_highs)
        self.edge_lows = list(edge_lows)


def test_a_site_is_not_tied_to_the_label_names():
    """**Region と違い、地点の名前は利用者が決める。**
    src/labels.py の10ラベルに縛られてはいけない。"""
    site = Site(name="komatsu", x=0.5, y=0.57)
    assert site.radius == DEFAULT_RADIUS
    Site(name="misawa", x=0.6, y=0.4)   # 例外が出ないこと


@pytest.mark.parametrize("x,y", [(-0.1, 0.5), (0.5, 1.2)])
def test_a_site_outside_the_image_is_refused(x, y):
    """相対座標の外を許すと、黙って誰も居ない場所を測ることになる。"""
    with pytest.raises(ValueError):
        Site(name="どこか", x=x, y=y)


@pytest.mark.parametrize("radius", [0.0, -0.1, 1.5])
def test_an_impossible_radius_is_refused(radius):
    with pytest.raises(ValueError):
        Site(name="どこか", x=0.5, y=0.5, radius=radius)


def test_it_keeps_only_what_is_inside_the_circle():
    site = Site(name="s", x=0.50, y=0.50, radius=0.10)
    points = [(0.50, 0.50), (0.55, 0.50), (0.90, 0.90), (0.50, 0.65)]
    assert nearby(points, site) == [(0.50, 0.50), (0.55, 0.50)]


def test_it_is_a_circle_not_a_square():
    """矩形で切ると角が入る。『地点から○○以内』と言うなら距離で切る。"""
    site = Site(name="s", x=0.5, y=0.5, radius=0.1)
    # 矩形なら (0.58, 0.58) は縦横とも0.08しか離れていないので内側に入る。
    # 円では距離が 0.113 になり、半径0.1の外になる
    assert site.contains(0.5, 0.59)
    assert not site.contains(0.58, 0.58)


def test_the_nearest_system_is_reported_first():
    site = Site(name="s", x=0.5, y=0.5, radius=0.2)
    detections = FakeDetections(highs=[(0.60, 0.50), (0.52, 0.50)])
    found = summarize(detections, site)
    assert found["highs"][0] == (0.52, 0.50), "近い順に並んでいない"
    assert found["nearest"]["kind"] == "H"
    assert found["nearest"]["distance"] == pytest.approx(0.02)


def test_systems_centred_off_the_chart_are_not_counted():
    """**中心が枠外の系は位置を主張しない。**文字の位置しか分かっておらず、
    距離で切ると嘘になる(src/chartfeatures.py の注記)。"""
    site = Site(name="s", x=0.5, y=0.5, radius=0.5)
    detections = FakeDetections(highs=[(0.5, 0.5)],
                                edge_highs=[(0.5, 0.5)], edge_lows=[(0.5, 0.5)])
    found = summarize(detections, site)
    assert found["n_high"] == 1, "枠外の系まで数えている"
    assert found["n_low"] == 0


def test_it_also_reports_the_whole_chart_count():
    """周辺の個数だけでは『近くに無い』のか『そもそも検出できていない』のかが
    分からない。全体の個数と並べて初めて読める。"""
    site = Site(name="s", x=0.1, y=0.1, radius=0.05)
    detections = FakeDetections(highs=[(0.9, 0.9)], lows=[(0.8, 0.8), (0.7, 0.7)])
    found = summarize(detections, site)
    assert (found["n_high"], found["n_low"]) == (0, 0)
    assert (found["n_high_all"], found["n_low_all"]) == (1, 2)
    assert found["nearest"] is None


def test_the_bundled_sites_file_loads():
    sites = load_sites()
    assert "komatsu" in sites, "小松が定義されていない"
    assert get_site("komatsu").radius > 0


def test_an_unknown_site_lists_the_ones_that_exist(tmp_path):
    """名前を間違えたとき、何が使えるのかを教える。"""
    with pytest.raises(SystemExit) as caught:
        get_site("そんな地点は無い")
    assert "komatsu" in str(caught.value)


def test_the_bundled_position_records_how_it_was_decided():
    """**位置は緯度経度から変換してはいけない。**天気図は正距円筒図法ではなく、
    経緯線の目盛りから作った線形の式は図の中央でずれる(実際に東京が本州北部に、
    ソウルが満州に載った)。天気図に重ねて目で決めたことを控えに残しておく。
    これが残っていないと、後から「緯度経度から計算し直そう」と戻されかねない。
    """
    note = get_site("komatsu").note
    assert "目で決めた" in note, f"決め方が書かれていない: {note!r}"
    assert "変換していない" in note, f"緯度経度から変換していないと明記が無い: {note!r}"


def test_the_bundled_position_is_on_the_japan_sea_side_of_honshu():
    """小松は本州の日本海側。**利用者の目視で決めた値がうっかり書き換わって
    いないか**を、大まかな範囲で見張る。厳密な位置は目視でしか決められないので、
    ここでは明らかにおかしい値(海の外・大陸・太平洋の真ん中)だけを弾く。"""
    site = get_site("komatsu")
    assert 0.45 <= site.x <= 0.62, f"東西の位置が日本付近から外れている: x={site.x}"
    assert 0.45 <= site.y <= 0.65, f"南北の位置が日本付近から外れている: y={site.y}"


def test_the_drawing_does_not_change_the_original():
    from PIL import Image

    from src.sites import draw_site

    original = Image.new("RGB", (100, 120), "white")
    before = original.tobytes()
    drawn = draw_site(original, Site(name="s", x=0.5, y=0.5, radius=0.2))
    assert original.tobytes() == before, "元の画像を書き換えている"
    assert drawn.tobytes() != before, "何も描かれていない"


@pytest.mark.parametrize("script", ["site_preview", "site_report"])
def test_the_scripts_start(script):
    done = subprocess.run([sys.executable, "-m", f"scripts.{script}", "--help"],
                          cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert "--site" in done.stdout


def test_the_report_refuses_an_empty_image_folder(tmp_path):
    """『0日』と静かに出すより、作り方を出して止まるほうがよい。"""
    (tmp_path / "dates.csv").write_text("発生日\n2023-01-01\n", encoding="utf-8")
    empty = tmp_path / "からっぽ"
    empty.mkdir()
    done = subprocess.run(
        [sys.executable, "-m", "scripts.site_report", "--site", "komatsu",
         "--dates-csv", str(tmp_path / "dates.csv"), "--images-dir", str(empty)],
        cwd=ROOT, capture_output=True, text=True)
    assert done.returncode != 0
    assert "preprocess_jma" in done.stdout + done.stderr


# ---- Grad-CAM を円で測る ----

def test_the_circle_mask_has_the_area_of_a_circle():
    """**0/1で切ると格子の細かさで数値が動く。**境界の画素は入っている割合を
    持たせる(src/regions.py の Region.mask と同じ理由)。"""
    import numpy as np

    site = Site(name="s", x=0.5, y=0.5, radius=0.2)
    mask = site.mask(224, 224)
    assert mask.sum() / mask.size == pytest.approx(site.area, abs=1e-3)
    assert np.all((mask >= 0) & (mask <= 1))


def test_a_uniform_heatmap_has_lift_one():
    """一様に見ているなら「その円に集中している」とは言えない。
    lift の目盛りがここで決まる。"""
    import numpy as np

    from src.sites import attention_lift, attention_mass

    site = Site(name="s", x=0.5, y=0.5, radius=0.2)
    cam = np.ones((7, 7), dtype="float32")
    assert attention_mass(cam, site) == pytest.approx(site.area, abs=0.01)
    assert attention_lift(cam, site) == pytest.approx(1.0, abs=0.05)


def test_heat_on_the_site_counts_and_heat_far_away_does_not():
    import numpy as np

    from src.sites import attention_mass

    site = Site(name="s", x=0.5, y=0.5, radius=0.2)
    on_site = np.zeros((7, 7), dtype="float32")
    on_site[3, 3] = 1.0
    far = np.zeros((7, 7), dtype="float32")
    far[0, 0] = 1.0
    assert attention_mass(on_site, site) > 0.9
    assert attention_mass(far, site) < 0.05


def test_an_empty_heatmap_does_not_divide_by_zero():
    """熱が無いCAMでは割合が決まらない。落ちずに0を返す。"""
    import numpy as np

    from src.sites import attention_mass

    site = Site(name="s", x=0.5, y=0.5, radius=0.2)
    assert attention_mass(np.zeros((7, 7), dtype="float32"), site) == 0.0


def test_mass_and_lift_rank_labels_the_same_way():
    """**順位を付け替えるだけなら mass と lift は同じ結果になる。**
    同じ円なら area は一定なので、割っても順位は変わらない。
    片方だけ直して食い違わせないための備忘。"""
    import numpy as np

    from src.sites import attention_lift, attention_mass

    site = Site(name="s", x=0.4, y=0.6, radius=0.15)
    cams = []
    for cy, cx in ((3, 3), (0, 0), (6, 2), (2, 5)):
        cam = np.zeros((7, 7), dtype="float32")
        cam[cy, cx] = 1.0
        cams.append(cam)
    by_mass = sorted(range(len(cams)), key=lambda i: -attention_mass(cams[i], site))
    by_lift = sorted(range(len(cams)), key=lambda i: -attention_lift(cams[i], site))
    assert by_mass == by_lift
