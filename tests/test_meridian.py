"""記号の傾きを経線から予測する仕組みの試験。

要点は3つ。

1. テンプレート自身の傾きを測る道具が正しいこと(相互照合と突き合わせる)
2. (位置, 傾き)から経線の集まる点を当てられること。外れ値が混じっても
3. **既定では検出が1画素も変わらないこと。**記録済みの実行結果が再現
   できなくなるのが、この変更でいちばん怖い
"""

import pathlib
import sys

import cv2
import numpy as np
import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.annotate_charts import resolve_meridian
from scripts.build_features import ink_image, load_templates_scaled, shrink
from src.chartsymbols import match_templates, rotate_template
from src.meridian import (COARSE_THRESHOLD, Meridian, fit, refine_hits,
                          template_tilt, template_tilts)

TEMPLATES = _ROOT / "data" / "templates"
SIZE = (1453, 1500)          # 前処理が揃える大きさ
ANGLES = tuple(np.arange(-60, 61, 5.0))


@pytest.fixture(scope="module")
def raw_templates():
    return load_templates_scaled(TEMPLATES, 1.0, quiet=True)


@pytest.fixture(scope="module")
def scaled_templates():
    return load_templates_scaled(TEMPLATES, 0.7, quiet=True)


@pytest.fixture(scope="module")
def tilts(raw_templates):
    return template_tilts(raw_templates)


def place(templates, spots, meridian, tilts, lines=0, line_width=5):
    """指定した場所に、そこの経線の傾きで記号を置いた天気図もどきを作る。"""
    width, height = SIZE
    rgb = np.full((height, width, 3), 255, np.uint8)
    for cx, cy, name in spots:
        want = float(meridian.tilt_at(cx, cy))
        mask = rotate_template(templates[name], want - tilts[name])
        h, w = mask.shape
        y0, x0 = int(cy * height - h / 2), int(cx * width - w / 2)
        ys, xs = np.nonzero(mask)
        rgb[ys + y0, xs + x0] = 0
        for i in range(lines):
            yy = y0 + int(h * (i + 1) / (lines + 1))
            pts = np.array([[x, yy + int(18 * np.sin(x / 140.0))]
                            for x in range(0, width, 4)])
            cv2.polylines(rgb, [pts], False, (0, 0, 0), line_width)
    return rgb


def test_template_tilt_recovers_a_known_rotation(raw_templates):
    """回した記号の傾きは、回した角度ぶん動く。"""
    base = raw_templates["L"]
    start = template_tilt(base)
    for turn in (-20.0, -7.5, 5.0, 18.0):
        got = template_tilt(rotate_template(base, turn))
        assert abs((got - start) - turn) <= 2.0, f"{turn}度 回したのに {got - start}度"


def test_template_tilts_agree_with_cross_matching(raw_templates, scaled_templates,
                                                  tilts):
    """記号 X をテンプレート Y で当てると tilt(X) - tilt(Y) で当たる。

    傾きの測りが本物であることの裏付け。ここが崩れたら、測り方が壊れている。
    """
    width, height = SIZE
    checked = 0
    for name, mask in raw_templates.items():
        rgb = np.full((height, width, 3), 255, np.uint8)
        h, w = mask.shape
        y0, x0 = int(0.5 * height - h / 2), int(0.5 * width - w / 2)
        ys, xs = np.nonzero(mask)
        rgb[ys + y0, xs + x0] = 0
        others = {k: v for k, v in scaled_templates.items() if k != name}
        hits = match_templates(shrink(ink_image(rgb), 0.7), others,
                               threshold=0.30, angles=ANGLES)
        if not hits:
            continue
        best = max(hits, key=lambda c: c.score)
        predicted = tilts[name] - tilts[best.label]
        assert abs(best.angle - predicted) <= 5.0, (
            f"{name} を {best.label} で当てたら {best.angle:+.0f}度、"
            f"傾きの差からは {predicted:+.0f}度")
        checked += 1
    assert checked >= 8, f"照合できた組が少なすぎます: {checked}組"


def test_fit_recovers_the_convergence_point():
    """外れ値が1割混じっても、経線の集まる点を当てられる。"""
    rng = np.random.default_rng(0)
    truth = Meridian(x=0.52, y=-0.80, width=SIZE[0], height=SIZE[1])
    xs = rng.uniform(0.05, 0.95, 300)
    ys = rng.uniform(0.05, 0.95, 300)
    tilt = truth.tilt_at(xs, ys) + rng.normal(0, 1.5, 300)
    tilt[rng.choice(300, 30, replace=False)] = rng.uniform(-60, 60, 30)

    got = fit(xs, ys, tilt, width=SIZE[0], height=SIZE[1])
    assert got.residual <= 2.0, f"残差 {got.residual:.2f}度"
    grid = np.linspace(0.05, 0.95, 5)
    gx, gy = np.meshgrid(grid, grid)
    error = np.abs(got.tilt_at(gx, gy) - truth.tilt_at(gx, gy))
    assert error.max() <= 2.0, f"画面全体での予測のずれ 最大 {error.max():.2f}度"


def test_fit_needs_enough_samples():
    with pytest.raises(ValueError):
        fit([0.1, 0.2], [0.3, 0.4], [1.0, 2.0])


def test_refine_finds_symbols_the_coarse_search_misses(raw_templates,
                                                       scaled_templates, tilts):
    """5度刻みでは落ちる記号を、予測した角度で測り直すと拾える。"""
    meridian = Meridian(x=0.50, y=-0.60, width=SIZE[0], height=SIZE[1])
    spots = [(0.22, 0.35, "L"), (0.50, 0.55, "L3"),
             (0.78, 0.40, "H2"), (0.65, 0.75, "H4")]
    rgb = place(raw_templates, spots, meridian, tilts, lines=2)
    image = shrink(ink_image(rgb), 0.7)

    plain = match_templates(image, scaled_templates, 0.65, angles=ANGLES)
    coarse = match_templates(image, scaled_templates, COARSE_THRESHOLD, angles=ANGLES)
    refined = refine_hits(image, coarse, scaled_templates, tilts, meridian,
                          threshold=0.50)

    assert len(refined) > len(plain), (
        f"測り直しで増えていません: そのまま {len(plain)}個 -> {len(refined)}個")
    # 置いた4個の近くに、それぞれ検出がある
    for cx, cy, _name in spots:
        near = [h for h in refined
                if (h.cx - cx) ** 2 + (h.cy - cy) ** 2 < 0.02 ** 2]
        assert near, f"({cx},{cy}) の記号が見つかりません"


def test_refine_drops_hits_whose_tilt_contradicts_the_meridian(raw_templates,
                                                               scaled_templates,
                                                               tilts):
    """その場所の経線とかけ離れた傾きで当たったものは捨てる。"""
    meridian = Meridian(x=0.50, y=-0.60, width=SIZE[0], height=SIZE[1])
    spots = [(0.22, 0.35, "L"), (0.50, 0.55, "L3"),
             (0.78, 0.40, "H2"), (0.65, 0.75, "H4")]
    rgb = place(raw_templates, spots, meridian, tilts, lines=2)
    image = shrink(ink_image(rgb), 0.7)

    coarse = match_templates(image, scaled_templates, 0.35, angles=ANGLES)
    refined = refine_hits(image, coarse, scaled_templates, tilts, meridian,
                          threshold=0.50)
    assert len(coarse) > len(refined), "捨てられた候補が1つもありません"
    for hit in refined:
        deviation = abs(float(meridian.deviation(
            hit.cx, hit.cy, tilts[hit.label] + hit.angle)))
        assert deviation <= 12.0, f"{deviation:.1f}度 ずれた検出が残っています"


def test_saved_fit_round_trips(tmp_path):
    path = tmp_path / "meridian.json"
    made = Meridian(x=0.51, y=-0.72, sign=-1.0, width=100, height=200,
                    residual=1.23, samples=456, note="試験")
    made.save(path)
    assert Meridian.load(path) == made


def test_meridian_is_off_by_default():
    """既定では経線を使わない。渡さなければ検出は1画素も変わらない。"""
    assert resolve_meridian(None, TEMPLATES) == (None, None)
    assert resolve_meridian(False, TEMPLATES) == (None, None)


def test_auto_is_silent_when_the_fit_is_missing(tmp_path, monkeypatch):
    """'auto' は当てはめが無ければ黙って使わない。例外にしない。"""
    import src.meridian as meridian_module

    monkeypatch.setattr(meridian_module, "DEFAULT_MERIDIAN_PATH",
                        tmp_path / "no_such.json")
    assert resolve_meridian("auto", TEMPLATES) == (None, None)


def test_auto_loads_the_fit_when_present(tmp_path, monkeypatch):
    import src.meridian as meridian_module

    path = tmp_path / "meridian.json"
    Meridian(x=0.5, y=-0.6, width=SIZE[0], height=SIZE[1]).save(path)
    monkeypatch.setattr(meridian_module, "DEFAULT_MERIDIAN_PATH", path)
    found, got_tilts = resolve_meridian("auto", TEMPLATES)
    assert isinstance(found, Meridian)
    assert set(got_tilts) == set(load_templates_scaled(TEMPLATES, 1.0, quiet=True))


def test_angles_for_centres_on_the_prediction(tilts):
    meridian = Meridian(x=0.50, y=-0.60, width=SIZE[0], height=SIZE[1])
    want = float(meridian.tilt_at(0.8, 0.4))
    angles = meridian.angles_for(0.8, 0.4, tilts["L"], span=3.0, step=0.5)
    assert len(angles) == 13
    assert angles[6] == pytest.approx(want - tilts["L"])
    assert min(angles) == pytest.approx(want - tilts["L"] - 3.0)
