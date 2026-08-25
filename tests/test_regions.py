"""ラベルごとの「見るべき領域」と、Grad-CAMとの突き合わせのテスト。"""

import numpy as np
import pytest

from src.labels import LABELS
from src.regions import (
    CAM_GRID,
    Region,
    attention_lift,
    attention_mass,
    draw_region,
    load_regions,
    peak_in_region,
    peak_position,
    resize_cam,
)

FULL = Region("typhoon", 0.0, 0.0, 1.0, 1.0)
QUARTER = Region("okhotsk_high", 0.5, 0.0, 1.0, 0.5)   # 右上の1/4


def test_area():
    assert FULL.area == pytest.approx(1.0)
    assert QUARTER.area == pytest.approx(0.25)


def test_rejects_unknown_label():
    with pytest.raises(ValueError, match="未知のラベル"):
        Region("no_such_label", 0.0, 0.0, 1.0, 1.0)


@pytest.mark.parametrize("box", [
    (0.5, 0.0, 0.5, 1.0),    # 幅0
    (0.5, 0.5, 0.2, 1.0),    # x1 < x0
    (0.0, 0.0, 1.2, 1.0),    # 範囲外
    (-0.1, 0.0, 1.0, 1.0),   # 範囲外
])
def test_rejects_broken_box(box):
    with pytest.raises(ValueError):
        Region("typhoon", *box)


def test_pixel_box():
    assert QUARTER.pixel_box(200, 100) == (100, 0, 200, 50)


def test_mask_sums_to_area():
    mask = QUARTER.mask(64, 64)
    assert mask.sum() / mask.size == pytest.approx(QUARTER.area)


def test_mask_splits_boundary_pixels():
    """境界が画素の途中にあっても、面積の割合で按分される。"""
    half_of_one_pixel = Region("typhoon", 0.0, 0.0, 0.5, 1.0)
    mask = half_of_one_pixel.mask(1, 1)
    assert mask[0, 0] == pytest.approx(0.5)


def test_uniform_attention_gives_mass_equal_to_area():
    """一様に注目しているCAMでは mass は面積比に一致し、lift は1になる。"""
    cam = np.ones((7, 7), dtype=np.float32)
    assert attention_mass(cam, QUARTER) == pytest.approx(QUARTER.area, abs=0.02)
    assert attention_lift(cam, QUARTER) == pytest.approx(1.0, abs=0.1)


def test_attention_inside_region_gives_high_mass():
    cam = np.zeros((CAM_GRID, CAM_GRID), dtype=np.float32)
    cam[10:40, CAM_GRID - 40:CAM_GRID - 10] = 1.0    # 右上だけ光らせる
    assert attention_mass(cam, QUARTER) == pytest.approx(1.0)
    assert attention_lift(cam, QUARTER) == pytest.approx(4.0)


def test_attention_outside_region_gives_zero_mass():
    cam = np.zeros((CAM_GRID, CAM_GRID), dtype=np.float32)
    cam[CAM_GRID - 40:CAM_GRID - 10, 10:40] = 1.0    # 左下だけ光らせる
    assert attention_mass(cam, QUARTER) == pytest.approx(0.0)


def test_all_zero_cam_is_zero_not_nan():
    cam = np.zeros((7, 7), dtype=np.float32)
    assert attention_mass(cam, QUARTER) == 0.0


def test_negative_values_are_clipped():
    """Grad-CAMはReLU済みだが、負の値が来ても割合が1を超えたりしない。"""
    cam = -np.ones((8, 8), dtype=np.float32)
    cam[0, -1] = 1.0
    assert 0.0 <= attention_mass(cam, QUARTER) <= 1.0


def test_resize_keeps_peak_value():
    cam = np.zeros((7, 7), dtype=np.float32)
    cam[0, 6] = 0.8
    grid = resize_cam(cam, 224)
    assert grid.shape == (224, 224)
    assert grid.max() == pytest.approx(0.8, abs=0.01)


def test_peak_position_and_pointing():
    cam = np.zeros((10, 10), dtype=np.float32)
    cam[1, 8] = 1.0    # 右上
    x, y = peak_position(cam)
    assert (x, y) == pytest.approx((0.85, 0.15))
    assert peak_in_region(cam, QUARTER)

    cam = np.zeros((10, 10), dtype=np.float32)
    cam[8, 1] = 1.0    # 左下
    assert not peak_in_region(cam, QUARTER)


def test_mass_does_not_depend_on_cam_resolution():
    """粗いCAMでも細かいCAMでも、同じ注目のしかたなら同じ数字になる。"""
    coarse = np.zeros((7, 7), dtype=np.float32)
    coarse[0, 4:] = 1.0
    fine = np.kron(coarse, np.ones((8, 8), dtype=np.float32))
    assert attention_mass(coarse, QUARTER) == pytest.approx(
        attention_mass(fine, QUARTER), abs=0.05
    )


def test_draw_region_does_not_modify_original():
    from PIL import Image

    image = Image.new("RGB", (40, 30), (255, 255, 255))
    drawn = draw_region(image, QUARTER, text="x")
    assert np.array(image).min() == 255            # 元は白のまま
    assert np.array(drawn).min() < 255             # 描いた方には線がある


def test_shipped_regions_file_covers_every_label():
    """同梱の data/regions.csv が全ラベルぶん揃っていて、矩形として妥当なこと。"""
    regions = load_regions()
    assert sorted(regions) == sorted(LABELS)
    for label, region in regions.items():
        assert 0.0 < region.area < 1.0, label
        assert region.note, f"{label} に note が無い"


def test_load_rejects_duplicate_label(tmp_path):
    path = tmp_path / "regions.csv"
    path.write_text(
        "label,x0,y0,x1,y1,note\n"
        "typhoon,0.1,0.1,0.2,0.2,a\n"
        "typhoon,0.3,0.3,0.4,0.4,b\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="2回定義"):
        load_regions(path)


def test_load_rejects_missing_column(tmp_path):
    path = tmp_path / "regions.csv"
    path.write_text("label,x0,y0,x1\ntyphoon,0.1,0.1,0.2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="列がありません"):
        load_regions(path)


def test_load_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_regions(tmp_path / "no_such.csv")
