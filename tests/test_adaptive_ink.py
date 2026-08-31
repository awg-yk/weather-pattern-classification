"""紙をスキャンした天気図で、線が拾えなくなる問題のテスト。

`DEFAULT_BANDS["isobar"]` は V<=90 という固定のしきい値で、気象庁PDF版の
真っ黒な等圧線に合わせてある。**紙のスキャンでは線が真っ黒にならない。**
実測(合成天気図)では、線の濃さが V=110 になった時点で等圧線マスクが
10.82% から 0.00% に落ち、検出も3個から0個になった。

利用者が最初に立てた仮説は「JPEGだから」だったが、**それは違った** ―
同じ図を品質60のJPEGで保存してもマスクは 10.82% のままで、検出も減らない。
壊すのは圧縮ではなく、線が薄いことと粒状ノイズである。拡張子は目印にすぎない。
"""

import cv2
import numpy as np
import pytest

from src.chartsymbols import (DEFAULT_BANDS, MIN_INK_SHARE, ink_mask, otsu_ink,
                              to_hsv)


def chart() -> np.ndarray:
    """黒い等圧線と記号を描いた図。気象庁PDF版に近い。"""
    rng = np.random.default_rng(3)
    img = np.full((700, 900, 3), 255, np.uint8)
    for _ in range(14):
        cv2.ellipse(img, (int(rng.integers(0, 900)), int(rng.integers(0, 700))),
                    (int(rng.integers(60, 400)), int(rng.integers(60, 300))),
                    float(rng.integers(0, 180)), 0, 360, (4, 4, 4), 3)
    cv2.putText(img, "H", (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (4, 4, 4), 8)
    return img


def faded(img: np.ndarray, level: int) -> np.ndarray:
    """線の濃さを上げる(=薄くする)。紙のスキャンを模す。"""
    out = img.copy()
    out[cv2.cvtColor(out, cv2.COLOR_RGB2GRAY) < 100] = level
    return out


def grainy(img: np.ndarray, sigma: float = 18.0) -> np.ndarray:
    rng = np.random.default_rng(0)
    return np.clip(img.astype(np.int16) + rng.normal(0, sigma, img.shape),
                   0, 255).astype(np.uint8)


def share(mask) -> float:
    return float(mask.mean())


def test_the_fixed_band_reads_a_normal_chart():
    mask, fell_back = ink_mask(chart())
    assert share(mask) > 0.05
    assert not fell_back, "普通の天気図で控えに切り替わってはいけない"


@pytest.mark.parametrize("level", [110, 140, 170])
def test_the_fixed_band_goes_empty_on_faded_lines(level):
    """これが利用者の見ている症状。固定のしきい値そのものが原因。"""
    img = faded(chart(), level)
    fixed = DEFAULT_BANDS["isobar"].mask(to_hsv(img))
    assert share(fixed) < MIN_INK_SHARE


@pytest.mark.parametrize("level", [110, 140, 170])
def test_the_fallback_recovers_faded_lines(level):
    mask, fell_back = ink_mask(faded(chart(), level))
    assert fell_back
    assert share(mask) > 0.05, "控えに切り替えても線が拾えていない"


def test_the_fallback_recovers_a_grainy_scan():
    mask, fell_back = ink_mask(grainy(faded(chart(), 145), 14))
    assert fell_back
    assert share(mask) > 0.05


def test_jpeg_compression_alone_does_not_break_it(tmp_path):
    """**最初の仮説は外れている。**圧縮そのものは線を壊さない。
    ここが赤くなったら、原因の説明を書き直すこと。"""
    from PIL import Image
    img = chart()
    for quality in (95, 85, 75, 60):
        path = tmp_path / f"q{quality}.jpg"
        Image.fromarray(img).save(path, quality=quality)
        loaded = np.array(Image.open(path).convert("RGB"))
        mask, fell_back = ink_mask(loaded)
        assert not fell_back, f"品質{quality}で控えに落ちた"
        assert share(mask) > 0.05


def test_a_readable_chart_is_untouched_to_the_pixel():
    """控えが働かない天気図では、結果が1画素も変わってはいけない。
    変わると、記録済みの交差検証の結果が再現できなくなる。"""
    img = chart()
    fixed = DEFAULT_BANDS["isobar"].mask(to_hsv(img))
    mask, _ = ink_mask(img)
    assert np.array_equal(mask, fixed)


def test_adaptive_can_be_turned_off():
    img = faded(chart(), 140)
    mask, fell_back = ink_mask(img, adaptive=False)
    assert not fell_back
    assert share(mask) < MIN_INK_SHARE


def test_otsu_does_not_depend_on_the_line_darkness():
    """濃さが違っても同じだけ拾う。それが固定のしきい値との違い。"""
    shares = [share(otsu_ink(faded(chart(), level))) for level in (4, 110, 170)]
    assert max(shares) - min(shares) < 0.01, f"濃さで結果が動く: {shares}"
