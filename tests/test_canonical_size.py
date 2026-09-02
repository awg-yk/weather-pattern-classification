"""切り取ったあとに大きさを揃える部分のテスト。

なぜ要るか
----------
同じ紙の上でも、天気図の版面(枠の大きさ)は時代で違う。実測で、生が
どちらも2339x1653なのに切り取り後は 2023年 1453x1500 / 2000年 1499x1548
だった(3.2%差)。揃えないと、時代ごとに検出の設定を打ち分けることになる。

**基準の大きさは変えてはいけない。**data/templates と同梱の重みがこの縮尺で
作られている。気象庁PDF版の切り取り結果がちょうどこの大きさなので、その
時代の画像は1画素も変わらず、いまの重みがそのまま使える。
"""

import numpy as np
import pytest
from PIL import Image

from scripts.preprocess_jma import (CANONICAL_SIZE, autocrop_to_content,
                                    fit_to_canonical, process_image)


def noise(width: int, height: int, seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (height, width, 3), dtype=np.uint8))


def test_the_canonical_size_is_what_the_weights_were_built_for():
    """変えると、いまの重みとテンプレートに違う縮尺の絵を渡すことになる。"""
    assert CANONICAL_SIZE == (1453, 1500)


def test_an_image_already_at_the_size_is_passed_through_untouched():
    """**同じ大きさへの resize でも補間で画素が変わりうる。**素通しにして
    おかないと、これまでと同じはずの画像が変わり、学習済みの重みに違う絵を
    渡すことになる。"""
    image = noise(*CANONICAL_SIZE)
    out = fit_to_canonical(image)
    assert out is image
    assert np.array_equal(np.array(out), np.array(image))


@pytest.mark.parametrize("size", [(1499, 1548), (1052, 1083), (2000, 2064)])
def test_other_sizes_are_brought_to_the_canonical_size(size):
    assert fit_to_canonical(noise(*size)).size == CANONICAL_SIZE


def test_the_aspect_ratio_barely_changes():
    """2023年 1453x1500 と 2000年 1499x1548 の縦横比の差は0.033%。
    揃えても形は歪まない。ここが大きくなる版面が現れたら考え直すこと。"""
    a = 1453 / 1500
    b = 1499 / 1548
    assert abs(a - b) / a < 0.001


def chart(frame, page=(2339, 1653)) -> Image.Image:
    """紙の上に枠を描く。枠の大きさが切り取り後を決める。"""
    arr = np.full((page[1], page[0], 3), 255, np.uint8)
    left, top, right, bottom = frame
    arr[top:bottom, left:left + 4] = 0
    arr[top:bottom, right - 4:right] = 0
    arr[top:top + 4, left:right] = 0
    arr[bottom - 4:bottom, left:right] = 0
    return Image.fromarray(arr)


def test_two_eras_on_the_same_page_end_up_the_same_size(tmp_path):
    """これが利用者の見ていた問題。実測の切り取り位置をそのまま使う。"""
    modern = chart((441, 75, 1894, 1575))     # 2023年の実測
    old = chart((426, 59, 1925, 1607))        # 2000年の実測
    assert modern.size == old.size, "紙は同じ"
    assert autocrop_to_content(modern).size != autocrop_to_content(old).size

    for name, image in (("modern.png", modern), ("old.png", old)):
        src = tmp_path / f"src_{name}"
        image.save(src)
        process_image(src, tmp_path / name, (0.65, 0.92, 1.0, 1.0))
    assert Image.open(tmp_path / "modern.png").size == CANONICAL_SIZE
    assert Image.open(tmp_path / "old.png").size == CANONICAL_SIZE


def test_resizing_can_be_turned_off(tmp_path):
    src = tmp_path / "src.png"
    chart((426, 59, 1925, 1607)).save(src)
    process_image(src, tmp_path / "out.png", (0.65, 0.92, 1.0, 1.0), size=None)
    assert Image.open(tmp_path / "out.png").size == (1499, 1548)
