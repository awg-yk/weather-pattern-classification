"""前処理の切り取りが何で決まるかを固定するテスト。

**生の大きさが同じでも、切り取り後の大きさは同じにならない。**
`autocrop_to_content` が切るのは「非白の最大の連結成分」の外接矩形なので、
紙の大きさではなく**枠が紙のどこにどれだけの大きさで描かれているか**で決まる。

利用者が「生が同じなら前処理後も同じはず」と考えるのは自然なので、
その期待が成り立たないことを、実際に動く例として残しておく。
"""

import numpy as np
import pytest
from PIL import Image

from scripts.preprocess_jma import autocrop_to_content, crop_box


def page(frame: tuple, size: tuple = (800, 600)) -> Image.Image:
    """同じ大きさの紙に、指定した位置と大きさで枠を描く。"""
    arr = np.full((size[1], size[0], 3), 255, np.uint8)
    left, top, right, bottom = frame
    arr[top:bottom, left:left + 3] = 0
    arr[top:bottom, right - 3:right] = 0
    arr[top:top + 3, left:right] = 0
    arr[bottom - 3:bottom, left:right] = 0
    return Image.fromarray(arr)


def test_same_page_size_can_give_different_crops():
    """これが利用者の見ている現象そのもの。"""
    small = page((100, 100, 500, 400))
    large = page((80, 80, 540, 440))
    assert small.size == large.size, "紙の大きさは同じ"
    assert autocrop_to_content(small).size != autocrop_to_content(large).size


def test_the_crop_follows_the_frame_not_the_page():
    cropped = autocrop_to_content(page((100, 100, 500, 400)))
    assert cropped.size == (400, 300)


def test_the_same_frame_on_a_bigger_page_crops_the_same():
    """逆に、枠が同じなら紙が違っても切り取り後は同じになる。"""
    a = autocrop_to_content(page((100, 100, 500, 400), size=(800, 600)))
    b = autocrop_to_content(page((100, 100, 500, 400), size=(1200, 900)))
    assert a.size == b.size


def test_crop_box_reports_where_it_cut():
    left, top, right, bottom, count = crop_box(page((100, 100, 500, 400)))
    assert (left, top, right, bottom) == (100, 100, 500, 400)
    assert count >= 1


def test_an_isolated_mark_outside_the_frame_is_ignored():
    """国会図書館のビューアの灰色のボタンが図郭の外に写り込む件。
    最大の連結成分だけを使うので、離れた印は切り取りに影響しない。"""
    image = page((100, 100, 500, 400))
    arr = np.array(image)
    arr[20:40, 700:760] = 128          # 図郭から離れた灰色の印
    with_mark = Image.fromarray(arr)
    assert autocrop_to_content(with_mark).size == autocrop_to_content(image).size
    assert crop_box(with_mark)[4] > crop_box(image)[4], "塊が増えたことは分かる"


def test_a_blank_page_is_left_alone():
    blank = Image.fromarray(np.full((100, 120, 3), 255, np.uint8))
    assert autocrop_to_content(blank).size == (120, 100)
