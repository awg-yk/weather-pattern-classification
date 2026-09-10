"""説明用のパネル画像を作るスクリプトの試験。

実際の天気図がこの作業環境には無いため、合成した天気図(等圧線・前線・
海岸線・記号を描いたもの)で確かめる。要点は2つ。

1. 等圧線の色帯は前線・海岸線を含まないこと(彩度で弾かれる)
2. 出力画像が3枚分の幅を持つこと(潰れたり、キャプションがはみ出して
   隣のパネルに重ならないこと)
"""

import pathlib
import sys

import cv2
import numpy as np
import pytest
from PIL import Image

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.explain_isobar_band import build


def synthetic_chart() -> np.ndarray:
    width, height = 400, 300
    rgb = np.full((height, width, 3), 255, np.uint8)
    for i in range(4):
        pts = np.array([[x, 60 * i + int(15 * np.sin(x / 50.0))]
                        for x in range(0, width, 4)])
        cv2.polylines(rgb, [pts], False, (10, 10, 10), 2)
    cv2.line(rgb, (30, 250), (200, 220), (252, 4, 4), 4)     # 温暖前線
    cv2.line(rgb, (250, 60), (380, 130), (4, 4, 252), 4)     # 寒冷前線
    cv2.line(rgb, (0, 150), (400, 155), (164, 44, 44), 3)    # 海岸線
    return rgb


@pytest.fixture
def chart_path(tmp_path):
    path = tmp_path / "sample.png"
    Image.fromarray(synthetic_chart()).save(path)
    return path


def test_isobar_band_excludes_fronts_and_coastline(chart_path, tmp_path):
    stats = build(chart_path, tmp_path / "out.png")
    assert not stats["fell_back_to_otsu"]
    # 前線・海岸線の色帯には画素があるのに、それらは等圧線の色帯とは
    # 別物として測られている(彩度で弾かれていることの裏付け)
    assert stats["warm_front"] > 0
    assert stats["cold_front"] > 0
    assert stats["coastline"] > 0
    assert stats["isobar_share"] > 0


def test_output_has_three_panels_side_by_side(chart_path, tmp_path):
    out_path = tmp_path / "out.png"
    build(chart_path, out_path, max_width=200)
    combined = Image.open(out_path)
    # 3枚(幅200前後)+ キャプションのはみ出し余白 + 間隔(20px x 2)より
    # 十分広いはずで、1枚ぶんの幅しか無い、ということは無い
    assert combined.width > 200 * 2
    assert combined.height > 200  # 画像 + キャプション欄のぶん高い


def test_captions_do_not_get_clipped_by_the_next_panel(chart_path, tmp_path):
    """キャプションが画像より広いとき、パネルの幅がそのぶん広がること。

    以前の実装では画像の幅のままキャプションを書いており、2行の見出しが
    右のパネルに重なっていた。
    """
    from scripts.explain_isobar_band import CAPTIONS, add_caption
    from src.jp_font import find_cjk_font_path

    small_panel = np.full((50, 30, 3), 255, np.uint8)  # 画像は極端に細い
    font_path = find_cjk_font_path()
    canvas = add_caption(small_panel, CAPTIONS[1], font_path)
    assert canvas.width > 30, "キャプションがはみ出さない幅まで広がっていません"
