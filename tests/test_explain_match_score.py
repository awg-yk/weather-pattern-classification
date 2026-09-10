"""一致度を数式抜きで見せる比較画像スクリプトの試験。

実際の天気図が無いため、合成天気図(`synthetic_chart`)で確かめる。
要点は「正しい位置がいちばん高く、白地がいちばん低い」という順序が
崩れないこと。数値そのもの(0.9や0.5)は絵の作り方に依存するので固定しない。
"""

import pathlib
import sys

import numpy as np
import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.build_features import load_templates_scaled
from scripts.explain_match_score import build, synthetic_chart


@pytest.fixture(scope="module")
def template():
    return load_templates_scaled(_ROOT / "data" / "templates", 1.0, quiet=True)["H"]


def test_correct_position_scores_highest(template, tmp_path):
    symbol_box = (150, 100)
    tilted_box = (340, 230)
    rgb = synthetic_chart(template, symbol_at=symbol_box, tilted_at=tilted_box)
    scores = build(rgb, template, tmp_path / "out.png", symbol_box)

    correct, tilted, blank = scores
    assert correct > tilted > blank, (
        f"正しい位置 > 傾いた記号 > 白地、の順にならない: {scores}")
    assert correct > 0.8, f"正しい位置なのにスコアが低い: {correct}"
    assert blank < 0.2, f"白地なのにスコアが高い: {blank}"


def test_output_image_has_three_panels(template, tmp_path):
    symbol_box = (150, 100)
    out_path = tmp_path / "out.png"
    build(synthetic_chart(template, symbol_at=symbol_box), template, out_path,
         symbol_box, max_width=150)
    from PIL import Image
    combined = Image.open(out_path)
    assert combined.width > 150 * 2
