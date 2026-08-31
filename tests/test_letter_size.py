"""解像度の違う天気図で検出が消える問題と、その直し方のテスト。

`cv2.matchTemplate` は大きさの違いに対応しない。テンプレートは特定の天気図
から切り出したものなので、**解像度の違う天気図では同じ H でも画素数が違い、
スコアがしきい値に届かず検出がゼロになる。**

`--scale` はこれを直せない。画像とテンプレートの両方に同じ倍率がかかるので、
相対的な大きさは変わらないからである。直せるのは `--letter-size` のほう。
"""

import ast
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.chartsymbols import match_templates, resize_template

ROOT = Path(__file__).resolve().parent.parent


def chart(scale: float = 1.0) -> np.ndarray:
    """白地に黒で H を1つ描いた、それらしい図。"""
    img = np.full((600, 800, 3), 255, np.uint8)
    cv2.putText(img, "H", (300, 350), cv2.FONT_HERSHEY_SIMPLEX,
                3.0, (4, 4, 4), 8, cv2.LINE_AA)
    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img


@pytest.fixture
def template():
    gray = cv2.cvtColor(chart(), cv2.COLOR_RGB2GRAY)
    ys, xs = np.nonzero(gray < 128)
    return {"H": gray[ys.min():ys.max() + 1, xs.min():xs.max() + 1] < 128}


def test_resize_template_keeps_the_shape_ratio():
    t = np.zeros((60, 40), dtype=bool)
    t[10:50, 10:30] = True
    assert resize_template(t, 1.0).shape == (60, 40)
    assert resize_template(t, 0.5).shape == (30, 20)
    assert resize_template(t, 2.0).shape == (120, 80)


def test_resize_never_produces_an_empty_template():
    """極端に小さい倍率でも0画素にしない。matchTemplate が落ちる。"""
    t = np.ones((4, 3), dtype=bool)
    out = resize_template(t, 0.01)
    assert out.shape[0] >= 1 and out.shape[1] >= 1


def test_detection_disappears_at_a_different_resolution(template):
    """これが利用者の見ている症状そのもの。"""
    assert len(match_templates(chart(1.0), template, threshold=0.7)) == 1
    assert len(match_templates(chart(0.6), template, threshold=0.7)) == 0


def test_a_matching_size_brings_the_detection_back(template):
    smaller = chart(0.6)
    shrunk = {"H": resize_template(template["H"], 0.6)}
    assert len(match_templates(smaller, shrunk, threshold=0.7)) == 1


def test_the_size_sweep_finds_it_without_being_told(template):
    hits = match_templates(chart(0.6), template, threshold=0.7,
                           sizes=(0.4, 0.5, 0.6, 0.8, 1.0))
    assert hits, "掃引しても見つからない"


def test_sizes_defaults_to_no_change(template):
    """既定を変えると、記録済みの実行結果が再現できなくなる。"""
    plain = match_templates(chart(1.0), template, threshold=0.7)
    explicit = match_templates(chart(1.0), template, threshold=0.7, sizes=(1.0,))
    assert len(plain) == len(explicit) == 1


def _init_worker_signature(script: str) -> list:
    """_init_worker の引数名を、実行せずにソースから読む。"""
    tree = ast.parse((ROOT / "scripts" / script).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_init_worker":
            return [a.arg for a in node.args.args]
    raise AssertionError(f"{script} に _init_worker がない")


@pytest.mark.parametrize("script", ["annotate_charts.py", "build_features.py"])
def test_the_parallel_path_also_gets_the_size(script):
    """**並列側はテンプレートを別に読み直す。**単体の経路だけ直すと、
    まとめて処理したときだけ倍率が黙って無視される。
    """
    assert "letter_size" in _init_worker_signature(script), (
        f"{script} の _init_worker が letter_size を受け取っていない。"
        "並列で処理したときだけ倍率が効かなくなる"
    )


@pytest.mark.parametrize("script", ["annotate_charts.py", "build_features.py",
                                    "predict.py"])
def test_the_option_exists(script):
    source = (ROOT / "scripts" / script).read_text(encoding="utf-8")
    assert '"--letter-size"' in source, f"{script} に --letter-size がない"


@pytest.mark.parametrize("script", ["annotate_charts.py", "build_features.py"])
def test_the_image_scale_is_not_multiplied_by_the_size(script):
    """画像側にも倍率を掛けると相対的な大きさが元に戻り、直したつもりで
    何も変わらない。analyse_chart には素の scale を渡すこと。"""
    source = (ROOT / "scripts" / script).read_text(encoding="utf-8")
    assert 'scale * letter_size' in source, "テンプレート側に掛けていない"
    assert '_WORKER["scale"] = scale * letter_size' not in source, \
        "画像側の倍率にも掛けている"
    assert 'scale=scale * letter_size' not in source, "画像側の倍率にも掛けている"
