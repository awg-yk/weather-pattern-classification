"""入手経路の違う画像を突き合わせる部分のテスト。

同じ日の天気図が、気象庁PDF版と国会図書館版の2通りで手に入る。
**同じ日でも絵が違えば、片方で学習した重みはもう片方に使えない。**
推測ではなく測って決めるための道具なので、判定が効いていることを固定する。
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def chart(seed: int, size=(300, 320)) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(
        rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))


@pytest.fixture
def pair(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    return a, b


def run(a: Path, b: Path, *extra) -> str:
    result = subprocess.run(
        [sys.executable, "-m", "scripts.compare_sources",
         "--a", str(a), "--b", str(b), "--a-processed", "--b-processed",
         "--templates", str(ROOT / "data" / "templates"),
         # 角度の掃引は判定に関係しない。テストを速くするため0度だけにする
         "--angle-range", "0", "--angle-step", "5", *extra],
        cwd=ROOT, capture_output=True, text=True)
    return result.stdout + result.stderr


def test_identical_images_are_reported_as_the_same(pair):
    a, b = pair
    image = chart(0)
    image.save(a / "Js_2023010100.png")
    image.save(b / "Js_2023010100_page001.png")
    out = run(a, b)
    assert "0.00%" in out
    assert "ほぼ同じ画像です" in out, out


def test_different_images_are_reported_as_different(pair):
    a, b = pair
    chart(0).save(a / "Js_2023010100.png")
    chart(99).save(b / "Js_2023010100_page001.png")
    out = run(a, b)
    assert "別物と考えるべきです" in out, out


def test_it_matches_across_naming_conventions(pair):
    """気象庁版は Js_2023010100.png、国会図書館版は Js_2023010100_page001.png。
    名前が違っても同じ日時として突き合わせること。"""
    a, b = pair
    image = chart(1)
    image.save(a / "Js_2023010100.png")
    image.save(b / "JS_2023010100_page001.png")
    assert "両方にある日時: 1件" in run(a, b)


def test_no_overlap_says_so_instead_of_a_traceback(pair):
    a, b = pair
    chart(0).save(a / "Js_2023010100.png")
    chart(0).save(b / "Js_2024010100_page001.png")
    out = run(a, b)
    assert "Traceback" not in out, out
    assert "同じ日時の天気図が1つもありません" in out


def test_the_script_actually_runs(pair):
    """**main ガードを書き忘れると、何も起きずに終了コード0で終わる。**
    実際にそうなり、--help を /dev/null に流していたので気づけなかった。"""
    a, b = pair
    chart(0).save(a / "Js_2023010100.png")
    chart(0).save(b / "Js_2023010100_page001.png")
    assert run(a, b).strip(), "何も出力されていない"


def test_the_model_comparison_separates_identical_from_different(pair):
    """**画素が違ってもモデルの出力が同じなら、重みはそのまま使える。**
    そこが「一本化してよいか」の決め手なので、判定が効くことを固定する。"""
    a, b = pair
    weights = ROOT / "weights" / "model.pt"
    if not weights.exists():
        pytest.skip("重みが無い")

    image = chart(0, size=(400, 420))
    image.save(a / "Js_2023010100.png")
    image.save(b / "Js_2023010100_page001.png")
    same = run(a, b, "--weights", str(weights))
    assert "確信度の差" in same, same
    assert "0.0000" in same or "0.000 / 0.000" in same, same

    chart(99, size=(400, 420)).save(b / "Js_2023010100_page001.png")
    differ = run(a, b, "--weights", str(weights))
    assert "確信度の差" in differ, differ


def test_the_model_is_not_loaded_unless_asked(pair):
    """--weights を渡さないときに重い読み込みを走らせないこと。"""
    a, b = pair
    image = chart(0)
    image.save(a / "Js_2023010100.png")
    image.save(b / "Js_2023010100_page001.png")
    out = run(a, b)
    assert "モデル:" not in out
    assert "確信度の差" not in out
