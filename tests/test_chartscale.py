"""天気図の大きさから、テンプレートの倍率を決める部分のテスト。

**この仕組みで直せるのは解像度の違いだけ。**記号の線の太さが違う天気図
(国立国会図書館由来の2000〜2022年)には効かない。実測では、基準の幅を
1052〜1900pxのどこに置いても検出はほぼ0個のままだった。そちらは
テンプレートを切り直すしかない(`docs` と `data/templates/README.md`)。
"""

import json

import pytest

from src.chartscale import (SPREAD, auto_letter_size, letter_size_arg,
                            load_reference, save_reference, sizes_around)


@pytest.fixture
def templates(tmp_path):
    return tmp_path


def test_without_a_reference_it_does_nothing(templates):
    size, note = auto_letter_size(1052, templates)
    assert size == 1.0, "基準が無いのに勝手に縮めてはいけない"
    assert "reference.json" in note, "理由を言っていない"


def test_the_ratio_is_width_over_reference(templates):
    save_reference(templates, 1500)
    size, note = auto_letter_size(750, templates)
    assert size == pytest.approx(0.5)
    assert "1500" in note and "750" in note


def test_the_same_size_chart_is_left_alone(templates):
    save_reference(templates, 1460)
    size, _ = auto_letter_size(1460, templates)
    assert size == pytest.approx(1.0)


def test_a_broken_reference_file_is_ignored(templates):
    (templates / "reference.json").write_text("これはJSONではない", encoding="utf-8")
    assert load_reference(templates) is None
    assert auto_letter_size(1052, templates)[0] == 1.0


@pytest.mark.parametrize("bad", [0, -5, "1500", None])
def test_a_nonsense_width_is_ignored(templates, bad):
    (templates / "reference.json").write_text(
        json.dumps({"chart_width": bad}), encoding="utf-8")
    assert load_reference(templates) is None


def test_the_spread_is_centred_on_the_estimate():
    assert sizes_around(1.0) == (1.0,), "調整しないときは振らない"
    around = sizes_around(0.6)
    assert len(around) == len(SPREAD)
    assert min(around) < 0.6 < max(around)
    assert around[len(around) // 2] == pytest.approx(0.6)


def test_the_spread_stays_narrow():
    """広げすぎると、縮んだテンプレートが等圧線の模様を拾い始める。"""
    assert max(SPREAD) / min(SPREAD) < 1.4


def test_the_option_accepts_auto_and_numbers():
    assert letter_size_arg("auto") == "auto"
    assert letter_size_arg("0.7") == pytest.approx(0.7)
    with pytest.raises(ValueError):
        letter_size_arg("おおきめ")
