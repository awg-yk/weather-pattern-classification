"""検出の設定を、ファイル名の日付から選ぶ部分のテスト。

**利用者に時代を覚えさせない。**以前はノートブックのセルを2つに分け、
古い天気図のときだけ letter_size="auto" と detect_threshold=0.55 を
手で渡す作りだった。渡し忘れれば黙って取りこぼす。

境目は2023-01-01(data/labels_v2.csv の始まり)。**それ以降は学習時と
まったく同じ設定で描く必要がある** ― 描き方が学習時と違うと、モデルは
見たことのない絵を渡されて成績が静かに落ちる。
"""

import pytest

from src.quicklook import OLD_ERA, TRAINING_ERA, detection_settings


def test_a_chart_from_the_training_era_uses_the_training_settings():
    settings, note = detection_settings("Js_2023010100.png")
    assert settings == TRAINING_ERA
    assert "2023-01-01" in note


def test_a_chart_from_before_the_training_era_uses_the_old_settings():
    settings, note = detection_settings("JS_2000010100_page001.png")
    assert settings == OLD_ERA
    assert "古い天気図" in note


def test_the_boundary_is_the_first_day_of_2023():
    """大晦日は古い、元日は学習時と同じ。"""
    assert detection_settings("Js_2022123112_page001.png")[0] == OLD_ERA
    assert detection_settings("Js_2023010100.png")[0] == TRAINING_ERA


def test_an_unreadable_name_falls_back_to_the_training_settings():
    """本来の用途は2023年以降なので、分からないときはそちらに寄せる。
    **黙って古い設定にすると、描き方が学習時とずれる。**"""
    settings, note = detection_settings("天気図.png")
    assert settings == TRAINING_ERA
    assert "読めない" in note


@pytest.mark.parametrize("name", ["Js_2023010100.png", "JS_2000010100_page001.png"])
def test_an_explicit_value_wins(name):
    settings, note = detection_settings(name, detect_threshold=0.42)
    assert settings["detect_threshold"] == 0.42
    assert "指定された値" in note

    settings, _ = detection_settings(name, letter_size=0.8)
    assert settings["letter_size"] == 0.8


def test_the_training_settings_match_what_the_weights_were_trained_with():
    """runs/cv_annot_boxes は しきい値0.65・原寸 で作った画像で学習した。
    ここを変えると、同梱の重みに学習時と違う絵を渡すことになる。"""
    assert TRAINING_ERA == {"letter_size": 1.0, "detect_threshold": 0.65}


def test_the_settings_are_copies():
    """返した辞書を書き換えても、次の呼び出しに影響しないこと。"""
    first, _ = detection_settings("Js_2023010100.png")
    first["detect_threshold"] = 0.1
    second, _ = detection_settings("Js_2023010100.png")
    assert second == TRAINING_ERA
