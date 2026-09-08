"""既にある注釈付き画像が、いまの検出設定と同じ条件で作られたかを確かめる道具。

**枠の個数が本番と揃っていないと、対照実験の比較が成り立たない。**
`--marks` を付けたかどうかで拾える数が変わるが、本番を作ったときの引数は
記録に残っていない。そこで画像に描かれた枠を数えて突き合わせる。
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.annotate_charts import HIGH_COLOR, LOW_COLOR, draw_annotations
from scripts.check_annot_match import count_boxes
from src.chartfeatures import ChartDetections


def _drawn(highs=(), lows=(), size=(400, 400)):
    rgb = np.full((size[1], size[0], 3), 255, dtype=np.uint8)
    detections = ChartDetections(highs=[tuple(p) for p in highs],
                                 lows=[tuple(p) for p in lows])
    return draw_annotations(rgb, {}, detections, boxes=True, fronts=False)


def test_it_counts_the_boxes_that_were_drawn():
    image = _drawn(highs=[(0.25, 0.25), (0.75, 0.25)], lows=[(0.5, 0.75)])
    assert count_boxes(image, HIGH_COLOR) == 2
    assert count_boxes(image, LOW_COLOR) == 1


def test_an_image_without_boxes_counts_zero():
    blank = np.full((200, 200, 3), 255, dtype=np.uint8)
    assert count_boxes(blank, HIGH_COLOR) == 0
    assert count_boxes(blank, LOW_COLOR) == 0


def test_the_two_colours_are_not_confused():
    """高気圧の緑と低気圧の橙を取り違えると、突き合わせが嘘になる。"""
    only_high = _drawn(highs=[(0.5, 0.5)])
    assert count_boxes(only_high, HIGH_COLOR) == 1
    assert count_boxes(only_high, LOW_COLOR) == 0


def _setup(tmp_path, drawn_lows: bool):
    """注釈付き画像と、検出の記録を用意する。drawn_lows=False で枠を減らす。"""
    annot = tmp_path / "annot"
    annot.mkdir()
    found = {}
    for i, stamp in enumerate(("2023010100", "2023010112", "2023010200")):
        name = f"Js_{stamp}_page001.png"
        highs = [(0.3, 0.3)]
        lows = [(0.6, 0.7)]
        Image.fromarray(_drawn(highs=highs, lows=lows if drawn_lows else [])).save(
            annot / name)
        found[name] = {"highs": [list(p) for p in highs],
                       "lows": [list(p) for p in lows],
                       "edge_highs": [], "edge_lows": []}
    path = tmp_path / "detections.json"
    path.write_text(json.dumps(found), encoding="utf-8")
    return annot, path


def _run(annot, detections):
    return subprocess.run(
        [sys.executable, "-m", "scripts.check_annot_match",
         "--annot-dir", str(annot), "--detections", str(detections), "--limit", "3"],
        cwd=ROOT, capture_output=True, text=True)


def test_it_says_so_when_the_conditions_match(tmp_path):
    annot, detections = _setup(tmp_path, drawn_lows=True)
    done = _run(annot, detections)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "同じ条件で作られています" in done.stdout


def test_it_catches_a_mismatch_and_says_which_way(tmp_path):
    """**黙って通してはいけない。**条件が違うまま比べると、位置以外の違いも
    一緒に測ることになる。どちらに揃えればよいかまで出す。"""
    annot, detections = _setup(tmp_path, drawn_lows=False)
    done = _run(annot, detections)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "条件が違います" in done.stdout
    assert "--marks" in done.stdout, "どう直せばよいかが出ていない"


def test_it_refuses_when_there_is_nothing_to_compare(tmp_path):
    empty = tmp_path / "annot"
    empty.mkdir()
    detections = tmp_path / "detections.json"
    detections.write_text(json.dumps({"Js_2023010100_page001.png": {}}),
                          encoding="utf-8")
    done = _run(empty, detections)
    assert done.returncode != 0
    assert "突き合わせられる画像がありません" in done.stdout + done.stderr
