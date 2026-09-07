"""枠を「位置だけ嘘」にする対照実験の作りを確かめる。

**何を切り分ける実験か。**枠を描くと macro F1 が 0.640 -> 0.669 に上がったが、
効いたのが「枠が正しい位置にあること」なのか「単に目立つ印が付いたこと」なのかは
まだ分かっていない。個数も見た目もそのままに位置だけを嘘にすれば、切り分けられる。

ここで守りたいのは、**対照として成り立っていること**。
自分の枠が残っていたり、枠の個数が変わっていたりすると比較にならない。
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.shuffle_annotations import _as_detections, derange


def test_no_chart_keeps_its_own_boxes():
    """自分に当たったままだと、その1枚だけ「正しい位置の枠」になってしまう。
    対照として成り立たない。"""
    names = [f"chart{i}.png" for i in range(30)]
    assigned = derange(names, seed=0)
    assert set(assigned) == set(names), "取りこぼしがある"
    assert all(assigned[n] != n for n in names), "自分の枠が残っている天気図がある"


def test_every_box_set_is_used_exactly_once():
    """**枠の個数の分布を変えない。**割り当てが並べ替えなら、データ全体で見た
    「1枚あたりの枠の数」の分布は元と完全に同じになる。ここが崩れると、
    位置ではなく枠の数を測ってしまう。"""
    names = [f"chart{i}.png" for i in range(30)]
    assigned = derange(names, seed=1)
    assert sorted(assigned.values()) == sorted(names), "並べ替えになっていない"


def test_the_assignment_is_reproducible():
    names = [f"chart{i}.png" for i in range(20)]
    assert derange(names, seed=7) == derange(names, seed=7)
    assert derange(names, seed=7) != derange(names, seed=8)


def test_it_refuses_when_it_cannot_avoid_itself():
    """1枚しかなければ自分以外を割り当てられない。黙って自分を渡さない。"""
    with pytest.raises(SystemExit):
        derange(["only.png"], seed=0)


def test_the_stored_coordinates_come_back_as_detections():
    found = {"highs": [[0.1, 0.2]], "lows": [[0.3, 0.4], [0.5, 0.6]],
             "edge_highs": [], "edge_lows": [[0.7, 0.8]]}
    detections = _as_detections(found)
    assert detections.highs == [(0.1, 0.2)]
    assert len(detections.lows) == 2
    assert detections.edge_highs == []
    assert detections.edge_lows == [(0.7, 0.8)]


def test_a_missing_key_is_treated_as_empty():
    """古い記録に列が無くても落ちない。"""
    detections = _as_detections({"highs": [[0.1, 0.2]]})
    assert detections.lows == []
    assert detections.edge_lows == []


def _chart(directory: Path, stamp: str) -> Path:
    from PIL import Image

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"Js_{stamp}_page001.png"
    Image.new("RGB", (200, 200), "white").save(path)
    return path


def test_it_draws_borrowed_boxes_and_says_so(tmp_path):
    """通しで動くこと。枠は借りものなので、元の天気図とは合わない。"""
    charts = tmp_path / "in"
    for stamp in ("2023010100", "2023010112", "2023010200"):
        _chart(charts, stamp)
    found = {
        "Js_2023010100_page001.png": {"highs": [[0.2, 0.2]], "lows": [],
                                      "edge_highs": [], "edge_lows": []},
        "Js_2023010112_page001.png": {"highs": [], "lows": [[0.8, 0.8]],
                                      "edge_highs": [], "edge_lows": []},
        "Js_2023010200_page001.png": {"highs": [[0.5, 0.5]], "lows": [[0.6, 0.3]],
                                      "edge_highs": [], "edge_lows": []},
    }
    detections_path = tmp_path / "detections.json"
    detections_path.write_text(json.dumps(found), encoding="utf-8")

    out = tmp_path / "out"
    done = subprocess.run(
        [sys.executable, "-m", "scripts.shuffle_annotations",
         "--in-dir", str(charts), "--detections", str(detections_path),
         "--out-dir", str(out)],
        cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr
    assert len(list(out.iterdir())) == 3, "全部の天気図が書き出されていない"
    # 狙いが伝わる案内が出ること(外れているのが正しい、と分かるように)
    assert "外れて" in done.stdout


def test_charts_without_a_record_are_reported_not_skipped_silently(tmp_path):
    """記録に無い天気図を黙って飛ばすと、枚数が合わない理由が分からなくなる。"""
    charts = tmp_path / "in"
    for stamp in ("2023010100", "2023010112", "2023010200"):
        _chart(charts, stamp)
    found = {f"Js_{s}_page001.png": {"highs": [[0.2, 0.2]], "lows": [],
                                     "edge_highs": [], "edge_lows": []}
             for s in ("2023010100", "2023010112")}
    detections_path = tmp_path / "detections.json"
    detections_path.write_text(json.dumps(found), encoding="utf-8")

    done = subprocess.run(
        [sys.executable, "-m", "scripts.shuffle_annotations",
         "--in-dir", str(charts), "--detections", str(detections_path),
         "--out-dir", str(tmp_path / "out")],
        cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "1枚" in done.stdout and "検出の記録に無い" in done.stdout
