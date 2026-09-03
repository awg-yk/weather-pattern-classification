"""日付一覧の判定を、手元の画像だけで回せることを確かめる。

以前はアーカイブから取得する経路しかなかった。**手元に17,898枚あるのに
1日ずつ通信しに行くのは遅いうえ、通信が通らない場所では回せない。**
--images-dir を渡したときは、通信せずそのフォルダから引く。
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "weights" / "model.pt"


def _chart(directory: Path, stamp: str) -> Path:
    """それらしい大きさの天気図を1枚でっちあげる(中身は問わない)。"""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"Js_{stamp}_page001.png"
    Image.new("RGB", (1453, 1500), "white").save(path)
    return path


def _run(*args):
    return subprocess.run(
        [sys.executable, "-m", "scripts.classify_dates", *args],
        cwd=ROOT, capture_output=True, text=True,
    )


def test_the_arguments_exist():
    """--help が通ることだけでなく、引数が実際に生えていること。"""
    done = _run("--help")
    assert done.returncode == 0, done.stderr
    for flag in ("--images-dir", "--annotate", "--templates", "--marks"):
        assert flag in done.stdout, f"{flag} が無い"


def test_an_empty_folder_says_how_to_fill_it(tmp_path):
    """「1枚もありません」だけでは、どうすればよいか分からない。"""
    empty = tmp_path / "からっぽ"
    empty.mkdir()
    (tmp_path / "dates.csv").write_text("発生日\n2004-01-15\n", encoding="utf-8")
    done = _run("--dates-csv", str(tmp_path / "dates.csv"),
                "--images-dir", str(empty),
                "--weights", str(WEIGHTS), "--out", str(tmp_path / "out.csv"))
    assert done.returncode != 0
    assert "preprocess_jma" in done.stdout + done.stderr, "作り方が案内されていない"


@pytest.mark.skipif(not WEIGHTS.exists(), reason="重みが無い")
def test_it_classifies_from_local_images_without_the_network(tmp_path):
    """通信しないこと。取得経路に落ちていれば、この環境では失敗するはず。"""
    charts = tmp_path / "charts"
    _chart(charts, "2004011500")
    _chart(charts, "2004011600")
    dates = tmp_path / "dates.csv"
    dates.write_text("発生日\n2004-01-15\n2004-01-16\n", encoding="utf-8")
    out = tmp_path / "out.csv"

    done = _run("--dates-csv", str(dates), "--images-dir", str(charts),
                "--weights", str(WEIGHTS), "--out", str(out))
    assert done.returncode == 0, done.stdout + done.stderr
    assert "通信しません" in done.stdout

    got = pd.read_csv(out)
    assert len(got) == 2
    assert got["ラベル"].notna().all(), "判定できていない行がある"
    assert (got["判定"] == "確定").sum() + (got["判定"] == "要確認").sum() == 2


@pytest.mark.skipif(not WEIGHTS.exists(), reason="重みが無い")
def test_a_missing_date_is_reported_not_silently_dropped(tmp_path):
    """手元に無い日を黙って落とすと、件数だけ合わなくなって原因が分からない。"""
    charts = tmp_path / "charts"
    _chart(charts, "2004011500")
    dates = tmp_path / "dates.csv"
    dates.write_text("発生日\n2004-01-15\n2004-01-17\n", encoding="utf-8")
    out = tmp_path / "out.csv"

    done = _run("--dates-csv", str(dates), "--images-dir", str(charts),
                "--weights", str(WEIGHTS), "--out", str(out))
    assert done.returncode == 0, done.stdout + done.stderr

    got = pd.read_csv(out)
    assert len(got) == 2, "行が消えている"
    missing = got[got["発生日"] == "2004-01-17"].iloc[0]
    assert "2004011700" in str(missing["備考"]), \
        f"どの日時を探したかが残っていない: {missing['備考']}"
