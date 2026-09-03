"""Colabと手元のノートブックが、同じ部品を呼んでいることを確かめる。

以前はノートブックの中に関数を書き写していた。**それだと片方だけ直して
食い違う。**この計画では「学習に使った描き方と推論の描き方が食い違うと
成績が静かに落ちる」という失敗をしているので、描き方を決める場所は
`src/quicklook.py` の1か所にしておく。
"""

import ast
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = {"colab": ROOT / "notebooks" / "predict.ipynb",
             "local": ROOT / "notebooks" / "predict_local.ipynb"}
SHARED = ("annotation_available", "classify_and_show")


def without_magics(code: str) -> str:
    """IPythonの %マジック・!コマンドを取り除き、Pythonとして読める形にする。

    **字下げを保つこと。**`else:` の中の `%matplotlib inline` を字下げなしの
    `pass` に置き換えると、そこで IndentationError になる(実際にそうなった)。
    """
    out = []
    for line in code.split("\n"):
        found = re.match(r"(\s*)[%!]", line)
        out.append(f"{found.group(1)}pass" if found else line)
    return "\n".join(out)


def code_of(path: Path) -> str:
    nb = json.loads(path.read_text(encoding="utf-8"))
    parts = []
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = cell["source"]
        parts.append(src if isinstance(src, str) else "".join(src))
    return "\n".join(parts)


@pytest.mark.parametrize("name", sorted(NOTEBOOKS))
def test_the_notebook_imports_the_shared_module(name):
    code = code_of(NOTEBOOKS[name])
    assert "from src.quicklook import" in code, (
        f"{name} が src/quicklook.py を使っていない。"
        "ノートブックに関数を書き写すと片方だけ直して食い違う"
    )


@pytest.mark.parametrize("name", sorted(NOTEBOOKS))
def test_the_notebook_does_not_redefine_the_shared_functions(name):
    code = code_of(NOTEBOOKS[name])
    cleaned = without_magics(code)
    defined = {n.name for n in ast.walk(ast.parse(cleaned))
               if isinstance(n, ast.FunctionDef)}
    clashes = sorted(defined & set(SHARED))
    assert not clashes, f"{name} が共通の関数を上書きしている: {clashes}"


@pytest.mark.parametrize("name", sorted(NOTEBOOKS))
def test_the_notebook_code_parses(name):
    code = code_of(NOTEBOOKS[name])
    ast.parse(without_magics(code))


def test_the_module_paths_are_absolute():
    """カレントディレクトリがどこでも動くこと。VS Code はノートブックのある
    場所をカレントにすることがある。"""
    from src import quicklook

    for attr in ("DEFAULT_WEIGHTS", "ANNOT_WEIGHTS", "TEMPLATES_DIR", "MARKS_DIR"):
        assert Path(getattr(quicklook, attr)).is_absolute(), f"{attr} が相対パス"


def test_annotation_availability_reports_what_is_missing(tmp_path):
    from src.quicklook import annotation_available

    ok, missing = annotation_available(tmp_path / "nope.pt", tmp_path / "nodir")
    assert not ok
    assert len(missing) == 2

    (tmp_path / "w.pt").write_bytes(b"")
    (tmp_path / "t").mkdir()
    assert annotation_available(tmp_path / "w.pt", tmp_path / "t") == (True, [])


def test_the_local_notebook_stays_inside_the_repository():
    """手元版は**このフォルダの中だけで完結**させる。隣の
    weather-pattern-classification-data を参照すると、フォルダを移した
    だけで動かなくなる。"""
    code = code_of(NOTEBOOKS["local"])
    assert "weather-pattern-classification-data" not in code, (
        "手元版が隣のリポジトリを参照している"
    )
    assert "/content/" not in code, "手元版にColab用のパスが混ざっている"


def test_the_local_notebook_builds_paths_from_the_repository_root():
    """カレントディレクトリに依らず動くこと。"""
    code = code_of(NOTEBOOKS["local"])
    assert "ROOT = Path.cwd()" in code, "ROOT を決めていない"
    lines = [ln for ln in code.split("\n") if ln.startswith("IMAGE =")]
    assert lines, "IMAGE が無い"
    for line in lines:
        assert "ROOT /" in line, f"IMAGE が ROOT からの相対になっていない: {line}"


def test_magics_keep_their_indentation():
    """字下げなしの pass に置き換えると、条件分岐の中のマジックで
    IndentationError になる。実際にそれでテストが赤くなった。"""
    code = "if x:\n    %matplotlib inline\n    y = 1\nelse:\n    !pip install z\n    y = 2\n"
    ast.parse(without_magics(code))


def test_the_local_notebook_does_not_split_cells_by_era():
    """時代ごとにセルを分けない。**利用者に時代を覚えさせる作りは、
    渡し忘れれば黙って取りこぼす。**前処理がすべての天気図を
    1453x1500 に揃えるので、設定は1つで足りる。"""
    code = code_of(NOTEBOOKS["local"])
    for gone in ("OLD_IMAGE", "detect_threshold=0.55", 'letter_size="auto"',
                 "processed/ndl", "processed/jma", '"ndl"', '"jma"'):
        assert gone not in code, f"時代ごとの打ち分けが残っている: {gone}"


def test_the_local_notebook_can_classify_by_date():
    """日付を書き換えるだけで分類できること。画像のファイル名を
    覚えていなくても使えるようにするための入口。"""
    code = code_of(NOTEBOOKS["local"])
    assert "classify_date" in code, "日付で引くセルが無い"
    assert re.search(r'^DATE\s*=\s*"\d{4}-\d{2}-\d{2}"', code, re.M), \
        "DATE を書き換える形になっていない"
    assert re.search(r"^HOUR\s*=\s*(0|12)\b", code, re.M), "HOUR が無い"


def test_annotated_images_are_written_outside_the_image_folder():
    """描き込んだ画像を入力の隣に置くと、**学習に使うフォルダに派生画像が
    混ざる。**次の学習でそれごと拾ってしまう。"""
    from src import quicklook

    annotated = Path(quicklook.ANNOTATED_DIR).resolve()
    processed = Path(quicklook.PROCESSED_DIR).resolve()
    assert processed not in annotated.parents and annotated != processed, (
        f"描き込みの書き出し先が画像フォルダの中にある: {annotated}"
    )


def _fake_chart(directory: Path, stamp: str) -> Path:
    from PIL import Image

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"Js_{stamp}_page001.png"
    Image.new("RGB", (8, 8), "white").save(path)
    return path


def test_chart_for_matches_by_timestamp_not_by_filename(tmp_path):
    """取得元でファイル名の表記が違う(Js_2023010100.png と
    Js_2023010100_page001.png)。名前の厳密一致で引くと、画像は手元に
    あるのに「見つからない」と言って止まる。"""
    from src.quicklook import chart_for

    made = _fake_chart(tmp_path, "2004011500")
    for written in ("2004-01-15", "2004/01/15", "20040115"):
        assert chart_for(written, 0, tmp_path) == made, f"{written} で引けない"


def test_chart_for_says_what_it_looked_for_and_what_is_there(tmp_path):
    """「ありません」だけでは、日付の書き方が悪いのか画像が無いのかが
    分からない。探した名前と、手元にある範囲を出す。"""
    from src.quicklook import chart_for

    _fake_chart(tmp_path, "2004011500")
    with pytest.raises(SystemExit) as caught:
        chart_for("2004-01-15", 12, tmp_path)
    message = str(caught.value)
    assert "2004011512" in message, "探した名前が出ていない"
    assert "20040115" in message, "手元にある範囲が出ていない"
    assert "12" in message and "hour" in message, "hour の決まりが出ていない"


def test_chart_for_points_at_the_preprocessing_command_when_empty(tmp_path):
    from src.quicklook import chart_for

    empty = tmp_path / "からっぽ"
    empty.mkdir()
    with pytest.raises(SystemExit) as caught:
        chart_for("2004-01-15", 0, empty)
    assert "preprocess_jma" in str(caught.value), "作り方が案内されていない"


def test_chart_for_notices_images_added_after_the_index_was_built(tmp_path):
    """索引は使い回すが、**フォルダが変わったら作り直す。**そうしないと、
    ノートブックを開いたまま画像を足したときに「ありません」と言い続ける。"""
    from src.quicklook import chart_for

    _fake_chart(tmp_path, "2004011500")
    chart_for("2004-01-15", 0, tmp_path)  # ここで索引ができる

    added = _fake_chart(tmp_path, "2004011600")
    assert chart_for("2004-01-16", 0, tmp_path) == added, (
        "索引を作り直していない。開いたまま画像を足すと引けなくなる"
    )
