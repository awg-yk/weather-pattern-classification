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
    cleaned = "\n".join("pass" if re.match(r"\s*[%!]", ln) else ln
                        for ln in code.split("\n"))
    defined = {n.name for n in ast.walk(ast.parse(cleaned))
               if isinstance(n, ast.FunctionDef)}
    clashes = sorted(defined & set(SHARED))
    assert not clashes, f"{name} が共通の関数を上書きしている: {clashes}"


@pytest.mark.parametrize("name", sorted(NOTEBOOKS))
def test_the_notebook_code_parses(name):
    code = code_of(NOTEBOOKS[name])
    cleaned = "\n".join("pass" if re.match(r"\s*[%!]", ln) else ln
                        for ln in code.split("\n"))
    ast.parse(cleaned)


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
