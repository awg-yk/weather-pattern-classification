"""Colabノートブックの版表示が食い違っていないかを確かめる。

ノートブック自身が「セットアップ後に出る `バージョン: vX` がタイトルの版と
一致していれば最新版が動いています」と案内している。**その2か所がずれると、
この案内そのものが嘘になる**(実際にタイトルが v12、実体が v13 でずれていた)。
Colabは古いコピーを掴んだままになりやすく、利用者が「反映されたか」を
確かめる手段がこれしかないので、ずれたままだと気づけない。
"""

import json
import re
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parent.parent / "notebooks" / "predict.ipynb"


def _sources():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        src = cell["source"]
        yield cell["cell_type"], src if isinstance(src, str) else "".join(src)


def test_title_and_constant_agree():
    title_version = constant_version = None
    for kind, text in _sources():
        if kind == "markdown":
            found = re.search(r"すぐに使う \((v\d+)\)", text)
            if found:
                title_version = found.group(1)
        else:
            found = re.search(r'NOTEBOOK_VERSION\s*=\s*"(v\d+)"', text)
            if found:
                constant_version = found.group(1)

    assert title_version, "タイトルに版が入っていない"
    assert constant_version, "NOTEBOOK_VERSION が見つからない"
    assert title_version == constant_version, (
        f"版がずれている: タイトル {title_version} / NOTEBOOK_VERSION {constant_version}。"
        "ノートブックの案内文がこの2つの一致を前提にしている"
    )


def test_version_is_printed_at_the_end_of_setup():
    """表示が消えると、利用者は版を確かめる手段を失う。"""
    assert any("NOTEBOOK_VERSION" in text and "print" in text
               for kind, text in _sources() if kind == "code"), \
        "セットアップセルがバージョンを表示していない"


def test_branch_is_selectable_from_the_form():
    """注釈方式の重みとテンプレートはまだmainに入っていない。ブランチを
    選べないと、Colabからは新機能を一切試せない(実際にそうなっていた)。"""
    setup = next(text for kind, text in _sources()
                 if kind == "code" and "BRANCH" in text)
    assert re.search(r'BRANCH\s*=\s*"[^"]*"\s*#@param', setup), \
        "BRANCH が #@param になっていないと、フォームから変更できない"


def test_setup_reports_whether_annotation_is_usable():
    """使えないまま USE_ANNOTATION にチェックを入れると素の方式に落ちるだけで、
    「チェックしたのに何も変わらない」と見える。セットアップ時点で知らせる。"""
    setup = next(text for kind, text in _sources()
                 if kind == "code" and "NOTEBOOK_VERSION" in text)
    tail = setup[setup.index("セットアップ完了"):]
    assert "annotation_available()" in tail, \
        "セットアップの最後に注釈方式の可否を確かめていない"
    assert "BRANCH" in tail, "使えないときに、どのブランチだったかを示していない"
