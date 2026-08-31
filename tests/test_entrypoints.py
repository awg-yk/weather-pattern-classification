"""READMEに書いてあるコマンドの形がそのまま動くかを確かめる。

`python scripts/predict.py ...` で起動すると、sys.path に入るのは `scripts/`
だけで、リポジトリのルートは入らない。そのため `from scripts...` や
`from src...` が `ModuleNotFoundError` で落ちる。`python -m scripts.predict`
では起きないので、テストを `-m` の形で書くと素通りしてしまう。
**ここでは実際にスクリプトのパスを渡して起動する。**

利用者に最初に触られるのがこの経路なので、壊れると「動かない」で終わる。
"""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# READMEで `python scripts/X.py` の形で案内しているもの。
DOCUMENTED = ["predict.py"]


def _readme_entrypoints():
    """READMEから `python scripts/X.py` を拾う。案内を増やしたら勝手に増える。"""
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    import re
    return sorted(set(re.findall(r"python scripts/([A-Za-z0-9_]+\.py)", text)))


def test_readme_still_documents_the_script_path_form():
    """READMEが `-m` 形式に書き換わったらこのテストの前提が消える。"""
    found = _readme_entrypoints()
    assert set(DOCUMENTED) <= set(found), (
        f"READMEから消えた案内がある: {sorted(set(DOCUMENTED) - set(found))}"
    )


@pytest.mark.parametrize("script", DOCUMENTED)
def test_runs_as_a_script_path(script, tmp_path):
    """カレントディレクトリがリポジトリの外でも起動できること。

    `--help` でも import は全部走るので、これで足りる。
    """
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), "--help"],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_annotate_defaults_do_not_depend_on_the_current_directory():
    """--templates / --marks の既定値が相対パスだと、リポジトリの外から
    起動したときにテンプレートを見失う(実際に一度そうなった)。"""
    from scripts.predict import build_parser

    args = build_parser().parse_args(["dummy.png"])
    for name in ("templates", "marks", "weights"):
        value = getattr(args, name)
        assert Path(value).is_absolute(), f"--{name} の既定値が相対パス: {value}"


def test_no_arguments_gives_the_japanese_error(tmp_path):
    """引数なしのときの案内。ここは `parser.error()` を呼ぶので、
    パーサを別関数に切り出したときに参照が切れやすい(実際に一度切れた)。"""
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "predict.py")],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "画像パスまたは --date" in result.stderr
    assert "Traceback" not in result.stderr
