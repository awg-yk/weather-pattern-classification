r"""ソース全体に「無効なエスケープ」の警告が無いことを確かめる。

**Windowsのパスをdocstringに書くと壊れる。**`..\weather-...` の `\w` は
Pythonのエスケープとして無効で、読み込むたびに警告が出る:

    SyntaxWarning: invalid escape sequence '\w'

この文書は日本語Windowsの利用者向けにPowerShellのコマンド例を多く載せるので
踏みやすい。raw文字列 (r\"\"\"...\"\"\") にすれば直る。将来のPythonでは
SyntaxError になる予定なので、警告のうちに潰しておく。

**種類ではなく本文で見ること。**Python 3.11 では DeprecationWarning、
3.12以降では SyntaxWarning として出る。開発機と利用者の環境で版が違うと、
種類で見ている検査は片方で素通りする(実際にそうなっていた)。
"""

import pathlib
import warnings

ROOT = pathlib.Path(__file__).resolve().parent.parent
SKIP = {".git", "__pycache__", ".pytest_cache", ".venv", "venv", "wpcvenv"}
MESSAGE = "invalid escape sequence"


def escape_warnings(source: str, filename: str) -> list:
    """無効なエスケープの警告だけを、Pythonの版に依らず拾う。"""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compile(source, filename, "exec")
        return [w for w in caught if MESSAGE in str(w.message)]


def python_files():
    for path in sorted(ROOT.glob("**/*.py")):
        if SKIP & set(path.parts):
            continue
        yield path


def test_no_invalid_escapes_anywhere():
    offenders = []
    for path in python_files():
        for w in escape_warnings(path.read_text(encoding="utf-8"), str(path)):
            offenders.append(f"{path.relative_to(ROOT)}: {w.message}")
    assert not offenders, (
        "無効なエスケープがあります。docstringにWindowsのパスを書いた場合は "
        "raw 文字列 (r\"\"\") にしてください:\n  " + "\n  ".join(offenders)
    )


def test_the_check_actually_sees_a_bad_docstring():
    """検査そのものが効いていることを確かめる。ここが通らなくなったら、
    上の検査は何も見ていない。"""
    # 本物の1個のバックスラッシュを書く。2個ではエスケープとして正しくなる
    bad = chr(34) * 3 + "..\\weather" + chr(34) * 3 + "\n"
    assert escape_warnings(bad, "bad.py"), "壊れたdocstringを見逃している"


def test_a_raw_string_is_accepted():
    good = "r" + chr(34) * 3 + "..\\weather" + chr(34) * 3 + "\n"
    assert not escape_warnings(good, "good.py")


def test_the_scan_covers_this_repository():
    """対象が空だと、上の検査は黙って通る。"""
    files = list(python_files())
    assert len(files) > 50, f"走査できたのは{len(files)}件だけ"
    assert any(p.name == "predict.py" for p in files)
