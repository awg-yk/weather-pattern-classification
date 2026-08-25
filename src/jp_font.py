"""日本語フォントを探す。matplotlibの図とPILの描画の両方で使う。

Colabのように実行中にフォントを入れた環境では、matplotlibの起動時キャッシュに
載っておらずOS側(fc-list)を直接見ないと拾えない。一方Windowsにはfc-listが無い。
その両方を吸収する。
"""

import subprocess
import sys
from pathlib import Path

import matplotlib
import matplotlib.font_manager as fm

# matplotlibに同梱されがちな環境で見かける日本語フォントの候補。
# Windows(Yu Gothic/Meiryo/MS Gothic)・mac(Hiragino)・Linux(Noto/IPA)を順に試す。
CJK_FAMILIES = (
    "Yu Gothic",
    "Meiryo",
    "MS Gothic",
    "Hiragino Sans",
    "Hiragino Kaku Gothic ProN",
    "Noto Sans CJK JP",
    "Noto Sans JP",
    "IPAexGothic",
    "TakaoGothic",
)


def _fontconfig_paths() -> list:
    try:
        result = subprocess.run(
            ["fc-list", ":lang=ja", "file"], capture_output=True, text=True, timeout=10
        )
    except Exception:  # Windowsなどfc-listが無い環境
        return []
    return [line.split(":")[0].strip() for line in result.stdout.splitlines() if line.strip()]


def register_matplotlib_cjk() -> bool:
    """matplotlibで日本語が表示できるようフォントを設定する。成否を返す。"""
    font_paths = _fontconfig_paths()
    for path in font_paths:
        try:
            fm.fontManager.addfont(path)
        except Exception:
            continue

    if font_paths:
        try:
            matplotlib.rcParams["font.family"] = fm.FontProperties(fname=font_paths[0]).get_name()
            return True
        except Exception:
            pass

    installed = {f.name for f in fm.fontManager.ttflist}
    for family in CJK_FAMILIES:
        if family in installed:
            matplotlib.rcParams["font.family"] = family
            return True
    return False


def find_cjk_font_path() -> str:
    """PILのImageFont.truetype()に渡せる日本語フォントのパス。無ければNone。"""
    for path in _fontconfig_paths():
        if Path(path).suffix.lower() in (".ttf", ".ttc", ".otf"):
            return path

    for family in CJK_FAMILIES:
        try:
            return fm.findfont(fm.FontProperties(family=family), fallback_to_default=False)
        except Exception:
            continue
    return None


def missing_font_hint() -> str:
    """フォントが見つからなかったときに出す導入方法の案内。"""
    if sys.platform.startswith("win"):
        return "Windowsなら通常「Yu Gothic」等が入っているはずです。"
    return (
        "`apt-get install -y fonts-noto-cjk`(Linux)や"
        "`brew install --cask font-noto-sans-cjk`(mac)で導入できます。"
    )
