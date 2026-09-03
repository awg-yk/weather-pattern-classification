r"""手元だけで作業できる状態かを点検する。

このリポジトリのフォルダの中だけで、分類・検出・学習ができるかを確かめる。
足りないものがあれば、**それを用意するコマンドまで出す。**

    python -m scripts.check_local_setup

置き場所の決まり(すべてこのフォルダの中):

    data/raw/new_png/           生の天気図(2000〜2025年)
    data/processed/all/         前処理後。**時代で分けない**
    data/processed/all_annot/   検出結果を描き込んだもの(学習用)
    data/labels_v2.csv          ラベル
    data/templates/             H/L のテンプレート
    weights/                    学習済みの重みと校正ファイル

前処理がすべての天気図を 1453x1500 に揃えるので、取得元や時代で
フォルダを分ける必要はない。詳しくは
docs/2026-09-02-rebuild-and-final-numbers.md を参照。

data/raw/ と data/processed/ はGitの追跡対象外なので、容量の心配はない。
"""

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

RAW = _ROOT / "data" / "raw" / "new_png"
PROC = _ROOT / "data" / "processed" / "all"
PROC_ANNOT = _ROOT / "data" / "processed" / "all_annot"

# 学習と評価に使う年。2026年は区切りが悪いので入れない
YEARS = (2023, 2024, 2025)


def count_images(directory: Path) -> int:
    if not directory.is_dir():
        return 0
    return sum(1 for p in directory.iterdir()
               if p.suffix.lower() in (".png", ".jpg", ".jpeg"))


def check_file(path: Path, what: str, fix: str = "") -> bool:
    if path.exists():
        size = path.stat().st_size
        unit = f"{size / 1e6:.1f}MB" if size > 1e6 else f"{size / 1e3:.0f}KB"
        print(f"  [OK] {what}: {path.relative_to(_ROOT)} ({unit})")
        return True
    print(f"  [--] {what}: 見つかりません ({path.relative_to(_ROOT)})")
    if fix:
        print(f"       {fix}")
    return False


def check_images(directory: Path, what: str, fix: str = "") -> int:
    n = count_images(directory)
    if n:
        print(f"  [OK] {what}: {n}枚 ({directory.relative_to(_ROOT)})")
    else:
        print(f"  [--] {what}: 0枚 ({directory.relative_to(_ROOT)})")
        if fix:
            print(f"       {fix}")
    return n


def check_packages() -> list:
    needed = {"torch": "学習と推論", "torchvision": "同上", "cv2": "検出(opencv)",
              "matplotlib": "図の表示", "pandas": "ラベルの読み込み",
              "PIL": "画像の読み書き", "scipy": "前処理の連結成分"}
    missing = []
    for name, why in needed.items():
        try:
            __import__(name)
        except ImportError:
            missing.append(f"{name}({why})")
    return missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default=str(_ROOT / "data" / "labels_v2.csv"))
    args = parser.parse_args()

    # **どのPythonで動いているかを出す。**ノートブックのカーネルが別の
    # Pythonだと、ここで「揃っている」と出てもノートブックでは足りない、
    # という分かりにくい食い違いが起きる
    print(f"リポジトリ: {_ROOT}")
    print(f"Python    : {sys.executable}\n")
    problems = 0

    print("== 重み ==")
    problems += not check_file(_ROOT / "weights" / "model.pt", "素の天気図用")
    problems += not check_file(_ROOT / "weights" / "model.calib.json", "その校正")
    problems += not check_file(_ROOT / "weights" / "model_annot.pt", "注釈方式用")
    problems += not check_file(_ROOT / "weights" / "model_annot.calib.json", "その校正")

    print("\n== 検出のテンプレート ==")
    n = check_images(_ROOT / "data" / "templates", "H/L のテンプレート")
    problems += n == 0

    print("\n== ラベル ==")
    labels = Path(args.labels)
    if check_file(labels, "ラベル"):
        import pandas as pd
        df = pd.read_csv(labels, dtype={"date": str})
        print(f"       {len(df)}件  {df['date'].min()} 〜 {df['date'].max()}")
    else:
        problems += 1

    print("\n== 天気図 ==")
    check_images(RAW, "生(2000〜2025年)")
    proc = check_images(
        PROC, "前処理後",
        f"python -m scripts.preprocess_jma --in-dir {RAW.relative_to(_ROOT)} "
        f"--out-dir {PROC.relative_to(_ROOT)}")
    years = " ".join(str(y) for y in YEARS)
    check_images(
        PROC_ANNOT, "描き込み後(学習用)",
        f"python -m scripts.annotate_charts --in-dir {PROC.relative_to(_ROOT)} "
        f"--out-dir {PROC_ANNOT.relative_to(_ROOT)} --years {years} --no-fronts")

    if not proc:
        print("       ★前処理後の天気図がありません。分類も学習もできません")
        problems += 1

    # ラベルの行が、前処理後のフォルダで実際に開けるか。
    # ファイル名の表記が取得元で違う(Js_2023010100.png と
    # Js_2023010100_page001.png)ので、名前ではなく10桁の日時で照合する
    if labels.exists() and proc:
        import pandas as pd

        from src.split import DATE_IN_FILENAME, index_images_by_stamp

        index = index_images_by_stamp(PROC)
        df = pd.read_csv(labels)
        stamps = (DATE_IN_FILENAME.search(str(n)) for n in df["filename"])
        found = sum(1 for m in stamps if m and m.group(0) in index)
        print(f"\n  ラベル{len(df)}件のうち、{PROC.relative_to(_ROOT)} で "
              f"見つかるのは {found}件")
        if found < len(df):
            print("       ★足りない分は学習に使えません。変換し直すか、"
                  "ファイル名の付け方を確かめてください")
            problems += 1

    print("\n== Pythonの道具 ==")
    missing = check_packages()
    if missing:
        print("  [--] 足りません: " + ", ".join(missing))
        print("       pip install -r requirements.txt")
        problems += 1
    else:
        print("  [OK] 揃っています")

    print()
    if problems:
        print(f"★{problems}件、足りないものがあります。上の指示に従ってください。")
        raise SystemExit(1)
    print("すべて揃っています。このフォルダの中だけで作業できます。")


if __name__ == "__main__":
    main()
