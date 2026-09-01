r"""手元だけで作業できる状態かを点検する。

このリポジトリのフォルダの中だけで、分類・検出・学習ができるかを確かめる。
足りないものがあれば、**それを用意するコマンドまで出す。**

    python -m scripts.check_local_setup

置き場所の決まり(すべてこのフォルダの中):

    data/raw/jma_add/png/       2023年以降の生の天気図(PDFから変換したPNG)
    data/raw/ndl_png/           2000〜2022年の生の天気図
    data/processed/jma/         2023年以降の前処理後
    data/processed/ndl/         2000〜2022年の前処理後
    data/labels_v2.csv          2023年以降のラベル(2432件)
    data/templates/             H/L のテンプレートと reference.json
    weights/                    学習済みの重みと校正ファイル

data/raw/ と data/processed/ はGitの追跡対象外なので、容量の心配はない。
"""

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

RAW_JMA = _ROOT / "data" / "raw" / "jma_add" / "png"
RAW_NDL = _ROOT / "data" / "raw" / "ndl_png"
PROC_JMA = _ROOT / "data" / "processed" / "jma"
PROC_NDL = _ROOT / "data" / "processed" / "ndl"


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

    print(f"リポジトリ: {_ROOT}\n")
    problems = 0

    print("== 重み ==")
    problems += not check_file(_ROOT / "weights" / "model.pt", "素の天気図用")
    problems += not check_file(_ROOT / "weights" / "model.calib.json", "その校正")
    problems += not check_file(_ROOT / "weights" / "model_annot.pt", "注釈方式用")
    problems += not check_file(_ROOT / "weights" / "model_annot.calib.json", "その校正")

    print("\n== 検出のテンプレート ==")
    n = check_images(_ROOT / "data" / "templates", "H/L のテンプレート")
    problems += n == 0
    problems += not check_file(
        _ROOT / "data" / "templates" / "reference.json", "基準の幅",
        "python -m scripts.set_template_reference --image data\\raw\\jma_add\\png\\<2023年の1枚>")

    print("\n== ラベル ==")
    labels = Path(args.labels)
    if check_file(labels, "ラベル"):
        import pandas as pd
        df = pd.read_csv(labels, dtype={"date": str})
        print(f"       {len(df)}件  {df['date'].min()} 〜 {df['date'].max()}")
    else:
        problems += 1

    print("\n== 天気図 ==")
    raw_jma = check_images(RAW_JMA, "2023年以降(生)")
    check_images(RAW_NDL, "2000〜2022年(生)")
    proc_jma = check_images(
        PROC_JMA, "2023年以降(前処理後)",
        f"python -m scripts.preprocess_jma --in-dir {RAW_JMA.relative_to(_ROOT)} "
        f"--out-dir {PROC_JMA.relative_to(_ROOT)}")
    check_images(
        PROC_NDL, "2000〜2022年(前処理後)",
        f"python -m scripts.preprocess_jma --in-dir {RAW_NDL.relative_to(_ROOT)} "
        f"--out-dir {PROC_NDL.relative_to(_ROOT)}")

    if raw_jma and not proc_jma:
        problems += 1
    if not raw_jma and not proc_jma:
        print("       ★2023年以降の天気図がありません。学習も評価もできません")
        problems += 1

    # ラベルの行が、前処理後のフォルダで実際に開けるか
    if labels.exists() and proc_jma:
        import pandas as pd
        names = {p.name for p in PROC_JMA.iterdir()}
        df = pd.read_csv(labels)
        found = sum(1 for n in df["filename"] if n in names)
        print(f"\n  ラベル{len(df)}件のうち、{PROC_JMA.relative_to(_ROOT)} で "
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
