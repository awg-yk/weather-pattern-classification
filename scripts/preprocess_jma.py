"""
JMA天気図PNGの前処理スクリプト。

以下を行う:
  1. 画像周囲の余白（PDFページの白い余白）を自動でクロップし、図郭（黒枠）に合わせる
  2. 右下の日時スタンプ枠（例: "2026.04.01.00UTC"）を白で塗りつぶす
     -> モデルが気圧パターンと無関係な文字を学習してしまうのを防ぐ

日時スタンプの位置は画像に対する相対座標(0.0〜1.0)で指定する。
scripts/collect_jma.py でダウンロードした画像で位置がずれる場合は
--stamp-box を調整すること。

使い方:
    python scripts/preprocess_jma.py --in-dir data/raw/jma/png --out-dir data/processed/jma
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

# 右下の日時スタンプのおおよその位置 (left, top, right, bottom) を画像サイズに対する割合で指定
DEFAULT_STAMP_BOX = (0.65, 0.92, 1.0, 1.0)


def autocrop_to_content(image: Image.Image, white_threshold: int = 250) -> Image.Image:
    """画像周囲の白い余白を除去し、図郭(枠)ぎりぎりまでクロップする。"""
    gray = ImageOps.grayscale(image)
    arr = np.array(gray)
    non_white_mask = arr < white_threshold
    if not non_white_mask.any():
        return image

    rows = np.where(non_white_mask.any(axis=1))[0]
    cols = np.where(non_white_mask.any(axis=0))[0]
    top, bottom = rows[0], rows[-1]
    left, right = cols[0], cols[-1]
    return image.crop((left, top, right + 1, bottom + 1))


def mask_stamp_box(image: Image.Image, stamp_box: tuple) -> Image.Image:
    """日時スタンプ部分を白く塗りつぶす。"""
    w, h = image.size
    left = int(stamp_box[0] * w)
    top = int(stamp_box[1] * h)
    right = int(stamp_box[2] * w)
    bottom = int(stamp_box[3] * h)

    arr = np.array(image)
    arr[top:bottom, left:right] = 255
    return Image.fromarray(arr)


def process_image(src_path: Path, dst_path: Path, stamp_box: tuple) -> None:
    image = Image.open(src_path).convert("RGB")
    image = autocrop_to_content(image)
    image = mask_stamp_box(image, stamp_box)
    image.save(dst_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--stamp-box",
        type=float,
        nargs=4,
        default=DEFAULT_STAMP_BOX,
        metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
        help="日時スタンプの位置(相対座標 0-1)。デフォルトは右下想定。",
    )
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*.png"))
    if not files:
        print(f"no PNG files found in {in_dir}")
        return

    for src_path in files:
        dst_path = out_dir / src_path.name
        process_image(src_path, dst_path, tuple(args.stamp_box))
        print(f"processed: {dst_path}")


if __name__ == "__main__":
    main()
