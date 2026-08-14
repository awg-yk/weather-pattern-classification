"""
PNG画像フォルダをまとめてJPEGに変換するスクリプト。

天気図は白背景+線画中心のため、JPEG(quality=92程度)でもほぼ画質を落とさずに
ファイルサイズを大幅に削減できる(体感でPNGの1/3〜1/5程度)。

使い方:
    python scripts/convert_to_jpeg.py --in-dir data/raw/ndl_manual/png --out-dir data/raw/ndl_manual/jpg
    # 変換元のPNGを削除する場合は --delete-source を付ける
"""

import argparse
from pathlib import Path

from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", required=True, help="変換元PNGフォルダ")
    parser.add_argument("--out-dir", required=True, help="変換先JPEGフォルダ")
    parser.add_argument("--quality", type=int, default=92, help="JPEG品質(0-100)")
    parser.add_argument(
        "--delete-source", action="store_true", help="変換後に元のPNGファイルを削除する"
    )
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*.png"))
    if not files:
        print(f"no PNG files found in {in_dir}")
        return

    total_before = 0
    total_after = 0
    for src_path in files:
        dst_path = out_dir / (src_path.stem + ".jpg")
        image = Image.open(src_path).convert("RGB")
        image.save(dst_path, "JPEG", quality=args.quality)

        total_before += src_path.stat().st_size
        total_after += dst_path.stat().st_size
        if args.delete_source:
            src_path.unlink()
        print(f"converted: {dst_path}")

    print(
        f"\n合計: {len(files)}枚, "
        f"{total_before / 1_000_000:.1f}MB -> {total_after / 1_000_000:.1f}MB "
        f"({total_after / total_before * 100:.0f}%)"
    )


if __name__ == "__main__":
    main()
