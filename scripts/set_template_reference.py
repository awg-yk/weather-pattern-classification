r"""テンプレートを切り出した天気図の幅を記録する。

これを1回だけ実行しておくと、以降 `--letter-size auto` が使えるようになり、
解像度の違う天気図(国立国会図書館由来の2000〜2022年など)でも高低気圧を
検出できるようになる。

使い方:
    python -m scripts.set_template_reference --image <テンプレートを切り出した天気図>

渡すのは **2023年以降の、data/templates の H/L を切り出した天気図**。
前処理(余白の切り取り)を済ませたものなら --no-preprocess を付ける。
"""

import argparse
import sys
from pathlib import Path

from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.preprocess_jma import (DEFAULT_STAMP_BOX, autocrop_to_content,
                                    mask_stamp_box)
from src.chartscale import load_reference, save_reference

DEFAULT_TEMPLATES = _ROOT / "data" / "templates"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True,
                        help="テンプレートを切り出した天気図(2023年以降)")
    parser.add_argument("--templates", default=str(DEFAULT_TEMPLATES))
    parser.add_argument("--no-preprocess", action="store_true",
                        help="既に余白の切り取りが済んでいる画像を渡す場合")
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    raw = image.size
    if not args.no_preprocess:
        image = mask_stamp_box(autocrop_to_content(image), DEFAULT_STAMP_BOX)
    width = image.size[0]

    old = load_reference(args.templates)
    if old:
        print(f"今の基準: {old['chart_width']}px({old.get('source', '出所不明')})")

    path = save_reference(args.templates, width, source=Path(args.image).name)
    print(f"元の大きさ: {raw[0]} x {raw[1]}"
          + ("" if args.no_preprocess else f"  前処理後: {image.size[0]} x {image.size[1]}"))
    print(f"基準の幅 {width}px を記録しました: {path}")
    print("これで --letter-size auto が使えます。")
    print("**渡した天気図が data/templates の切り出し元と違うと、"
          "倍率がずれて検出が悪くなります。**")


if __name__ == "__main__":
    main()
