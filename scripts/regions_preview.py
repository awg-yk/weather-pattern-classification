"""data/regions.csv の矩形を天気図に重ねて描き、目で確認できるようにする。

矩形は相対座標で持っているだけなので(src/regions.py)、それが本当に
オホーツク海や日本の南岸を指しているかは、実際の天気図に重ねないと分からない。
数値を測り始める前に必ず1回これを見て、ずれていたら data/regions.csv の
x0,y0,x1,y1 を直す。

使い方:
    # 全ラベルを1枚ずつ並べて確認する
    python -m scripts.regions_preview --image ..\\weather-pattern-classification-data\\processed\\Js_2025050100.png \\
        --out reports\\regions_preview.png

    # 1ラベルだけ、天気図と同じ解像度で書き出す
    python -m scripts.regions_preview --image <天気図> --labels okhotsk_high \\
        --out reports\\okhotsk_region.png
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image

from scripts.preprocess_jma import DEFAULT_STAMP_BOX, autocrop_to_content, mask_stamp_box
from src.jp_font import missing_font_hint, register_matplotlib_cjk
from src.labels import LABELS, LABEL_JA
from src.regions import load_regions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="重ねて見る天気図")
    parser.add_argument("--out", default="reports/regions_preview.png")
    parser.add_argument("--regions", default=None, help="既定は data/regions.csv")
    parser.add_argument("--labels", nargs="+", default=None, choices=list(LABELS),
                        help="このラベルだけ描く。省略すると全ラベルを並べる")
    parser.add_argument("--raw", action="store_true",
                        help="前処理前のJMA画像(枠・スタンプ付き)を渡すときに指定する")
    parser.add_argument("--columns", type=int, default=5, help="全ラベルを並べるときの列数")
    args = parser.parse_args()

    if not register_matplotlib_cjk():
        print(f"警告: 日本語フォントが見つかりませんでした。{missing_font_hint()}")

    regions = load_regions(args.regions)
    wanted = args.labels or [label for label in LABELS if label in regions]
    unknown = [label for label in wanted if label not in regions]
    if unknown:
        raise SystemExit(f"領域が定義されていないラベルです: {unknown}")

    image = Image.open(args.image).convert("RGB")
    if args.raw:
        image = autocrop_to_content(image)
        image = mask_stamp_box(image, DEFAULT_STAMP_BOX)

    columns = min(args.columns, len(wanted))
    rows = (len(wanted) + columns - 1) // columns
    width, height = image.size
    panel = 3.2
    fig, axes = plt.subplots(
        rows, columns, figsize=(panel * columns, panel * rows * height / width), squeeze=False
    )

    for ax, label in zip(axes.ravel(), wanted):
        region = regions[label]
        ax.imshow(image)
        left, top, right, bottom = region.pixel_box(width, height)
        ax.add_patch(
            patches.Rectangle(
                (left, top), right - left, bottom - top,
                linewidth=2.0, edgecolor="red", facecolor="red", alpha=0.12,
            )
        )
        ax.set_title(f"{LABEL_JA[label]}(面積 {region.area:.0%})", fontsize=10)
        ax.axis("off")

    for ax in axes.ravel()[len(wanted):]:
        ax.axis("off")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"書き出しました: {out_path}")
    print("矩形が実際の海域とずれていたら data/regions.csv の x0,y0,x1,y1 を直してください。")


if __name__ == "__main__":
    main()
