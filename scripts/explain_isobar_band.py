r"""資料用に、色帯で「等圧線だけ」を取り出す様子を1枚の比較画像にする。

なぜ要るか
----------
テンプレートマッチングの説明資料で「②では前線や海岸線は除去されている」と
言葉で説明しても伝わりにくい。**実際の天気図で、色帯を通す前後を並べて
見せたほうが早い。**

出す画像は横に3枚並び:

  ① 元の天気図
  ② 等圧線の色帯(彩度60以下・明度90以下)だけを2値化した画像。
     `src.chartsymbols.match_templates` に実際に渡っているのはこれ
  ③ ①に、前線(赤・青・紫)と海岸線(赤茶)の色帯に入る画素を塗って重ねた図。
     ②に写っていない画素がどこから来ているかが分かる

使い方
------
    python -m scripts.explain_isobar_band --image data\processed\all\Js_2023010100_page001.png

    --out reports\isobar_band_explained.png   書き出し先(既定はこの場所)

コンソールには、等圧線・前線・海岸線それぞれの画素数の割合と、
**大津の方法に切り替わったかどうか**(切り替わっていれば②に前線・海岸線が
混じる)を出す。
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.chartsymbols import (DEFAULT_BANDS, FRONT_BANDS, color_masks, ink_mask,
                             to_hsv)
from src.jp_font import find_cjk_font_path, missing_font_hint

# 前線・海岸線をどの色で塗って見せるか(RGB)。天気図側の実際の色とは
# 揃えていない ― 実物の赤・青は地味で目立たないので、蛍光色にして目を引かせる
HIGHLIGHT = {
    "warm_front": (255, 60, 200),      # 蛍光ピンク
    "cold_front": (0, 200, 255),       # 水色
    "occluded_front": (180, 60, 255),  # 紫
    "coastline": (255, 160, 0),        # オレンジ
}
LABELS_JA = {
    "warm_front": "温暖前線",
    "cold_front": "寒冷前線",
    "occluded_front": "閉塞前線",
    "coastline": "海岸線",
}

CAPTIONS = ["1. 元の天気図", "2. 等圧線の色帯だけを2値化\n(実際にマッチングへ渡す画像)",
           "3. 除外された画素に色を付けた図\n(等圧線以外は色帯で弾かれている)"]
CAPTION_HEIGHT = 70


def make_binary_panel(mask: np.ndarray) -> np.ndarray:
    """真偽マスクを、白地に黒の見た目に変える(実際の入力と同じ表現)。"""
    panel = np.full((*mask.shape, 3), 255, np.uint8)
    panel[mask] = (0, 0, 0)
    return panel


def make_highlight_panel(rgb: np.ndarray, masks: dict) -> np.ndarray:
    """元画像に、前線・海岸線の画素だけ目立つ色で塗った図を作る。"""
    panel = rgb.copy()
    for name in (*FRONT_BANDS, "coastline"):
        panel[masks[name]] = HIGHLIGHT[name]
    return panel


def add_caption(panel: np.ndarray, text: str, font_path: str) -> Image.Image:
    """パネルの下に見出しを書き足す。**見出しがはみ出すぶんだけ幅を広げる**

    (画像そのものは縮小済みで幅が小さいので、2行の見出しがそのままでは
    右にはみ出す)。
    """
    image = Image.fromarray(panel)
    img_width, height = image.size
    font = None
    if font_path:
        from PIL import ImageFont
        font = ImageFont.truetype(font_path, 22)

    probe = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(probe)
    box = draw.multiline_textbbox((0, 0), text, font=font, spacing=6)
    text_width = box[2] - box[0]

    width = max(img_width, text_width + 24)
    canvas = Image.new("RGB", (width, height + CAPTION_HEIGHT), (255, 255, 255))
    canvas.paste(image, ((width - img_width) // 2, 0))
    draw = ImageDraw.Draw(canvas)
    draw.multiline_text((12, height + 10), text, fill=(0, 0, 0), font=font, spacing=6)
    return canvas


def build(image_path: Path, out_path: Path, max_width: int = 700) -> dict:
    rgb = np.array(Image.open(image_path).convert("RGB"))
    height, width = rgb.shape[:2]

    ink, fell_back = ink_mask(rgb, band="isobar")
    masks = color_masks(rgb)

    binary = make_binary_panel(ink)
    highlighted = make_highlight_panel(rgb, masks)

    font_path = find_cjk_font_path()
    if font_path is None:
        print(f"警告: 日本語フォントが見つかりません。{missing_font_hint()}")
        print("      見出しの日本語が文字化けする可能性があります。")

    panels = []
    for arr, caption in zip((rgb, binary, highlighted), CAPTIONS):
        image = Image.fromarray(arr)
        if max_width and image.width > max_width:
            scale = max_width / image.width
            image = image.resize((max_width, round(image.height * scale)), Image.LANCZOS)
        panels.append(add_caption(np.array(image), caption, font_path))

    total_width = sum(p.width for p in panels) + 20 * (len(panels) - 1)
    combined = Image.new("RGB", (total_width, panels[0].height), (255, 255, 255))
    x = 0
    for p in panels:
        combined.paste(p, (x, 0))
        x += p.width + 20
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.save(out_path)

    total = height * width
    stats = {
        "isobar_share": float(ink.mean()),
        "fell_back_to_otsu": fell_back,
        **{name: float(masks[name].mean()) for name in masks},
    }
    return stats


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", type=Path, required=True, help="前処理済みの天気図1枚")
    parser.add_argument("--out", type=Path,
                        default=_ROOT / "reports" / "isobar_band_explained.png")
    parser.add_argument("--max-width", type=int, default=700,
                        help="1枚あたりの表示幅。0で縮小しない")
    args = parser.parse_args(argv)

    if not args.image.exists():
        raise SystemExit(f"{args.image} が見つかりません")

    stats = build(args.image, args.out, args.max_width)

    print(f"{args.image.name} を処理しました")
    print(f"  等圧線の色帯に入った画素の割合: {stats['isobar_share']:.2%}")
    if stats["fell_back_to_otsu"]:
        print("  ★色帯がほとんど空だったため、濃さだけで判定する方式に切り替わりました。")
        print("    この天気図では前線・海岸線も等圧線と一緒に拾われています。")
        print("    別の(色帯が読める)天気図で作り直すことを勧めます。")
    else:
        print("  色帯は正常に読めています(切り替わっていません)。")
    for name in FRONT_BANDS:
        print(f"  {LABELS_JA[name]}の色帯: {stats[name]:.2%}"
              f"{'  ※2の画像には入っていません' if not stats['fell_back_to_otsu'] else ''}")
    print(f"  {LABELS_JA['coastline']}の色帯: {stats['coastline']:.2%}"
          f"{'  ※2の画像には入っていません' if not stats['fell_back_to_otsu'] else ''}")
    print(f"\n{args.out} に書きました")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
