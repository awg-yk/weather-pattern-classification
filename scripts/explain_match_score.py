r"""資料用に、一致度(スコア)を数式抜きで見せるための比較画像を作る。

なぜ要るか
----------
`cv2.matchTemplate(cv2.TM_CCOEFF_NORMED)` の数式は聴衆に伝わりにくい。
**「型紙(テンプレート)を正しい位置に重ねたとき」と「ずらした位置・別の
場所に重ねたとき」の見た目とスコアを並べたほうが、数式より早く伝わる。**

出す画像は横に3枚並び、それぞれ「テンプレートを重ねた場所」と、その場所の
一致度を見出しに書く:

  ① 正しい位置(記号そのものの上)      -> 高いスコア
  ② 何でもない模様の上(等圧線の交差点) -> 中くらいのスコア
  ③ 何もない白地の上                   -> 低いスコア

使い方
------
    python -m scripts.explain_match_score --image data\processed\all\Js_2023010100_page001.png

天気図を渡さない場合は、テンプレート自身で作った簡単な合成天気図を使う
(手元に前処理済みの天気図が無いときの下見用)。
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.build_features import load_templates_scaled
from src.chartsymbols import ink_mask
from src.jp_font import find_cjk_font_path, missing_font_hint

CAPTION_HEIGHT = 90
BOX_COLOR = (255, 60, 60)


def synthetic_chart(template: np.ndarray, symbol_at=(150, 100),
                    tilted_at=(340, 230), tilt_degrees=25.0) -> np.ndarray:
    """手元に天気図が無いときの下見用。等圧線と記号を2つ描く。

    2つ目は角度を変えて置く。**同じ記号でも傾いていれば一致度は下がる**
    ―「そこそこ似ているが完全ではない」の実例として使う。

    `symbol_at` `tilted_at` は (x0, y0) ― 記号の左上を置く位置。
    """
    from src.chartsymbols import rotate_template

    width, height = 620, 420
    rgb = np.full((height, width, 3), 255, np.uint8)
    for i in range(7):
        pts = np.array([[x, 60 * i + int(18 * np.sin(x / 55.0))]
                        for x in range(0, width, 4)])
        cv2.polylines(rgb, [pts], False, (10, 10, 10), 2)

    def paste(mask, x0, y0):
        ys, xs = np.nonzero(mask)
        rgb[ys + y0, xs + x0] = 0

    paste(template, *symbol_at)
    paste(rotate_template(template, tilt_degrees), *tilted_at)
    return rgb


def score_at(ink: np.ndarray, template: np.ndarray, near: tuple,
            search: int = 0) -> tuple:
    """`near` の周り(±search画素)でいちばん一致度が高い位置を探して返す。

    `search=0` なら `near` そのものの位置だけを見る(正しい位置・白地など、
    厳密にそこを見せたいとき)。傾いた記号のように、置いた場所と
    テンプレートの左上がぴったり重ならない場合は `search` を渡し、
    近くのいちばん良い場所を採用する。
    """
    h, w = template.shape
    response = cv2.matchTemplate(ink, template.astype(np.float32),
                                 cv2.TM_CCOEFF_NORMED)
    x0, y0 = near
    x0 = max(0, min(x0, ink.shape[1] - w))
    y0 = max(0, min(y0, ink.shape[0] - h))
    if search:
        rx0, ry0 = max(0, x0 - search), max(0, y0 - search)
        rx1 = min(response.shape[1], x0 + search + 1)
        ry1 = min(response.shape[0], y0 + search + 1)
        window = response[ry0:ry1, rx0:rx1]
        dy, dx = np.unravel_index(np.argmax(window), window.shape)
        x0, y0 = rx0 + dx, ry0 + dy
    score = float(response[y0, x0])
    return score, (x0, y0, x0 + w, y0 + h)


def draw_panel(rgb: np.ndarray, box: tuple) -> np.ndarray:
    panel = rgb.copy()
    cv2.rectangle(panel, box[:2], box[2:], BOX_COLOR, 3)
    return panel


def add_caption(panel: np.ndarray, text: str, font_path: str) -> Image.Image:
    image = Image.fromarray(panel)
    img_width, height = image.size
    font = None
    if font_path:
        from PIL import ImageFont
        font = ImageFont.truetype(font_path, 24)
    probe = Image.new("RGB", (1, 1))
    box = ImageDraw.Draw(probe).multiline_textbbox((0, 0), text, font=font, spacing=6)
    width = max(img_width, box[2] - box[0] + 24)
    canvas = Image.new("RGB", (width, height + CAPTION_HEIGHT), (255, 255, 255))
    canvas.paste(image, ((width - img_width) // 2, 0))
    ImageDraw.Draw(canvas).multiline_text((12, height + 10), text, fill=(0, 0, 0),
                                          font=font, spacing=6)
    return canvas


def build(rgb: np.ndarray, template: np.ndarray, out_path: Path,
         symbol_box: tuple, max_width: int = 260) -> list:
    mask, _ = ink_mask(rgb)
    ink = mask.astype(np.float32)   # matchTemplate は0/1の2値をそのまま渡せる
    h, w = template.shape

    spots = [
        ("正しい位置\n(記号そのもの)", (symbol_box[0], symbol_box[1]), 0),
        ("同じ記号だが傾いている\n(形は似ているが完全ではない)",
         (symbol_box[0] + w + 130, symbol_box[1] + 40), max(h, w) // 2),
        ("何も無い白地の上", (20, max(rgb.shape[0] - h - 20, 0)), 0),
    ]

    font_path = find_cjk_font_path()
    if font_path is None:
        print(f"警告: 日本語フォントが見つかりません。{missing_font_hint()}")

    panels, scores = [], []
    for label, top_left, search in spots:
        score, box = score_at(ink, template, top_left, search)
        scores.append(score)
        marked = draw_panel(rgb, box)
        image = Image.fromarray(marked)
        if max_width and image.width > max_width:
            scale = max_width / image.width
            image = image.resize((max_width, round(image.height * scale)), Image.LANCZOS)
        caption = f"{label}\n一致度 {score:.2f}"
        panels.append(add_caption(np.array(image), caption, font_path))

    total_width = sum(p.width for p in panels) + 20 * (len(panels) - 1)
    combined = Image.new("RGB", (total_width, panels[0].height), (255, 255, 255))
    x = 0
    for p in panels:
        combined.paste(p, (x, 0))
        x += p.width + 20
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.save(out_path)
    return scores


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", type=Path, default=None,
                        help="前処理済みの天気図1枚。省略すると合成天気図で作る")
    parser.add_argument("--symbol", default="H", help="使う記号(data/templates 内の名前)")
    parser.add_argument("--out", type=Path,
                        default=_ROOT / "reports" / "match_score_explained.png")
    args = parser.parse_args(argv)

    templates = load_templates_scaled(_ROOT / "data" / "templates", 1.0, quiet=True)
    if args.symbol not in templates:
        raise SystemExit(f"{args.symbol} が data/templates にありません: "
                         f"{sorted(templates)}")
    template = templates[args.symbol]

    if args.image:
        rgb = np.array(Image.open(args.image).convert("RGB"))
        symbol_box = (rgb.shape[1] // 2, rgb.shape[0] // 3)
    else:
        print("--image が無いので、合成した天気図で作ります(下見用)。")
        symbol_box = (150, 100)
        tilted_box = (340, 230)
        rgb = synthetic_chart(template, symbol_at=symbol_box, tilted_at=tilted_box)

    scores = build(rgb, template, args.out, symbol_box)
    print(f"正しい位置    : 一致度 {scores[0]:.2f}")
    print(f"傾いた同じ記号: 一致度 {scores[1]:.2f}")
    print(f"白地の上      : 一致度 {scores[2]:.2f}")
    print(f"\n{args.out} に書きました")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
