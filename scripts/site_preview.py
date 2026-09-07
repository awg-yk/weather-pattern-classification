r"""地点の円を天気図に重ねて描き、位置が合っているか目で確かめる。

**数字を出す前に必ず1回これを見ること。**地点の座標は緯度経度から変換せず
天気図に重ねて目で決める決まりなので(src/sites.py)、確認しないまま集計すると
「小松の周辺」と言いながら別の場所を測ることになる。ずれていたら
data/sites.csv の x, y, radius を直す。

使い方:
    python -m scripts.site_preview --site komatsu ^
        --image data\processed\all\Js_2023010100_page001.png ^
        --out reports\site_komatsu.png

    # 日付で引く(data/processed/all から探す)
    python -m scripts.site_preview --site komatsu --date 2023-01-01

    # 検出した高気圧・低気圧も一緒に描いて、絞り込みの効き方を見る
    python -m scripts.site_preview --site komatsu --date 2023-01-01 --detect
"""

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PIL import Image, ImageDraw

from src.sites import draw_site, get_site

# 描き分けの色。annotate_charts.py と揃えている
HIGH_COLOR = (0, 160, 0)
LOW_COLOR = (200, 0, 0)
FAR_COLOR = (170, 170, 170)


def _mark(draw, point, size, color, width=3):
    """検出した系の中心に印を打つ。"""
    x = point[0] * size[0]
    y = point[1] * size[1]
    arm = 14
    draw.line([x - arm, y - arm, x + arm, y + arm], fill=color, width=width)
    draw.line([x - arm, y + arm, x + arm, y - arm], fill=color, width=width)


def _draw_grid(image, step: float):
    """相対座標の目盛りを重ねる。

    **緯度経度からは変換しない。**この天気図は正距円筒図法ではないので、
    経緯線の目盛りから作った線形の式は図の中央でずれる(実際に東京を置いたら
    本州北部に載った)。地点を置くときは、この目盛りを見て地図の上で直接
    読み取り、data/sites.csv に書き写す。
    """
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    faint = (150, 150, 150)
    strong = (90, 90, 90)

    value = 0.0
    while value <= 1.0001:
        # 0.1刻みは濃く、その間は薄く。数えやすくするため
        major = abs(round(value / 0.1) * 0.1 - value) < 1e-9
        color = strong if major else faint
        x = int(round(value * width))
        y = int(round(value * height))
        draw.line([x, 0, x, height], fill=color, width=2 if major else 1)
        draw.line([0, y, width, y], fill=color, width=2 if major else 1)
        if major:
            draw.text((x + 4, 6), f"x={value:.1f}", fill=strong)
            draw.text((6, y + 4), f"y={value:.1f}", fill=strong)
        value += step
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", required=True, help="data/sites.csv の地点名")
    parser.add_argument("--image", default=None, help="重ねて見る天気図")
    parser.add_argument("--date", default=None, help="YYYY-MM-DD。--image の代わりに使う")
    parser.add_argument("--hour", type=int, default=0, choices=[0, 12])
    parser.add_argument("--sites", default=None, help="既定は data/sites.csv")
    parser.add_argument("--radius", type=float, default=None,
                        help="半径を一時的に変えて試す(CSVは書き換えない)")
    parser.add_argument("--detect", action="store_true",
                        help="高気圧・低気圧も検出して描く。円の内側は色付き、外側は灰色")
    parser.add_argument("--grid", action="store_true",
                        help="相対座標の目盛りを重ねる。地点の位置を読み取って"
                             "data/sites.csv に書き写すために使う")
    parser.add_argument("--grid-step", type=float, default=0.05,
                        help="目盛りの間隔(相対座標)")
    parser.add_argument("--out", default=None, help="既定は reports/site_<地点>.png")
    args = parser.parse_args()

    if not args.image and not args.date:
        parser.error("--image か --date のどちらかを指定してください")

    site = get_site(args.site, args.sites)
    if args.radius is not None:
        site = type(site)(name=site.name, x=site.x, y=site.y,
                          radius=args.radius, note=site.note)

    if args.date:
        from src.quicklook import chart_for
        image_path = chart_for(args.date, args.hour)
    else:
        image_path = Path(args.image)
    print(f"天気図: {image_path.name}")

    from scripts.preprocess_jma import (CANONICAL_SIZE, DEFAULT_STAMP_BOX,
                                        autocrop_to_content, fit_to_canonical,
                                        mask_stamp_box)

    image = Image.open(image_path).convert("RGB")
    # 前処理済みの画像なら、もう一度かけても結果は変わらない(実測で画素一致)
    image = mask_stamp_box(fit_to_canonical(autocrop_to_content(image), CANONICAL_SIZE),
                           DEFAULT_STAMP_BOX)

    if args.detect:
        import numpy as np

        from scripts.annotate_charts import annotate_one
        from src.quicklook import DETECTION, MARKS_DIR, TEMPLATES_DIR
        from src.sites import summarize

        marks = str(MARKS_DIR) if MARKS_DIR.exists() else None
        _, detections = annotate_one(
            np.array(image), str(TEMPLATES_DIR), marks,
            letter_size=DETECTION["letter_size"],
            threshold=DETECTION["detect_threshold"],
            boxes=False, fronts=False,
        )
        found = summarize(detections, site)
        draw = ImageDraw.Draw(image)
        for points, color in ((detections.highs, HIGH_COLOR), (detections.lows, LOW_COLOR)):
            for point in points:
                inside = site.contains(point[0], point[1])
                _mark(draw, point, image.size, color if inside else FAR_COLOR,
                      width=4 if inside else 2)
        print(f"検出: 全体で 高 {found['n_high_all']}個 / 低 {found['n_low_all']}個")
        print(f"      {site.name} の周辺(半径 {site.radius}) に "
              f"高 {found['n_high']}個 / 低 {found['n_low']}個")
        if found["nearest"]:
            near = found["nearest"]
            print(f"      最も近い系: {near['kind']} 距離 {near['distance']:.3f}")
        else:
            print("      周辺には1つも検出されていません")

    if args.grid:
        image = _draw_grid(image, args.grid_step)

    image = draw_site(image, site, text=site.name)

    out = Path(args.out) if args.out else _ROOT / "reports" / f"site_{site.name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)
    print(f"\n書き出しました: {out}")
    if args.grid:
        print(f"目盛りは {args.grid_step} 刻み(0.1ごとに濃い線)。"
              "地点の載っている交点を読んで、その値を data/sites.csv に書きます。")
    print("★円が本当にその地点を囲んでいるか目で確かめてください。")
    print("  ずれていたら data/sites.csv の x, y を直します"
          "(x は左→右、y は上→下、どちらも0〜1)。")


if __name__ == "__main__":
    main()
