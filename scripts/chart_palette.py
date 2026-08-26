"""天気図に実際に使われている色を測る。色マスクの閾値を決める前の第一歩。

`src/chartsymbols.py` の `DEFAULT_BANDS` は仕様(海岸線=赤茶色、等圧線=黒、
温暖前線=赤、寒冷前線=青、閉塞前線=ピンク)から起こした暫定値で、実測値では
ない。閾値をいじる前にこれを走らせ、本物の天気図に出る色とHSVを見ること。

見るところ:
  - 赤(温暖前線)と赤茶色(海岸線)が、HSVで離れているか。
    近ければ色だけで前線を切り出せないので、この方式は前線については見送り。
  - 手動アーカイブ(2000〜2022・国会図書館のスキャンJPEG)と気象庁PDF版で
    色がどれだけ違うか。--in-dir を分けて2回走らせて見比べる。

使い方:
    python -m scripts.chart_palette --in-dir data/processed/jma --limit 8
    python -m scripts.chart_palette --in-dir data/processed/jma --probe 0.5 0.5
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from src.chartsymbols import DEFAULT_BANDS, band_overlap, color_masks, dominant_colors, to_hsv

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


def iter_images(in_dir: Path, limit: int):
    paths = sorted(p for p in in_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise SystemExit(f"画像が見つかりません: {in_dir}")
    return paths[:limit]


def report_image(path: Path, top: int, probe: tuple | None) -> None:
    rgb = np.array(Image.open(path).convert("RGB"))
    print(f"\n=== {path.name}  {rgb.shape[1]}x{rgb.shape[0]} ===")

    print("  多い色(白い地色を除く):")
    for entry in dominant_colors(rgb, top=top):
        r, g, b = entry["rgb"]
        h, s, v = entry["hsv"]
        print(f"    RGB({r:3d},{g:3d},{b:3d})  HSV({h:3d},{s:3d},{v:3d})  "
              f"{entry['pixels']:8d}px  {entry['share']:6.2%}")

    masks = color_masks(rgb)
    total = rgb.shape[0] * rgb.shape[1]
    print("  暫定の帯に入った画素:")
    for name, mask in masks.items():
        count = int(mask.sum())
        print(f"    {name:16s} {count:8d}px  {count / total:6.2%}")

    overlap = band_overlap(masks)
    if overlap:
        print("  帯どうしの重なり(0でないなら閾値が甘い):")
        for (a, b), count in sorted(overlap.items(), key=lambda kv: -kv[1]):
            print(f"    {a:16s} x {b:16s} {count:8d}px")
    else:
        print("  帯どうしの重なり: なし")

    if probe is not None:
        hsv = to_hsv(rgb)
        x = int(probe[0] * rgb.shape[1])
        y = int(probe[1] * rgb.shape[0])
        print(f"  指定画素 ({probe[0]:.3f}, {probe[1]:.3f}) = 画素({x},{y}): "
              f"RGB{tuple(int(v) for v in rgb[y, x])} HSV{tuple(int(v) for v in hsv[y, x])}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--in-dir", required=True, help="天気図画像のディレクトリ")
    parser.add_argument("--limit", type=int, default=5, help="測る枚数")
    parser.add_argument("--top", type=int, default=12, help="1枚あたり何色まで出すか")
    parser.add_argument("--probe", type=float, nargs=2, metavar=("X", "Y"),
                        help="相対座標(0〜1)の1画素の色を出す。線の上を指して色を確かめる用")
    args = parser.parse_args()

    print("暫定の帯 (src/chartsymbols.py DEFAULT_BANDS):")
    for name, band in DEFAULT_BANDS.items():
        print(f"  {name:16s} H[{band.h_min:3d}-{band.h_max:3d}] "
              f"S[{band.s_min:3d}-{band.s_max:3d}] V[{band.v_min:3d}-{band.v_max:3d}]")

    for path in iter_images(Path(args.in_dir), args.limit):
        report_image(path, args.top, tuple(args.probe) if args.probe else None)


if __name__ == "__main__":
    main()
