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

from src.chartsymbols import (
    DEFAULT_BANDS,
    FURNITURE_CV,
    FURNITURE_STABILITY,
    MaskAccumulator,
    band_overlap,
    band_variation,
    color_masks,
    dominant_colors,
    to_hsv,
)

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


def iter_images(in_dir: Path, limit: int):
    paths = sorted(p for p in in_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise SystemExit(f"画像が見つかりません: {in_dir}")
    return paths[:limit]


def report_image(path: Path, top: int, probe: tuple | None,
                 accumulator: MaskAccumulator | None = None) -> dict[str, int]:
    rgb = np.array(Image.open(path).convert("RGB"))
    print(f"\n=== {path.name}  {rgb.shape[1]}x{rgb.shape[0]} ===")

    print("  多い色(白い地色を除く):")
    for entry in dominant_colors(rgb, top=top):
        r, g, b = entry["rgb"]
        h, s, v = entry["hsv"]
        print(f"    RGB({r:3d},{g:3d},{b:3d})  HSV({h:3d},{s:3d},{v:3d})  "
              f"{entry['pixels']:8d}px  {entry['share']:6.2%}")

    masks = color_masks(rgb)
    if accumulator is not None:
        accumulator.add(masks)
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

    return {name: int(mask.sum()) for name, mask in masks.items()}


def report_variation(per_band: dict[str, list[int]], n_images: int,
                     accumulator: MaskAccumulator) -> None:
    """画像をまたいだ画素数の動きから、地図の備品を掴んでいる帯を指摘する。

    帯の重なりが0でも安心できない。片方の帯が空でも0になるからで、実際に
    最初の暫定値ではそれが起きていた。海岸線・経緯度線は毎回同じだけ描かれる
    ので画素数が動かない。前線は日によって有無すら変わる。
    """
    if n_images < 3:
        print("\n(画素数の変動を見るには3枚以上が要る。--limit を増やすこと)")
        return

    stability = accumulator.stability() if not accumulator.size_mismatch else {}
    if accumulator.size_mismatch:
        print("\n(画像の大きさが揃っていないので、同じ画素かどうかは測れなかった)")

    print(f"\n=== {n_images}枚での動き ===")
    print(f"{'帯':16s} {'最小':>9s} {'最大':>9s} {'変動係数':>9s} {'同じ場所':>9s} {'毎回点灯':>9s}  判定")
    stats = band_variation(per_band)
    flagged = []
    for name, st in stats.items():
        detail = stability.get(name)
        typical = detail["typical"] if detail else None
        if st["mean"] == 0:
            verdict = "空(この帯は何も拾っていない)"
        elif typical is not None and typical >= FURNITURE_STABILITY:
            verdict = "★毎回同じ場所 = 地図の備品"
            flagged.append(name)
        elif typical is not None:
            verdict = "場所が動く = 気象を掴んでいる"
        elif st["looks_like_furniture"]:
            verdict = "総量が動かない(場所は未測定)"
        else:
            verdict = "日によって変わる = 気象を掴んでいる"
        if detail:
            same_text = f"{typical:9.3f} {detail['always']:9.3f}"
        else:
            same_text = f"{'-':>9s} {'-':>9s}"
        print(f"{name:16s} {st['min']:9d} {st['max']:9d} {st['cv']:9.3f} {same_text}  {verdict}")

    front_flagged = [n for n in flagged if n.endswith("_front")]
    if front_flagged:
        print(f"\n前線の帯 {', '.join(front_flagged)} が毎回同じ画素を掴んでいる。")
        print("海岸線や経緯度線を前線として数えている可能性が高い。重ね描きで確かめること。")
    print(f"\n判定は「同じ場所」({FURNITURE_STABILITY}以上で備品)を使う。変動係数だけでは")
    print("等圧線も備品に見えてしまう(常に図全体を覆うので総量が動かない)。")
    print("「毎回点灯」は参考値。上書きで日ごとに違う所が隠れるため、備品でも1にならない")
    print("(実測の海岸線は 同じ場所1.000 に対し 毎回点灯0.533)。")


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

    per_band: dict[str, list[int]] = {name: [] for name in DEFAULT_BANDS}
    accumulator = MaskAccumulator()
    paths = iter_images(Path(args.in_dir), args.limit)
    for path in paths:
        counts = report_image(path, args.top,
                              tuple(args.probe) if args.probe else None, accumulator)
        for name, count in counts.items():
            per_band[name].append(count)

    report_variation(per_band, len(paths), accumulator)


if __name__ == "__main__":
    main()
