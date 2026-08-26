"""天気図が日付をまたいで画素単位で揃っているかを測る。

なぜ要るか
----------
`src/regions.py` の「見るべき領域」も、物体検出で取る中心座標も、
**全画像が同じ基準でトリミングされている**ことを前提にしている
(`src/regions.py` の座標系の説明)。前提が崩れていれば、
`data/regions.csv` の矩形も Phase 3 の位置の特徴量も、ずれた分だけ嘘になる。

海岸線・経緯度線は毎日まったく同じ地図なので、揃っていれば ずれは0になる。
0でなければ `scripts/preprocess_jma.py` の autocrop_to_content() が
日ごとに違う位置で切っているということである。

きっかけ
--------
2024年7月の12枚で、海岸線の「毎回同じ画素の割合」が 0.533 しかなかった
(`scripts/chart_palette.py`)。毎日同じ地図なのに半分弱しか重ならない。
さらに2023年1月が1453x1500、2024年7月が1453x1499 と、同じ配信シリーズなのに
高さが1画素違っていた。

読み方
------
    ずれの幅が0〜1画素      揃っている。座標をそのまま使ってよい
    数画素                  細い線の一致は崩れるが、矩形の判定には影響しにくい
    十数画素以上            `data/regions.csv` の矩形を測り直す必要がある

使い方:
    python -m scripts.check_alignment --in-dir data/processed/jma --limit 30
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from src.chartsymbols import DEFAULT_BANDS, color_masks, dominant_colors, estimate_shift

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")

# 位相相関の確からしさがこれ未満なら、平行移動では説明できないとみなす。
MIN_RESPONSE = 0.05


def explain_stability(images: list, band: str) -> None:
    """帯の画素が画像ごとに入れ替わる原因を、点灯回数の分布と色で調べる。

    位置が揃っているのに「毎回同じ画素の割合」が低いなら、ずれ以外の理由が
    ある。考えられるのは2つで、どちらかは分布を見れば分かる。

      縁のちらつき  線の縁の中間色(なめらかに描くための色)が、帯の境目を
                    またいで出たり入ったりする。**n回中n-1回**のような
                    高い回数に山ができ、色は中間色になる。無害。
      上書き        前線や等圧線がその上に描かれて隠す。隠れ方は日ごとに
                    違うので回数はばらけ、色は芯の色のまま。
    """
    n = len(images)
    counts = None
    for _, rgb in images:
        mask = color_masks(rgb)[band]
        counts = mask.astype(np.int32) if counts is None else counts + mask
    ever = int(np.count_nonzero(counts))
    always = int(np.count_nonzero(counts == n))

    print(f"\n=== {band} の画素が入れ替わる理由 ({n}枚) ===")
    median_k = float(np.median(counts[counts >= 1]))
    print(f"一度でも点灯: {ever:8d}px   毎回点灯: {always:8d}px   割合 {always / ever:.3f}")
    print(f"点灯回数の中央値: {median_k:.0f}/{n}  -> 同じ場所 {median_k / n:.3f}")
    print(f"{'点灯回数':>10s} {'画素数':>10s}  割合")
    histogram = np.bincount(counts.ravel(), minlength=n + 1)
    for k in range(1, n + 1):
        if histogram[k]:
            share = histogram[k] / ever
            bar = "#" * int(share * 40)
            print(f"{k:10d} {histogram[k]:10d}  {share:6.1%} {bar}")

    unstable = (counts >= 1) & (counts < n)
    if unstable.any():
        _, sample = images[0]
        colors = sample[unstable & (color_masks(sample)[band])]
        if colors.size:
            print(f"\n入れ替わる画素の色 (1枚目 {images[0][0].name} から):")
            for entry in dominant_colors(colors.reshape(-1, 1, 3), top=5,
                                         ignore_near_white=256):
                r, g, b = entry["rgb"]
                h, sat, v = entry["hsv"]
                print(f"    RGB({r:3d},{g:3d},{b:3d})  HSV({h:3d},{sat:3d},{v:3d})  "
                      f"{entry['pixels']:7d}px  {entry['share']:6.1%}")
        print("\n中間色(芯より薄い色)が並ぶなら縁のちらつきで、無害。")
        print("芯の色のままなら、前線や等圧線に上書きされて隠れている。")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--in-dir", required=True)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--explain-stability", action="store_true",
                        help="帯の画素が画像ごとに入れ替わる原因を調べる")
    parser.add_argument("--band", default="coastline",
                        choices=sorted(DEFAULT_BANDS),
                        help="位置合わせに使う色。既定の海岸線は毎日同じなので基準に向く")
    args = parser.parse_args()

    paths = sorted(p for p in Path(args.in_dir).iterdir()
                   if p.suffix.lower() in IMAGE_SUFFIXES)[:args.limit]
    if not paths:
        raise SystemExit(f"画像が見つかりません: {args.in_dir}")

    sizes = Counter()
    loaded = []
    for path in paths:
        with Image.open(path) as image:
            sizes[image.size] += 1
            loaded.append((path, np.array(image.convert("RGB"))))

    print(f"=== 画像の大きさ ({len(paths)}枚) ===")
    for size, count in sizes.most_common():
        print(f"  {size[0]}x{size[1]}  {count}枚")
    if len(sizes) > 1:
        print("  ★大きさが揃っていない。トリミングが日ごとに違う位置で切っている。")

    # 一番多い大きさだけを比べる(大きさが違うものは位相相関にかけられない)
    modal_size, _ = sizes.most_common(1)[0]
    same_size = [(p, rgb) for p, rgb in loaded if (rgb.shape[1], rgb.shape[0]) == modal_size]
    if len(same_size) < 2:
        raise SystemExit("同じ大きさの画像が2枚未満で、ずれを測れません。")

    ref_path, ref_rgb = same_size[0]
    ref_mask = color_masks(ref_rgb)[args.band]
    print(f"\n=== {args.band} のずれ (基準: {ref_path.name}, {len(same_size)}枚) ===")
    print(f"{'画像':24s} {'dx':>8s} {'dy':>8s} {'確からしさ':>10s}")

    shifts = []
    for path, rgb in same_size[1:]:
        dx, dy, response = estimate_shift(ref_mask, color_masks(rgb)[args.band])
        flag = "" if response >= MIN_RESPONSE else "  (平行移動で説明できない)"
        print(f"{path.name:24s} {dx:8.2f} {dy:8.2f} {response:10.3f}{flag}")
        if response >= MIN_RESPONSE:
            shifts.append((dx, dy))

    if not shifts:
        print("\n平行移動として説明できる組がありませんでした。")
        return

    arr = np.array(shifts)
    span_x = float(arr[:, 0].max() - arr[:, 0].min())
    span_y = float(arr[:, 1].max() - arr[:, 1].min())
    worst = float(np.abs(arr).max())
    print(f"\ndx: {arr[:, 0].min():+.2f} 〜 {arr[:, 0].max():+.2f} (幅 {span_x:.2f} 画素)")
    print(f"dy: {arr[:, 1].min():+.2f} 〜 {arr[:, 1].max():+.2f} (幅 {span_y:.2f} 画素)")

    if args.explain_stability:
        explain_stability(same_size, args.band)

    if max(span_x, span_y) < 1.0:
        print("\n揃っている。座標をそのまま画像間で比べてよい。")
    elif max(span_x, span_y) < 10.0:
        print(f"\n最大 {worst:.1f} 画素ずれている。細い線どうしの一致は崩れるが、")
        print("data/regions.csv のような大きな矩形の判定には影響しにくい。")
    else:
        print(f"\n★最大 {worst:.1f} 画素ずれている。画像間で座標が比較できない。")
        print("data/regions.csv の矩形を測り直すか、前処理で位置を揃えること。")


if __name__ == "__main__":
    main()
