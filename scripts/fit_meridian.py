r"""記号の傾きが「経線の向き」で説明できるかを確かめ、その向きを当てる。

なぜ要るか
----------
気象庁の天気図の H / L は経線と平行に描かれているので、記号は場所ごとに
傾く。テンプレートマッチングはこれを -60度〜+60度の5度刻みで吸収しているが、
刻みの真ん中に落ちた記号はスコアが 0.935 から 0.705 まで下がり、等圧線が
1本重なると 0.605 でしきい値(0.65)を割る。

経線は投影法によらず**図の外の1点から放射状に出る直線**になるので、その点
さえ分かれば、どの画素でも期待される傾きが計算できる。ここでは検出できて
いる記号から、その点を当てる。

使い方
------
    python -m scripts.fit_meridian --images-dir data\processed\all --limit 200

出力は `data\meridian.json`。**残差(中央絶対偏差)が2度を下回れば、
傾きは位置で決まっていると言える。**10度を超えるなら、この考え方が
成り立っていないので、先に進んではいけない。

    --report reports\meridian_fit.csv    標本を1行ずつ書き出す(図を描く用)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.build_features import ink_image, load_templates_scaled, shrink
from src.chartsymbols import match_templates
from src.meridian import DEFAULT_MERIDIAN_PATH, Meridian, fit, template_tilts

DEFAULT_TEMPLATES = _ROOT / "data" / "templates"
DEFAULT_IMAGES = _ROOT / "data" / "processed" / "all"

# 当てはめに使う標本のしきい値。**検出の 0.65 より高くしてある。**
# ぎりぎりで当たった記号は角度も当てずっぽうに近く、当てはめを濁らせる。
# 位置と傾きの関係さえ分かればよいので、確かなものだけ使えばよい。
SAMPLE_THRESHOLD = 0.75


def collect(paths, templates: dict, tilts: dict, scale: float,
            threshold: float, angle_range: float, angle_step: float,
            progress_every: int = 20) -> tuple:
    """天気図から(位置, 記号の傾き)の標本を集める。

    記号の傾き = テンプレート自身の傾き + 当たった回転角。
    """
    angles = np.arange(-angle_range, angle_range + angle_step, angle_step)
    xs, ys, got, names, scores, sources = [], [], [], [], [], []
    for i, path in enumerate(paths, 1):
        if progress_every and i % progress_every == 0:
            print(f"  {i}/{len(paths)} 枚 標本 {len(xs)}個", flush=True)
        rgb = np.array(Image.open(path).convert("RGB"))
        hits = match_templates(shrink(ink_image(rgb), scale), templates,
                               threshold=threshold, angles=angles)
        for hit in hits:
            xs.append(hit.cx)
            ys.append(hit.cy)
            got.append(tilts[hit.label] + hit.angle)
            names.append(hit.label)
            scores.append(hit.score)
            sources.append(Path(path).name)
    return xs, ys, got, names, scores, sources


def images_in(directory: Path, limit: int) -> list:
    paths = sorted(p for p in Path(directory).iterdir()
                   if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not paths:
        raise SystemExit(f"{directory} に画像がありません")
    if limit and len(paths) > limit:
        # 端の年に偏らないよう、等間隔で間引く
        step = len(paths) / limit
        paths = [paths[int(i * step)] for i in range(limit)]
    return paths


def report(meridian: Meridian, xs, ys, tilts) -> None:
    """当てはめの結果と、信用してよいかどうかを出す。"""
    dev = np.abs(meridian.deviation(xs, ys, tilts))
    print(f"\n経線が集まる点: x={meridian.x:+.3f} y={meridian.y:+.3f} "
          f"(相対座標。画像の外に出るのが正常)")
    print(f"標本 {meridian.samples}個  残差(中央絶対偏差) {meridian.residual:.2f}度")
    print(f"  ずれが 2度以内 {np.mean(dev <= 2):.0%} / "
          f"5度以内 {np.mean(dev <= 5):.0%} / 10度以内 {np.mean(dev <= 10):.0%}")

    grid = np.linspace(0.05, 0.95, 5)
    gx, gy = np.meshgrid(grid, grid)
    predicted = meridian.tilt_at(gx, gy)
    print(f"予測される傾きの範囲 {predicted.min():+.1f}度 〜 {predicted.max():+.1f}度 "
          f"(幅 {predicted.max() - predicted.min():.1f}度)")
    print("\n画面を5x5に割ったときの予測(度):")
    print("        " + "".join(f"x={v:4.2f} ".rjust(8) for v in grid))
    for row, v in enumerate(grid):
        cells = "".join(f"{predicted[row, col]:+7.1f} " for col in range(len(grid)))
        print(f"  y={v:4.2f} {cells}")

    if meridian.residual <= 2.0:
        print("\n-> 傾きは位置で決まっている。予測した角度で測り直す価値がある。")
    elif meridian.residual <= 5.0:
        print("\n-> おおむね位置で決まっているが、ずれが残る。予測の前後を"
              "広めに振ること(--span を大きく)。")
    else:
        print("\n★残差が大きすぎる。傾きが経線で決まっているとは言えない。"
              "標本のしきい値を上げるか、この方針を見直すこと。")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--limit", type=int, default=200,
                        help="使う枚数。多いほど当てはめは安定するが時間がかかる")
    parser.add_argument("--templates", type=Path, default=DEFAULT_TEMPLATES)
    parser.add_argument("--scale", type=float, default=0.7)
    parser.add_argument("--letter-size", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=SAMPLE_THRESHOLD)
    parser.add_argument("--angle-range", type=float, default=60.0)
    parser.add_argument("--angle-step", type=float, default=5.0)
    parser.add_argument("--out", type=Path, default=DEFAULT_MERIDIAN_PATH)
    parser.add_argument("--report", type=Path, default=None,
                        help="標本を1行ずつCSVに書き出す")
    args = parser.parse_args(argv)

    templates = load_templates_scaled(args.templates,
                                      args.scale * args.letter_size, quiet=True)
    if not templates:
        raise SystemExit(f"{args.templates} に H/L のテンプレートがありません")

    print("テンプレート自身の傾きを測ります")
    raw = load_templates_scaled(args.templates, 1.0, quiet=True)
    tilts = template_tilts(raw)
    for name in sorted(tilts):
        print(f"  {name:>4} {tilts[name]:+6.1f}度")

    paths = images_in(args.images_dir, args.limit)
    print(f"\n{len(paths)}枚から標本を集めます(しきい値 {args.threshold})")
    xs, ys, got, names, scores, sources = collect(
        paths, templates, tilts, args.scale, args.threshold,
        args.angle_range, args.angle_step)
    print(f"標本 {len(xs)}個")
    if len(xs) < 3:
        raise SystemExit("標本が足りません。--threshold を下げるか --limit を増やしてください")

    meridian = fit(xs, ys, got, note=f"{len(paths)}枚 しきい値{args.threshold}")
    report(meridian, xs, ys, got)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    meridian.save(args.out)
    print(f"\n{args.out} に書きました")

    if args.report:
        import pandas as pd
        args.report.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "image": sources, "template": names, "x": xs, "y": ys,
            "tilt": got, "score": scores,
            "predicted": meridian.tilt_at(xs, ys),
            "deviation": meridian.deviation(xs, ys, got),
        }).to_csv(args.report, index=False, encoding="utf-8-sig")
        print(f"{args.report} に標本を書きました")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
