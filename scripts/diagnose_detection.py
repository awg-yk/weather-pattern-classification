r"""高低気圧が検出できない天気図について、その原因を切り分ける。

なぜ要るか
----------
テンプレートマッチング(`cv2.matchTemplate`)は**大きさの違いに対応しない。**
テンプレートは特定の天気図から切り出したものなので、解像度の違う天気図では
同じ H でも画素数が違い、スコアがしきい値に届かず検出がゼロになる。

原因は他にもありうる。切り分けたいのは次の4つで、どれなのかで直し方が変わる:

  1. 解像度が違う      -> テンプレートの大きさを変えれば当たる
  2. 白黒のスキャン    -> 等圧線の色帯が空になり、そもそも当てる下地が無い
  3. 記号の書体が違う  -> どの大きさでも当たらない。テンプレートを切り直す
  4. 画像が違う        -> 天気図でないか、前処理が済んでいない

使い方
------
    python -m scripts.diagnose_detection --images <画像かフォルダ>

うまくいっている天気図と、いっていない天気図の**両方**を渡すこと。
片方だけでは「その値が普通なのか」が分からない。

    python -m scripts.diagnose_detection `
        --images ..\weather-pattern-classification-data\processed\Js_2023010100.png `
        --images <2023年より前の天気図>
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.build_features import ink_image, load_templates_scaled
from src.chartsymbols import (DEFAULT_BANDS, ink_mask, match_templates,
                             to_hsv)

DEFAULT_TEMPLATES = _ROOT / "data" / "templates"

# 掃引する倍率。0.5〜2.0 を対数的に並べてある。天気図の解像度がこの範囲の外に
# あることは考えにくい(範囲外なら、掃引の端で当たったと分かるので気づける)。
SIZES = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.4, 1.6, 1.8, 2.0)

# 等圧線の画素がこの割合を下回ると、色帯が空とみなす。白黒スキャンや、
# 天気図でない画像で起きる。実測では気象庁PDF版で 2〜5% 程度。
MIN_INK_SHARE = 0.005

# 1枚あたりの H+L がこれを超えたら、その倍率の検出は信用しない。
#
# **テンプレートを縮めると誤検出が急に増える。**画素数が減るぶん、等圧線の
# 曲がりのような何でもない模様と相関が上がるためで、合成天気図では記号2個の
# 図で0.5倍にすると25個当たった。実測の平均は H 2.80 / L 3.90 なので、
# 15個を超える時点で模様を拾っている。
MAX_PLAUSIBLE = 15


def symbol_of(name: str) -> str:
    from scripts.extract_symbols import symbol_of as _symbol_of
    return _symbol_of(name)


def colourfulness(rgb: np.ndarray) -> float:
    """色のある画素の割合。白黒スキャンかどうかの目安になる。"""
    hsv = to_hsv(rgb)
    return float((hsv[..., 1] > 60).mean())


def counts_at(ink: np.ndarray, templates: dict, size: float,
              threshold: float, angles) -> dict:
    hits = match_templates(ink, templates, threshold=threshold,
                           angles=angles, sizes=(size,))
    counts = {"H": 0, "L": 0}
    for hit in hits:
        kind = symbol_of(hit.label)
        if kind in counts:
            counts[kind] += 1
    return counts


def diagnose(path: Path, templates: dict, threshold: float, angles) -> dict:
    rgb = np.array(Image.open(path).convert("RGB"))
    height, width = rgb.shape[:2]
    fixed = DEFAULT_BANDS["isobar"].mask(to_hsv(rgb))
    ink_share = float(fixed.mean())
    adaptive_mask, fell_back = ink_mask(rgb)
    ink = ink_image(rgb)   # 控えが要る画像では、ここで自動的に切り替わる

    sweep = {size: counts_at(ink, templates, size, threshold, angles)
             for size in SIZES}
    totals = {size: c["H"] + c["L"] for size, c in sweep.items()}
    # 信用できる倍率のなかから選ぶ。数が多すぎるものは模様を拾っている
    credible = {s: n for s, n in totals.items() if n <= MAX_PLAUSIBLE}
    best = max(credible, key=lambda s: (credible[s], -abs(s - 1.0)))

    return {
        "path": path,
        "width": width,
        "height": height,
        "ink_share": ink_share,
        "adaptive_share": float(adaptive_mask.mean()),
        "fell_back": fell_back,
        "colour_share": colourfulness(rgb),
        "sweep": sweep,
        "best_size": best,
        "best_total": totals[best],
        "plain_total": totals[1.0],
    }


def verdict(result: dict) -> str:
    """4つの原因のどれかを言う。**断定できるのは2つだけ**なので、残りは
    疑いとして言い、目で確かめる手順を添える。

    縮めたテンプレートは誤検出が増えるので(MAX_PLAUSIBLE 参照)、当たった数が
    増えたことだけを根拠に「解像度の違いだ」と言い切ってはいけない。
    """
    if result["fell_back"]:
        head = (f"固定の色帯では読めません(黒い画素 {result['ink_share']:.2%})。"
                "紙のスキャンなどで線が真っ黒でない場合に起きます。"
                f"**濃さのしきい値に自動で切り替えました**(同 {result['adaptive_share']:.2%})。")
        if result["plain_total"] > 0:
            return head + f"\n     切り替え後は原寸で {result['plain_total']}個。" \
                          "色を見ないので海岸線も拾います。枠を目で確かめてください。"
        return head + "\n     切り替えても0個なので、別の原因が重なっています。"
    if result["adaptive_share"] < MIN_INK_SHARE:
        return ("線の画素がほとんどありません。天気図でないか、前処理が"
                "済んでいない可能性があります。**大きさの問題ではありません。**")
    if result["plain_total"] > 0:
        return f"原寸で {result['plain_total']}個。この天気図は問題ありません。"
    if result["best_total"] == 0:
        return ("どの大きさでも1個も当たりません。記号の書体が違うか、"
                "掃引の範囲(0.5〜2.0倍)の外です。テンプレートの切り直しが要ります。")
    edge = result["best_size"] in (SIZES[0], SIZES[-1])
    note = ("**掃引の端なので、範囲の外かもしれません。**" if edge else "")
    return (f"原寸では0個ですが、{result['best_size']:.2f}倍で "
            f"{result['best_total']}個当たります。解像度の違いの疑いがあります。"
            + note
            + "\n     縮めたテンプレートは模様も拾うので、数だけでは決められません。"
            "\n     次のコマンドで枠を描いて、本物の H・L に付いているか目で確かめてください:"
            f"\n     python -m scripts.annotate_charts --in-dir <フォルダ> --out-dir <出力> "
            f"--letter-sizes {result['best_size']:g}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", action="append", required=True,
                        help="天気図の画像かフォルダ。複数回指定できる")
    parser.add_argument("--templates", default=str(DEFAULT_TEMPLATES))
    parser.add_argument("--threshold", type=float, default=0.65,
                        help="テンプレートマッチングのしきい値")
    parser.add_argument("--angle-range", type=float, default=6.0)
    parser.add_argument("--angle-step", type=float, default=3.0)
    parser.add_argument("--limit", type=int, default=3,
                        help="フォルダを渡したときに見る枚数")
    args = parser.parse_args()

    paths: list[Path] = []
    for item in args.images:
        p = Path(item)
        if p.is_dir():
            paths.extend(sorted(p.glob("*.png"))[:args.limit]
                         + sorted(p.glob("*.jpg"))[:args.limit])
        else:
            paths.append(p)
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise SystemExit("見つかりません: " + ", ".join(str(p) for p in missing))

    templates = load_templates_scaled(Path(args.templates), 1.0)
    angles = np.arange(-args.angle_range, args.angle_range + args.angle_step,
                       args.angle_step)
    print(f"テンプレート {len(templates)}枚、しきい値 {args.threshold}\n")

    results = []
    for path in paths:
        result = diagnose(path, templates, args.threshold, angles)
        results.append(result)
        print(f"=== {path.name}")
        print(f"  大きさ    : {result['width']} x {result['height']}")
        share = f"{result['ink_share']:.2%}"
        if result["fell_back"]:
            share += f" -> 控えで {result['adaptive_share']:.2%}"
        print(f"  線の画素  : {share}"
              f"   色のある画素: {result['colour_share']:.2%}")
        # 信用できない数には * を付ける。25個などは模様を拾っている
        row = "  ".join(
            f"{s:g}:{c['H'] + c['L']}" + ("*" if c["H"] + c["L"] > MAX_PLAUSIBLE else "")
            for s, c in result["sweep"].items())
        print(f"  倍率:検出数: {row}")
        if any(c["H"] + c["L"] > MAX_PLAUSIBLE for c in result["sweep"].values()):
            print(f"  (* は1枚に{MAX_PLAUSIBLE}個超。模様を拾っているので数えない)")
        print(f"  -> {verdict(result)}\n")

    # 2枚以上あれば、大きさの比を出す。これが直すときの倍率になる
    if len(results) >= 2:
        ok = [r for r in results if r["plain_total"] > 0]
        ng = [r for r in results if r["plain_total"] == 0 and r["best_total"] > 0]
        if ok and ng:
            ratio = ng[0]["width"] / ok[0]["width"]
            print(f"うまくいく天気図の幅 {ok[0]['width']}px に対し、"
                  f"いかないほうは {ng[0]['width']}px(比 {ratio:.3f})。")
            print(f"当たった倍率 {ng[0]['best_size']:.2f} と比べてください。"
                  "近ければ、幅を揃えるだけで直ります。")


if __name__ == "__main__":
    main()
