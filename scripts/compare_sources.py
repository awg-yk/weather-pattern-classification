r"""同じ日の天気図を、入手経路の違う2つのフォルダで比べる。

なぜ要るか
----------
2023〜2025年の天気図は2通りの経路で手に入る。

  気象庁PDF版   気象庁のサイトのPDFを画像にしたもの (Js_2023010100.png)
  国会図書館版  国会図書館のデジタル化資料から取ったもの (Js_2023010100_page001.png)

**同じ日でも絵が違えば、片方で学習した重みはもう片方に使えない。**
逆に、ほとんど同じなら全期間を1つの経路に揃えられる。どちらなのかを
推測せず、同じ日の画像を突き合わせて測る。

前処理(切り取り+基準の大きさへ揃える)を通してから比べるので、
版面の違いは吸収された状態で、絵そのものの違いだけが出る。

使い方:
    python -m scripts.compare_sources --a data\processed\jma --b data\raw\new_png --limit 5

    # 見比べる画像も書き出す
    python -m scripts.compare_sources --a ... --b ... --save-dir compare_out
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
from scripts.extract_symbols import symbol_of
from scripts.preprocess_jma import (DEFAULT_STAMP_BOX, autocrop_to_content,
                                    fit_to_canonical, mask_stamp_box)
from src.chartsymbols import match_templates
from src.split import index_images_by_stamp


def prepare(path: Path, already_processed: bool) -> np.ndarray:
    """前処理を通して、基準の大きさに揃えた画像を返す。"""
    image = Image.open(path).convert("RGB")
    if not already_processed:
        image = mask_stamp_box(fit_to_canonical(autocrop_to_content(image)),
                               DEFAULT_STAMP_BOX)
    else:
        image = fit_to_canonical(image)
    return np.array(image)


def detect(rgb: np.ndarray, templates: dict, threshold: float, angles) -> dict:
    counts = {"H": 0, "L": 0}
    for hit in match_templates(ink_image(rgb), templates,
                               threshold=threshold, angles=angles):
        kind = symbol_of(hit.label)
        if kind in counts:
            counts[kind] += 1
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True, help="片方のフォルダ")
    parser.add_argument("--b", required=True, help="もう片方のフォルダ")
    parser.add_argument("--a-processed", action="store_true",
                        help="--a が前処理済み(切り取り済み)なら指定する")
    parser.add_argument("--b-processed", action="store_true")
    parser.add_argument("--limit", type=int, default=5,
                        help="比べる枚数。全期間に散らして拾う")
    parser.add_argument("--templates", default=str(_ROOT / "data" / "templates"))
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--angle-range", type=float, default=60.0)
    parser.add_argument("--angle-step", type=float, default=5.0)
    parser.add_argument("--save-dir", default=None,
                        help="見比べる画像(左:A 右:B 下:差)の書き出し先")
    args = parser.parse_args()

    a_index = index_images_by_stamp(args.a)
    b_index = index_images_by_stamp(args.b)
    shared = sorted(set(a_index) & set(b_index))
    if not shared:
        raise SystemExit(
            f"同じ日時の天気図が1つもありません。\n"
            f"  {args.a}: {len(a_index)}件  {args.b}: {len(b_index)}件\n"
            "  期間が重なっているか確かめてください")
    print(f"両方にある日時: {len(shared)}件"
          f"({shared[0]} 〜 {shared[-1]})\n")

    picked = shared
    if args.limit and len(shared) > args.limit:
        step = len(shared) / args.limit
        picked = [shared[int(i * step)] for i in range(args.limit)]

    templates = load_templates_scaled(Path(args.templates), 1.0, quiet=True)
    angles = np.arange(-args.angle_range,
                       args.angle_range + args.angle_step, args.angle_step)
    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    print("日時          違う画素   平均の差   検出 A(H/L)  検出 B(H/L)")
    diffs = []
    for stamp in picked:
        a = prepare(Path(a_index[stamp]), args.a_processed)
        b = prepare(Path(b_index[stamp]), args.b_processed)
        if a.shape != b.shape:
            print(f"{stamp}  大きさが違う: {a.shape} と {b.shape}")
            continue
        gap = np.abs(a.astype(np.int16) - b.astype(np.int16))
        differing = float((gap.max(axis=2) > 16).mean())
        diffs.append(differing)
        ca = detect(a, templates, args.threshold, angles)
        cb = detect(b, templates, args.threshold, angles)
        print(f"{stamp}  {differing:7.2%}  {gap.mean():8.2f}   "
              f"{ca['H']} / {ca['L']}        {cb['H']} / {cb['L']}")
        if save_dir:
            band = np.full((a.shape[0], 8, 3), 255, np.uint8)
            Image.fromarray(np.hstack([a, band, b])).save(
                save_dir / f"{stamp}_ab.png")

    if not diffs:
        return
    worst = max(diffs)
    print(f"\n違う画素の割合: 平均 {np.mean(diffs):.2%}  最大 {worst:.2%}")
    if worst < 0.02:
        print("**ほぼ同じ画像です。**片方に一本化してよい"
              "(学習済みの重みもそのまま使える見込み)。")
    elif worst < 0.10:
        print("少し違います。検出の数が揃っているなら検出には使えますが、"
              "**学習に使う画像を入れ替えるなら学習し直しが要ります。**")
    else:
        print("**別物と考えるべきです。**片方で学習した重みは、"
              "もう片方には使えません。--save-dir で書き出して目で見比べてください。")


if __name__ == "__main__":
    main()
