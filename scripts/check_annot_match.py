r"""既にある注釈付き画像が、いまの検出設定と同じ条件で作られたかを確かめる。

なぜ要るのか
------------
「位置だけ嘘の枠」で対照実験をするとき、**枠の個数が本番と揃っていないと
比較にならない。**位置以外の違いも一緒に測ってしまう。

`--marks` を付けたかどうかで拾える数が変わる(実測で 高2.65 / 低3.40 の差)。
しかし本番の `all_annot` を作ったときの引数は記録に残っていない。

そこで**画像に実際に描かれている枠を数えて**、いまの検出結果と突き合わせる。
記憶に頼らずに済む。

使い方:
    python -m scripts.check_annot_match ^
        --annot-dir data\processed\all_annot ^
        --detections data\detections.json --limit 40
"""

import argparse
import json
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from PIL import Image

from scripts.annotate_charts import HIGH_COLOR, LOW_COLOR

# 枠の色は完全一致で置かれる(cv2.rectangle は補間しない)ので、
# 少しだけ幅を持たせれば十分。JPEG保存などで滲んだ場合に備える
TOLERANCE = 12


def count_boxes(rgb: np.ndarray, color, tolerance: int = TOLERANCE) -> int:
    """その色の矩形がいくつ描かれているかを数える。

    枠は輪郭だけなので、色の付いた画素のかたまりを数えると1枠=1かたまりになる。
    重なった枠は1つに数えてしまうが、**目的は「本番と揃っているか」の突き合わせ**
    なので、同じ数え方を両方に当てれば足りる。
    """
    from scipy import ndimage

    target = np.array(color, dtype=np.int16)
    close = (np.abs(rgb.astype(np.int16) - target).max(axis=2) <= tolerance)
    if not close.any():
        return 0
    _labelled, count = ndimage.label(close)
    return int(count)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--annot-dir", required=True,
                        help="本番で使った注釈付き画像(data\\processed\\all_annot)")
    parser.add_argument("--detections", required=True,
                        help="いまの設定で取った座標(data\\detections.json)")
    parser.add_argument("--limit", type=int, default=40, help="調べる枚数")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--diagnose", action="store_true",
                        help="食い違った画像について、色の付いた画素数まで出す。"
                             "**枠が本当に無いのか、数え方が悪いのか**を切り分ける")
    args = parser.parse_args()

    found_all = json.loads(Path(args.detections).read_text(encoding="utf-8"))
    annot_dir = Path(args.annot_dir)
    names = sorted(n for n in found_all if (annot_dir / n).exists())
    if not names:
        raise SystemExit(
            f"突き合わせられる画像がありません。\n"
            f"  {annot_dir} に、{args.detections} と同じ名前の画像が要ります")

    rng = random.Random(args.seed)
    sample = rng.sample(names, min(args.limit, len(names)))
    print(f"{annot_dir} と {args.detections} を {len(sample)}枚で突き合わせます\n")

    drawn_high = drawn_low = json_high = json_low = 0
    mismatched = []
    for name in sample:
        rgb = np.array(Image.open(annot_dir / name).convert("RGB"))
        in_image = (count_boxes(rgb, HIGH_COLOR), count_boxes(rgb, LOW_COLOR))
        found = found_all[name]
        # 画像には縁の枠(細線)も同じ色で描かれている
        in_json = (len(found.get("highs", [])) + len(found.get("edge_highs", [])),
                   len(found.get("lows", [])) + len(found.get("edge_lows", [])))
        drawn_high += in_image[0]
        drawn_low += in_image[1]
        json_high += in_json[0]
        json_low += in_json[1]
        if in_image != in_json:
            mismatched.append((name, in_image, in_json))

    n = len(sample)
    print(f"  {'':<12}{'画像の枠':>12}{'いまの検出':>12}")
    print("  " + "-" * 38)
    print(f"  {'高気圧':<12}{drawn_high / n:>11.2f}{json_high / n:>12.2f}")
    print(f"  {'低気圧':<12}{drawn_low / n:>11.2f}{json_low / n:>12.2f}")
    print(f"\n  枚数が食い違った画像: {len(mismatched)}枚 / {n}枚")

    for name, in_image, in_json in mismatched[:5]:
        print(f"    {name}  画像 高{in_image[0]}/低{in_image[1]}"
              f"  検出 高{in_json[0]}/低{in_json[1]}")
    if len(mismatched) > 5:
        print(f"    ... 他{len(mismatched) - 5}枚")

    if args.diagnose and mismatched:
        # **枠が本当に無いのか、数え方が悪いのかを切り分ける。**
        # 画素が0なら描かれていない。画素はあるのに数が合わないなら、
        # かたまりの数え方(重なりを1つにしている等)の問題
        print("\n  --- 色の付いた画素数(0なら本当に描かれていない) ---")
        for name, in_image, in_json in mismatched[:5]:
            rgb = np.array(Image.open(annot_dir / name).convert("RGB"))
            counts = []
            for label, color in (("緑(高)", HIGH_COLOR), ("橙(低)", LOW_COLOR)):
                close = (np.abs(rgb.astype(np.int16)
                                - np.array(color, dtype=np.int16)).max(axis=2)
                         <= TOLERANCE)
                counts.append(f"{label} {int(close.sum())}画素")
            print(f"    {name}  " + " / ".join(counts))
        print("\n  橙が0画素なら、その画像には低気圧の枠が描かれていません"
              "(検出設定が違う)。")
        print("  橙があるのに数が合わないなら、数え方の問題です"
              "(枠が重なって1つに数えられている)。")

    print()
    if not mismatched:
        print("★同じ条件で作られています。対照実験の比較はそのまま成り立ちます。")
    elif len(mismatched) <= n * 0.1:
        print("★ほぼ同じです。重なった枠を1つに数えている分の食い違いと考えられます。")
        print("  比較は成り立つと見てよいでしょう。")
    else:
        drawn = drawn_high + drawn_low
        current = json_high + json_low
        print("★**条件が違います。**このまま比べると、位置以外の違いも"
              "一緒に測ることになります。")
        if current > drawn * 1.15:
            print("  いまの検出のほうが多く拾っています。本番は --marks を"
                  "**付けずに**作られた可能性が高いです。")
            print("  手順1から --marks を外して取り直してください。")
        elif drawn > current * 1.15:
            print("  画像のほうが枠が多いです。本番は --marks を**付けて**"
                  "作られた可能性が高いです。")
            print("  手順1に --marks data\\marks を足して取り直してください。")
        else:
            print("  個数は近いのに枚数が食い違っています。しきい値や"
                  "テンプレートの違いかもしれません。")


if __name__ == "__main__":
    main()
