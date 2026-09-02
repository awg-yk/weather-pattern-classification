"""
JMA天気図PNGの前処理スクリプト。

以下を行う:
  1. 画像周囲の余白（PDFページの白い余白）を自動でクロップし、図郭（黒枠）に合わせる
  2. 右下の日時スタンプ枠（例: "2026.04.01.00UTC"）を白で塗りつぶす
     -> モデルが気圧パターンと無関係な文字を学習してしまうのを防ぐ

日時スタンプの位置は画像に対する相対座標(0.0〜1.0)で指定する。
scripts/collect_jma.py でダウンロードした画像で位置がずれる場合は
--stamp-box を調整すること。

使い方:
    python scripts/preprocess_jma.py --in-dir data/raw/jma/png --out-dir data/processed/jma
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage

# 右下の日時スタンプのおおよその位置 (left, top, right, bottom) を画像サイズに対する割合で指定
DEFAULT_STAMP_BOX = (0.65, 0.92, 1.0, 1.0)


def autocrop_to_content(image: Image.Image, white_threshold: int = 250) -> Image.Image:
    """画像周囲の白い余白を除去し、図郭(枠)ぎりぎりまでクロップする。

    2000〜2002年頃の国立国会図書館デジタルコレクション由来のJPEGには、図郭の外側に
    ビューアの灰色のページ送りボタン(右に2つ・左に1つ)が写り込んでいることがある。
    単純な非白領域のバウンディングボックスだとこのボタンまで含めてクロップしてしまう
    ため、非白領域を連結成分に分解し、最大の連結成分(図郭+等圧線+文字などが繋がった
    本体)のバウンディングボックスのみを使う。ボタンは図郭から離れた孤立した連結成分
    になるため、自然に除外される。
    """
    gray = ImageOps.grayscale(image)
    arr = np.array(gray)
    non_white_mask = arr < white_threshold
    if not non_white_mask.any():
        return image

    labeled, num_components = ndimage.label(non_white_mask, structure=np.ones((3, 3)))
    if num_components > 1:
        component_sizes = ndimage.sum(non_white_mask, labeled, range(1, num_components + 1))
        largest_component = np.argmax(component_sizes) + 1
        non_white_mask = labeled == largest_component

    rows = np.where(non_white_mask.any(axis=1))[0]
    cols = np.where(non_white_mask.any(axis=0))[0]
    top, bottom = rows[0], rows[-1]
    left, right = cols[0], cols[-1]
    return image.crop((left, top, right + 1, bottom + 1))


def mask_stamp_box(image: Image.Image, stamp_box: tuple) -> Image.Image:
    """日時スタンプ部分を白く塗りつぶす。"""
    w, h = image.size
    left = int(stamp_box[0] * w)
    top = int(stamp_box[1] * h)
    right = int(stamp_box[2] * w)
    bottom = int(stamp_box[3] * h)

    arr = np.array(image)
    arr[top:bottom, left:right] = 255
    return Image.fromarray(arr)


# 切り取ったあとに揃える大きさ。**data/templates と同梱の重みがこの縮尺で
# 作られている。**気象庁PDF版(2023年以降)の切り取り結果がちょうどこの
# 大きさなので、その時代の画像は1画素も変わらない(=いまの重みが使える)。
#
# 国立国会図書館由来の天気図は、同じ紙の上に枠が3.2%大きく描かれていて、
# 切り取ると1499x1548になる。ここで揃えることで、時代ごとの場合分けが要らなく
# なる。縦横比の差は0.033%しかないので、揃えても形は歪まない。
CANONICAL_SIZE = (1453, 1500)


def fit_to_canonical(image: Image.Image, size=CANONICAL_SIZE) -> Image.Image:
    """決まった大きさに揃える。**すでにその大きさなら何もしない。**

    同じ大きさへの resize でも補間が走って画素が変わりうる。素通しにして
    おかないと、これまでと同じ画像であるはずのものが変わってしまい、
    学習済みの重みに違う絵を渡すことになる。
    """
    if image.size == tuple(size):
        return image
    return image.resize(tuple(size), Image.LANCZOS)


def crop_box(image: Image.Image, white_threshold: int = 250) -> tuple:
    """autocrop_to_content が切る位置と、非白の連結成分の数を返す。

    `(left, top, right, bottom, 連結成分の数)`。書き出さずに理由だけ見たいときに使う。
    """
    gray = ImageOps.grayscale(image)
    arr = np.array(gray)
    non_white = arr < white_threshold
    if not non_white.any():
        return (0, 0, image.size[0], image.size[1], 0)
    labeled, count = ndimage.label(non_white, structure=np.ones((3, 3)))
    if count > 1:
        sizes = ndimage.sum(non_white, labeled, range(1, count + 1))
        non_white = labeled == (np.argmax(sizes) + 1)
    rows = np.where(non_white.any(axis=1))[0]
    cols = np.where(non_white.any(axis=0))[0]
    return (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1, int(count))


def report(files: list, size=CANONICAL_SIZE) -> None:
    """切り取り位置を並べて出す。**生が同じでも切り取り後が違う**理由が見える。

    揃える大きさを渡すと、揃えても画像が変わらない枚数を数える。
    変わらない枚数が多いほど、いまの重みをそのまま使える。
    """
    print("ファイル                          生の大きさ    切り取り位置(左,上,右,下)      切り取り後   揃えると")
    unchanged = 0
    for path in files:
        image = Image.open(path).convert("RGB")
        left, top, right, bottom, _ = crop_box(image)
        cropped = (right - left, bottom - top)
        if size and cropped == tuple(size):
            verdict, unchanged = "変わらない", unchanged + 1
        elif size:
            verdict = f"{size[0]}x{size[1]} に揃える"
        else:
            verdict = "揃えない"
        print(f"{path.name:32s} {image.size[0]:5d}x{image.size[1]:<5d} "
              f"({left:4d},{top:4d},{right:5d},{bottom:5d})  "
              f"{cropped[0]:5d}x{cropped[1]:<5d} {verdict}")
    if size:
        print(f"\n{len(files)}枚のうち {unchanged}枚は揃えても変わらない"
              "(=いまの重みにこれまでと同じ絵を渡せる)。")
    print("\n切るのは紙の大きさではなく「非白の最大の連結成分」の外接矩形。")
    print("**枠が紙のどこにどれだけの大きさで描かれているか**で切り取り後の大きさが決まる。")
    print("塊の数が多いのは、図郭の外に何か写り込んでいるということ")
    print("(国会図書館のビューアの灰色のボタンなど。最大の塊だけ使うので普通は無害)。")


def process_image(src_path: Path, dst_path: Path, stamp_box: tuple,
                  size=CANONICAL_SIZE) -> None:
    image = Image.open(src_path).convert("RGB")
    image = autocrop_to_content(image)
    if size:
        # **揃えてから塗る。**日時スタンプの位置は相対座標なので、
        # 揃えた後の格子の上で正確に当たる
        image = fit_to_canonical(image, size)
    image = mask_stamp_box(image, stamp_box)
    image.save(dst_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--limit", type=int, default=3,
                        help="--report で見る枚数")
    parser.add_argument(
        "--stamp-box",
        type=float,
        nargs=4,
        default=DEFAULT_STAMP_BOX,
        metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
        help="日時スタンプの位置(相対座標 0-1)。デフォルトは右下想定。",
    )
    parser.add_argument(
        "--report", action="store_true",
        help="書き出さずに、切り取り位置だけを報告する。**生の大きさが同じでも"
             "切り取り後が違う**理由を見るためのもの。切るのは紙の大きさではなく"
             "「非白の最大の連結成分」の外接矩形なので、枠が紙のどこにどれだけの"
             "大きさで描かれているかで結果が変わる",
    )
    parser.add_argument(
        "--size", default="x".join(str(v) for v in CANONICAL_SIZE),
        help="切り取ったあとに揃える大きさ(幅x高さ)。"
             "**これを変えると、いまの重みとテンプレートが合わなくなる。**")
    parser.add_argument(
        "--no-resize", action="store_true",
        help="大きさを揃えない(切り取ったまま)。時代ごとに縮尺が違ったままになる")
    args = parser.parse_args()

    size = None if args.no_resize else tuple(
        int(v) for v in args.size.lower().split("x"))

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*.png"))
    if not files:
        print(f"no PNG files found in {in_dir}")
        return

    if args.report:
        report(files[:args.limit] if args.limit else files, size)
        return

    for src_path in files:
        dst_path = out_dir / src_path.name
        process_image(src_path, dst_path, tuple(args.stamp_box), size)
        print(f"processed: {dst_path}")


if __name__ == "__main__":
    main()
