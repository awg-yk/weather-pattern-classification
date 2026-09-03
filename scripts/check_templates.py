r"""切り出したテンプレートが使える状態かを、当てる前に点検する。

なぜ要るか
----------
テンプレートは人が手で切り出す。**周りの等圧線を消し忘れるのが一番多い失敗**で、
そうなると「自分自身にしか当たらない」テンプレートができあがる。天気図側は
記号だけを探しているのに、テンプレートには余分な線が入っているので、
一致スコアがどこでも下がるためである。

見た目では気づきにくい(線が1本入っているだけ)ので、機械で数えて知らせる。

使い方:
    python -m scripts.check_templates --templates data/templates_ndl

    # その時代の天気図に実際に当ててみる(こちらが本番の確認)
    python -m scripts.check_templates --templates data/templates_ndl `
        --chart <その時代の天気図>
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.build_features import ink_image
from scripts.extract_symbols import INK_RANGE, load_templates, symbol_of
from src.chartscale import auto_letter_size, letter_size_arg, sizes_around
from src.chartsymbols import match_templates, resize_template

# 記号の本体が占めるべき割合。これを下回ると、余分なものが写り込んでいる。
# 輪郭文字は1つの塊になるので、本来はほぼ100%になる。
MIN_MAIN_SHARE = 0.80

# 行または列がこの割合以上インクなら、まっすぐな線が通っているとみなす。
# 記号の画線は斜めなので行を埋め尽くさない。正しく切り出した12枚の実測は
# 最大でも0.90未満だった。
LINE_FILL = 0.95


def inspect(name: str, template: np.ndarray) -> list:
    """1枚を見て、気になった点を日本語で返す。"""
    notes = []
    h, w = template.shape
    share = float(template.mean())

    if not (INK_RANGE[0] <= share <= INK_RANGE[1]):
        notes.append(f"記号の割合が{share:.0%}で目安"
                     f"({INK_RANGE[0]:.0%}〜{INK_RANGE[1]:.0%})の外。"
                     "切り出しの範囲が広すぎるか狭すぎる")

    n, _, stats, _ = cv2.connectedComponentsWithStats(
        template.astype(np.uint8), 8)
    if n <= 1:
        notes.append("記号の画素がない")
        return notes

    areas = stats[1:, cv2.CC_STAT_AREA]
    main = float(areas.max() / areas.sum())
    if main < MIN_MAIN_SHARE:
        notes.append(f"一番大きい塊が全体の{main:.0%}しかない(塊は{n - 1}個)。"
                     "**等圧線や隣の文字が混ざっている疑い。**消してから切り直すこと")

    # 端から端まで通っている「塊」は、まず等圧線
    for i in range(1, n):
        x, y, bw, bh, _ = stats[i]
        if (bw >= w * 0.95 and bh <= h * 0.35) or (bh >= h * 0.95 and bw <= w * 0.35):
            notes.append("画像を端から端まで横切る細い塊がある。等圧線の消し忘れ")
            break

    # **線が記号に接していると1つの塊になり、上の2つをすり抜ける。**
    # まっすぐな線そのものを探す。記号の画線は斜めなので、行や列を
    # 埋め尽くすことはない(実測で、正しく切り出した12枚はすべて9割未満)。
    if (template.mean(axis=1).max() >= LINE_FILL
            or template.mean(axis=0).max() >= LINE_FILL):
        notes.append("行または列を埋め尽くす直線がある。"
                     "記号に接した等圧線の消し忘れ(塊としては記号と一体になる)")

    if min(h, w) < 20:
        notes.append(f"小さすぎる({w}x{h})。切り出しの範囲を確かめること")
    return notes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--templates", required=True)
    parser.add_argument("--chart", default=None,
                        help="その時代の天気図。渡すと実際に当ててみる")
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--angle-range", type=float, default=60.0)
    parser.add_argument("--angle-step", type=float, default=5.0)
    parser.add_argument("--letter-size", type=letter_size_arg, default=1.0,
                        help="テンプレートを縮める倍率。auto で天気図の幅から"
                             "自動で決める(data/templates/reference.json が要る)")
    parser.add_argument("--save-annotated", default=None,
                        help="当たった場所に枠を描いた画像の保存先。"
                             "**どれが当たってどれが外れたかは、これを見ないと分からない**")
    args = parser.parse_args()

    templates = load_templates(Path(args.templates))
    print(f"{len(templates)}枚を点検します: {args.templates}\n")

    kinds: dict = {}
    problems = 0
    for name in sorted(templates):
        template = templates[name]
        kinds.setdefault(symbol_of(name), []).append(name)
        h, w = template.shape
        notes = inspect(name, template)
        mark = "  " if not notes else "★"
        print(f"{mark}{name:10s} {w:3d} x {h:3d}  記号{template.mean():4.0%}")
        for note in notes:
            print(f"    - {note}")
        problems += bool(notes)

    print()
    for kind, names in sorted(kinds.items()):
        print(f"  {kind}: {len(names)}枚 ({', '.join(names)})")
    for kind in ("H", "L"):
        if kind not in kinds:
            print(f"★{kind} のテンプレートが1枚もありません")
            problems += 1

    if problems:
        print(f"\n★{problems}件、気になる点があります。上を見て切り直してください。")
    else:
        print("\n形の点検は問題なし。")

    if not args.chart:
        print("\n--chart にその時代の天気図を渡すと、実際に当ててみます"
              "(**こちらが本番の確認です**)。")
        return

    chart = Path(args.chart)
    if not chart.exists():
        # **原因は2通りある。**フォルダが無いのか、そのフォルダに
        # その名前が無いのか。取り違えると見当違いの案内になる
        lines = [f"天気図が見つかりません: {args.chart}"]
        if chart.parent.exists():
            lines.append(f"  フォルダはあります: {chart.parent}")
            stem = chart.stem.split("_")[0] + "_" + chart.stem.split("_")[1][:8] \
                if chart.stem.count("_") >= 1 else chart.stem[:12]
            near = sorted(q.name for q in chart.parent.glob(f"{stem}*"))[:5]
            if near:
                lines.append("  似た名前のファイル:")
                lines.extend(f"    {name}" for name in near)
                lines.append("  末尾が違うだけかもしれません"
                             "(国会図書館由来のものは _page001 が付く)")
            else:
                lines.append(f"  {stem}* に当てはまるファイルはありません")
        else:
            lines.append(f"  フォルダがありません: {chart.parent}")
            lines.append(f"  いまの場所: {Path.cwd()}")
        raise SystemExit("\n".join(lines))
    rgb = np.array(Image.open(chart).convert("RGB"))

    sizes = (1.0,)
    size = args.letter_size
    if size == "auto":
        size, note = auto_letter_size(rgb.shape[1], args.templates)
        print(f"\n大きさの自動調整: {note}")
        # 推定がぴったりとは限らないので、まわりを少しだけ振る
        sizes = tuple(s / size for s in sizes_around(size)) if size != 1.0 else (1.0,)
    if size != 1.0:
        templates = {name: resize_template(t, size) for name, t in templates.items()}

    angles = np.arange(-args.angle_range,
                       args.angle_range + args.angle_step, args.angle_step)
    hits = match_templates(ink_image(rgb), templates,
                           threshold=args.threshold, angles=angles, sizes=sizes)
    counts: dict = {}
    for hit in hits:
        counts[symbol_of(hit.label)] = counts.get(symbol_of(hit.label), 0) + 1
    print(f"\n{chart.name}({rgb.shape[1]}x{rgb.shape[0]})に当てた結果: "
          + (", ".join(f"{k} {v}個" for k, v in sorted(counts.items())) or "0個"))
    print("  2023年以降の天気図では H 2.8個 / L 3.9〜4.2個 が平均です。")
    if len(hits) <= 1:
        print("★1個以下です。**自分自身にしか当たっていない可能性が高い。**"
              "テンプレートに等圧線が混ざっていないか確かめてください。")

    if args.save_annotated:
        out = rgb.copy()
        for hit in hits:
            kind = symbol_of(hit.label)
            colour = (0, 150, 0) if kind == "H" else (255, 130, 0)
            cv2.rectangle(out, (hit.x0, hit.y0), (hit.x1, hit.y1), colour, 3)
            cv2.putText(out, f"{kind} {hit.score:.2f}", (hit.x0, max(hit.y0 - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA)
        path = Path(args.save_annotated)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(out).save(path)
        print(f"\n枠を描いた画像: {path}")
        print("  緑=高気圧 橙=低気圧。**当たらなかった H・L がどれかを見てください。**")
        print("  取りこぼしが多いなら、その個体からもう1枚テンプレートを作ると当たります。")


if __name__ == "__main__":
    main()
