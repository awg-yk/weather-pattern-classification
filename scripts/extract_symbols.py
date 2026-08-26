"""Phase 1 の下見: H/L/T/TD の記号を、アノテーション無しで取れるか確かめる。

`docs/2026-08-26-detection-plan.md`「着手前に1時間で試すこと」の後半。
記号は同じフォント・同じ大きさで機械的に描画されているので、テンプレート
マッチングで見つかる可能性がある。見つかるなら Phase 1 に YOLO は要らず、
中心座標がそのまま Phase 3 の特徴量(`src/regions.py` の region.contains)になる。

二段構え
--------
1. `scan`  テンプレート無しで、記号になりうる小さな孤立した黒い塊を数える。
           等圧線は図の端から端まで繋がった巨大な成分なので大きさで落ちる。
           残るのは記号と、気圧の数値・目盛といった文字である。
           **1枚あたりの候補数**が、この方式が現実的かどうかの目安になる。
           数十個ならテンプレートで選り分けられる。数百個なら苦しい。

2. `match` 本物の天気図から切り出したテンプレートを全画像に当てる。
           フォントが手元に無いのでテンプレートは実物から採るしかない。
           まず `scan --overlay` で候補に振った番号を見て、H の候補の番号を
           `cut` に渡してテンプレートを作る。

使い方:
    # 1. 候補の規模を見る
    python -m scripts.extract_symbols scan --in-dir data/processed/jma --limit 20 \
        --overlay reports/symbols

    # 2. reports/symbols の番号を見て、Hの候補からテンプレートを作る
    python -m scripts.extract_symbols cut --image data/processed/jma/Js_2025050100.png \
        --index 7 --name H --out data/templates

    # 3. 全画像に当てる
    python -m scripts.extract_symbols match --in-dir data/processed/jma \
        --templates data/templates --limit 20 --overlay reports/symbols
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from src.chartsymbols import crop_template, glyph_candidates, match_templates

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


def iter_images(in_dir: Path, limit: int):
    paths = sorted(p for p in Path(in_dir).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise SystemExit(f"画像が見つかりません: {in_dir}")
    return paths[:limit]


def draw_boxes(rgb: np.ndarray, candidates, out_path: Path, numbered: bool) -> None:
    image = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(image)
    for i, c in enumerate(candidates):
        draw.rectangle((c.x0 - 2, c.y0 - 2, c.x1 + 2, c.y1 + 2), outline=(0, 170, 0), width=2)
        tag = str(i) if numbered else f"{c.label} {c.score:.2f}"
        draw.text((c.x0 - 2, max(0, c.y0 - 14)), tag, fill=(0, 130, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def size_cluster(sizes: Counter, tolerance: int = 3) -> tuple[tuple[int, int], int]:
    """似た大きさをまとめて、一番票の集まった大きさとその票数を返す。

    記号は同じフォント・同じ大きさで描かれるが、線の太さの丸めで1〜2画素
    ゆれる。34x57 と 36x56 と 37x56 は同じ記号とみなしたい。
    """
    best, best_votes = None, 0
    for (w, h) in sizes:
        votes = sum(
            n for (w2, h2), n in sizes.items()
            if abs(w2 - w) <= tolerance and abs(h2 - h) <= tolerance
        )
        if votes > best_votes:
            best, best_votes = (w, h), votes
    return best, best_votes


def cmd_scan(args):
    counts = []
    sizes = Counter()
    located: list[tuple[str, int, int, int]] = []   # (画像, 番号, 幅, 高さ)
    for path in iter_images(args.in_dir, args.limit):
        rgb = np.array(Image.open(path).convert("RGB"))
        candidates = glyph_candidates(rgb, min_side=args.min_side, max_side=args.max_side)
        counts.append(len(candidates))
        for i, c in enumerate(candidates):
            sizes[(c.width, c.height)] += 1
            located.append((path.name, i, c.width, c.height))
        print(f"{path.name:24s} 候補 {len(candidates):4d}")
        if args.overlay:
            draw_boxes(rgb, candidates, Path(args.overlay) / f"{path.stem}_cands.png", True)

    if counts:
        print(f"\n1枚あたりの候補数: 中央値 {int(np.median(counts))}, "
              f"最小 {min(counts)}, 最大 {max(counts)}")
        print("多い大きさ(幅x高さ): " +
              ", ".join(f"{w}x{h}({n})" for (w, h), n in sizes.most_common(8)))
        print("\n記号は同じ大きさで描かれているので、特定の幅x高さに票が集まるはず。")
        print("集まらないなら、記号と数字の大きさが同じか、前処理で拡大率がばらついている。")
        at_ceiling = sum(n for (w, h), n in sizes.items()
                         if max(w, h) >= args.max_side - 2)
        if at_ceiling:
            print(f"※ 上限({args.max_side}画素)ぎりぎりの候補が{at_ceiling}個ある。"
                  f"--max-side を広げて、拾える個数が増えないか確かめること。")
        print(f"※ この結果は --min-side {args.min_side} --max-side {args.max_side} "
              f"のもの。上限を変えた測定どうしは比べられない。")

        center, votes = size_cluster(sizes)
        if center:
            w, h = center
            print(f"\n一番票の集まった大きさ: {w}x{h} あたりに {votes}個 "
                  f"(全{sum(sizes.values())}個中)")
            print("この大きさの候補がどこにあるか(cut --index にこの番号を渡す):")
            shown = 0
            for name, index, cw, ch in located:
                if abs(cw - w) <= 3 and abs(ch - h) <= 3:
                    print(f"    {name}  --index {index}   ({cw}x{ch})")
                    shown += 1
                    if shown >= 12:
                        print("    ...")
                        break
            print("重ね描きでこの番号の箱を見て、H か L か数字かを確かめること。")
    if args.overlay:
        print(f"重ね描き: {args.overlay}/ ― 箱の番号を cut --index に渡す。")


def cmd_cut(args):
    rgb = np.array(Image.open(args.image).convert("RGB"))
    candidates = glyph_candidates(rgb)
    if not 0 <= args.index < len(candidates):
        raise SystemExit(f"--index は 0〜{len(candidates) - 1} で指定してください")
    c = candidates[args.index]
    template = crop_template(rgb, (c.x0, c.y0, c.x1, c.y1))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.name}.png"
    Image.fromarray((template * 255).astype(np.uint8)).save(out_path)
    print(f"{out_path} に保存 ({template.shape[1]}x{template.shape[0]}, "
          f"{int(template.sum())}px)")


def load_templates(template_dir: Path) -> dict[str, np.ndarray]:
    templates = {}
    for path in sorted(Path(template_dir).glob("*.png")):
        arr = np.array(Image.open(path).convert("L"))
        templates[path.stem] = arr > 127
    if not templates:
        raise SystemExit(f"テンプレートがありません: {template_dir} (先に cut を実行)")
    return templates


def cmd_match(args):
    templates = load_templates(args.templates)
    print("テンプレート: " + ", ".join(
        f"{k}({v.shape[1]}x{v.shape[0]})" for k, v in templates.items()))
    per_label = Counter()
    for path in iter_images(args.in_dir, args.limit):
        rgb = np.array(Image.open(path).convert("RGB"))
        hits = match_templates(rgb, templates, threshold=args.threshold)
        found = Counter(h.label for h in hits)
        per_label.update(found)
        detail = ", ".join(f"{k}={v}" for k, v in sorted(found.items())) or "なし"
        print(f"{path.name:24s} {detail}")
        if args.overlay:
            draw_boxes(rgb, hits, Path(args.overlay) / f"{path.stem}_match.png", False)
    print("\n合計: " + ", ".join(f"{k}={v}" for k, v in sorted(per_label.items())))
    print("天気図1枚に高気圧・低気圧はふつう2〜6個。桁が違うなら閾値(--threshold)を上げる。")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="テンプレート無しで候補を数える")
    scan.add_argument("--in-dir", required=True)
    scan.add_argument("--limit", type=int, default=20)
    scan.add_argument("--overlay")
    scan.add_argument("--min-side", type=int, default=6,
                      help="候補とみなす塊の一辺の下限(画素)")
    scan.add_argument("--max-side", type=int, default=64,
                      help="同・上限。上限に貼りつく候補が多いなら広げて試す")
    scan.set_defaults(func=cmd_scan)

    cut = sub.add_parser("cut", help="候補からテンプレートを切り出す")
    cut.add_argument("--image", required=True)
    cut.add_argument("--index", type=int, required=True)
    cut.add_argument("--name", required=True, help="H / L / T / TD / cross など")
    cut.add_argument("--out", default="data/templates")
    cut.set_defaults(func=cmd_cut)

    match = sub.add_parser("match", help="テンプレートを当てる")
    match.add_argument("--in-dir", required=True)
    match.add_argument("--templates", default="data/templates")
    match.add_argument("--limit", type=int, default=20)
    match.add_argument("--threshold", type=float, default=0.8,
                       help="テンプレートの一致スコアの下限。誤検出が多ければ上げる")
    match.add_argument("--overlay")
    match.set_defaults(func=cmd_match)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
