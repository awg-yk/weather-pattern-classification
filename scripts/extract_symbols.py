"""Phase 1 の下見: H/L/T/TD の記号を、アノテーション無しで取れるか確かめる。

`docs/2026-08-26-detection-plan.md`「着手前に1時間で試すこと」の後半。
記号は同じフォント・同じ大きさで機械的に描画されているので、テンプレート
マッチングで見つかる可能性がある。見つかるなら Phase 1 に YOLO は要らず、
中心座標がそのまま Phase 3 の特徴量(`src/regions.py` の region.contains)になる。

三段構え
--------
1. `scan`  テンプレート無しで、記号になりうる小さな孤立した黒い塊を数える。
           等圧線は図の端から端まで繋がった巨大な成分なので大きさで落ちる。
           残るのは記号と、気圧の数値・目盛といった文字である。
           **1枚あたりの候補数**が、この方式が現実的かどうかの目安になる。
           数十個ならテンプレートで選り分けられる。数百個なら苦しい。

2. `cluster` 似た形の候補をまとめ、山ごとの代表を小さなPNGとして書き出す。
           記号は同じフォントで機械的に描かれるので、同じ記号どうしは形が
           揃う。**どの山がHでどれがLかは機械には分からない**ので、
           小さなPNGを人が見て名前を付ける(cluster00.png -> H.png)。
           大きな天気図から番号で箱を探すより速い。
           1つだけ切り出したいときは `cut --index` も使える。

3. `match` 名前を付けたテンプレートを全画像に当てる。

使い方:
    # 1. 候補の規模を見る
    python -m scripts.extract_symbols scan --in-dir data/processed/jma --limit 20 \
        --overlay reports/symbols

    # 2. 似た形をまとめて、山ごとの代表を書き出す(scan が出した大きさで絞る)
    python -m scripts.extract_symbols cluster --in-dir data/processed/jma \
        --limit 5 --size 36 57 --out data/templates
    #    data/templates/cluster00.png などを見て、H なら H.png に付け替える

    # 3. 全画像に当てる
    python -m scripts.extract_symbols match --in-dir data/processed/jma \
        --templates data/templates --limit 20 --overlay reports/symbols
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.chartsymbols import (
    DEFAULT_BANDS,
    cluster_patches,
    correlation,
    crop_template,
    glyph_candidates,
    match_templates,
    patch_of,
    to_hsv,
)

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
        candidates = glyph_candidates(rgb, min_side=args.min_side,
                                      max_side=args.max_side, band=args.band,
                                      erode=args.erode)
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


def cmd_cluster(args):
    """似た形の候補をまとめ、山ごとの代表を小さなPNGとして書き出す。

    大きな天気図から番号を頼りに箱を探すより、山ごとの代表を並べて見るほうが
    速い。機械にはどの山がHでどれがLかは分からないので、人が見て名前を付ける。
    """
    collected = []      # (画像名, 番号, 候補, マスク)
    for path in iter_images(args.in_dir, args.limit):
        rgb = np.array(Image.open(path).convert("RGB"))
        mask = DEFAULT_BANDS[args.band].mask(to_hsv(rgb))
        for i, c in enumerate(glyph_candidates(rgb, min_side=args.min_side,
                                               max_side=args.max_side)):
            if args.size:
                w, h = args.size
                if abs(c.width - w) > args.tolerance or abs(c.height - h) > args.tolerance:
                    continue
            collected.append((path.name, i, c, mask))

    if not collected:
        raise SystemExit("候補がありません。--size の指定か --limit を見直すこと。")

    common = (args.patch_width, args.patch_height)
    patches = [patch_of(mask, c, common) for _, _, c, mask in collected]
    clusters = cluster_patches(patches, args.threshold)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 前回の山を消してから書く。条件を変えて実行し直すと山の数が減ることが
    # あり、消さないと前回の余りが残る。match はディレクトリの中を全部
    # 読むので、条件の違う山が混ざったまま照合してしまう。
    stale = sorted(out_dir.glob("cluster*.png"))
    for path in stale:
        path.unlink()
    if stale:
        print(f"前回の山 {len(stale)}個を消した ({out_dir}/)")

    print(f"候補 {len(collected)}個 -> {len(clusters)}個の山 (相関 {args.threshold} 以上でまとめた)")
    for rank, cluster in enumerate(clusters):
        if cluster["size"] < args.min_cluster:
            continue
        core = cluster["members"][0]
        _, _, candidate, mask = collected[core]
        template = mask[candidate.y0:candidate.y1, candidate.x0:candidate.x1]
        out_path = out_dir / f"cluster{rank:02d}.png"
        Image.fromarray((template * 255).astype(np.uint8)).save(out_path)
        where = ", ".join(
            f"{collected[m][0]}#{collected[m][1]}" for m in cluster["members"][:3])
        print(f"  cluster{rank:02d}.png  {cluster['size']:3d}個  "
              f"{template.shape[1]}x{template.shape[0]}  例: {where}")

    sheet = write_contact_sheet(clusters, collected, out_dir, args.min_cluster)
    if sheet:
        print(f"\n一覧: {sheet}  <- まずこれを開くと全部の山が1枚で見える")

    report_cluster_similarity(clusters, args.min_cluster)
    report_threshold_sweep(patches, args.threshold)

    print(f"\n{out_dir}/ の小さなPNGを見て、H や L だと分かったものを")
    print("その名前に付け替えること (例: cluster00.png -> H.png)。")
    print("付け替えたら match で全画像に当てる。数字や目盛の山は消してよい。")


def write_contact_sheet(clusters: list[dict], collected: list, out_dir: Path,
                        min_cluster: int, cell_height: int = 90) -> Path | None:
    """山の代表を1枚の画像に並べる。番号と個数を添える。

    山が多いと、小さなPNGを1枚ずつ開いて回るのが辛い。並べて一度に見れば、
    どれが記号でどれが数字かはすぐ分かる。実際、36x57の山は H でも L でも
    なく、気圧の数値の 0 と 9 だった。

    記号が白・地が黒のままだと読みにくいので、白地に黒で描く。
    """
    shown = [c for c in clusters if c["size"] >= min_cluster]
    if not shown:
        return None

    tiles = []
    for rank, cluster in enumerate(shown):
        _, _, candidate, mask = collected[cluster["members"][0]]
        patch = mask[candidate.y0:candidate.y1, candidate.x0:candidate.x1]
        # 記号を黒、地を白にして拡大する
        glyph = Image.fromarray(((~patch) * 255).astype(np.uint8)).convert("L")
        scale = cell_height / max(glyph.height, 1)
        glyph = glyph.resize((max(1, int(glyph.width * scale)), cell_height),
                             Image.NEAREST)
        tiles.append((f"{rank:02d} ({cluster['size']})", glyph))

    label_height = 16
    pad = 8
    width = sum(t.width + pad for _, t in tiles) + pad
    height = cell_height + label_height + pad * 2
    sheet = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.load_default()
    except OSError:                                    # 実行環境によっては無い
        font = None

    x = pad
    for label, tile in tiles:
        sheet.paste(tile.convert("RGB"), (x, pad))
        draw.text((x, pad + cell_height + 2), label, fill=(0, 0, 0), font=font)
        draw.rectangle((x - 1, pad - 1, x + tile.width, pad + cell_height),
                       outline=(180, 180, 180))
        x += tile.width + pad

    path = out_dir / "clusters.png"
    sheet.save(path)
    return path


def report_cluster_similarity(clusters: list[dict], min_cluster: int, top: int = 6) -> None:
    """山どうしがどれだけ似ているかを出す。同じ記号が割れていないか見るため。

    山が多く出たとき、それが「記号の種類が多い」のか「同じ記号がしきい値で
    割れた」のかは、山の数だけでは分からない。山の平均どうしの相関を見れば
    分かる。しきい値のすぐ下(0.6前後)なら、同じ記号が割れている。
    """
    shown = [c for c in clusters if c["size"] >= min_cluster][:top]
    if len(shown) < 2:
        return
    print(f"\n山どうしの似かた (1.0=同じ形):")
    print("        " + "".join(f"{i:8d}" for i in range(len(shown))))
    for i, a in enumerate(shown):
        cells = "".join(f"{correlation(a['mean'], b['mean']):8.2f}" for b in shown)
        print(f"cluster{i:02d}" + cells)
    print("0.6前後の組があれば、同じ記号がしきい値で割れている見込み。")
    print("0.2以下なら別の記号。")


# 記号の大半を覆えたとみなす割合。残りは汚れや、等圧線が重なった個体。
COVERAGE = 0.8


def report_threshold_sweep(patches: list, current: float) -> None:
    """しきい値を変えると山の数がどう変わるかを出し、選ぶべき値を示す。

    山の数だけを見ると、しきい値を下げれば必ず減るので判断できない。
    **候補の大半を覆うのに山がいくつ要るか**を見る。記号の種類は
    せいぜい数種類なので、その数が一番小さくなるところが答えになる。
    """
    print(f"\nしきい値ごとの山 (今は {current}):")
    best = None
    for threshold in (0.4, 0.5, 0.6, 0.7, 0.8):
        sizes = [c["size"] for c in cluster_patches(patches, threshold)]
        needed, covered = 0, 0
        for n in sizes:
            if covered >= COVERAGE * len(patches):
                break
            covered += n
            needed += 1
        head = ", ".join(str(n) for n in sizes[:6])
        more = f" ...計{len(sizes)}山" if len(sizes) > 6 else f" (計{len(sizes)}山)"
        mark = " <- 今" if abs(threshold - current) < 1e-9 else ""
        print(f"  {threshold:.1f}: {head}{more}  "
              f"{COVERAGE:.0%}を覆うのに{needed}山{mark}")
        # 同じ山数なら、しきい値は高いほうが安全(別の記号を混ぜにくい)
        if best is None or needed <= best[1]:
            best = (threshold, needed)

    if best:
        print(f"\n{COVERAGE:.0%}を一番少ない山で覆えるのは しきい値 {best[0]:.1f} "
              f"({best[1]}山)。記号は{best[1]}種類と見てよい。")
        if abs(best[0] - current) > 1e-9:
            print(f"  --threshold {best[0]:.1f} でやり直すと、名前を付けるPNGが{best[1]}枚に絞れる。")


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
        raise SystemExit(
            f"テンプレートがありません: {template_dir} (先に cluster か cut を実行)")

    # clusterNN のままのものが残っていたら、名前を付ける手順が抜けている。
    # そのまま当てても、どれがHでどれがLか分からない数が並ぶだけになる。
    unnamed = sorted(name for name in templates if name.startswith("cluster"))
    if unnamed:
        raise SystemExit(
            f"名前を付けていないテンプレートがあります: {', '.join(unnamed)}\n"
            f"{template_dir}/ の小さなPNGを見て、H なら H.png に、L なら L.png に\n"
            "付け替えてください。記号でない山(数字・目盛)は消してください。\n"
            "どの山がHでどれがLかは機械には分かりません。"
        )
    return templates


def cmd_match(args):
    templates = load_templates(args.templates)
    print("テンプレート: " + ", ".join(
        f"{k}({v.shape[1]}x{v.shape[0]})" for k, v in templates.items()))
    template_sizes = [(v.shape[1], v.shape[0]) for v in templates.values()]

    per_label = Counter()
    n_images = 0
    n_candidates = 0
    for path in iter_images(args.in_dir, args.limit):
        rgb = np.array(Image.open(path).convert("RGB"))
        hits = match_templates(rgb, templates, threshold=args.threshold)
        found = Counter(h.label for h in hits)
        per_label.update(found)
        n_images += 1
        # テンプレートと同じくらいの大きさの候補が、そもそも何個あったか
        n_candidates += sum(
            1 for c in glyph_candidates(rgb)
            if any(abs(c.width - w) <= 3 and abs(c.height - h) <= 3
                   for w, h in template_sizes)
        )
        detail = ", ".join(f"{k}={v}" for k, v in sorted(found.items())) or "なし"
        print(f"{path.name:24s} {detail}")
        if args.overlay:
            draw_boxes(rgb, hits, Path(args.overlay) / f"{path.stem}_match.png", False)

    total = sum(per_label.values())
    print("\n合計: " + ", ".join(f"{k}={v}" for k, v in sorted(per_label.items())))
    print(f"1枚あたり {total / n_images:.1f}個。"
          "天気図1枚の高気圧・低気圧はふつう2〜6個。")

    # 取りこぼしの見える化。同じ大きさの候補があるのに当たっていないなら、
    # 一致スコアの下限が厳しすぎるか、テンプレートが1個体に寄りすぎている。
    print(f"\n同じ大きさの候補: {n_candidates}個 / 当たった: {total}個 "
          f"({total / n_candidates:.0%})" if n_candidates else "")
    if n_candidates and total < 0.6 * n_candidates:
        print(f"取りこぼしが多い。--threshold を {args.threshold} から下げて試すこと。")
        print("テンプレートは山の代表1個体なので、同じ記号でも汚れ方が違うと外れる。")
    elif n_candidates and total > 1.4 * n_candidates:
        print("候補より多く当たっている。--threshold を上げること。")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="テンプレート無しで候補を数える")
    scan.add_argument("--in-dir", required=True)
    scan.add_argument("--limit", type=int, default=20)
    scan.add_argument("--overlay")
    scan.add_argument("--erode", type=int, default=0,
                      help="記号を数える前に何画素細らせるか。等圧線が記号の上を"
                           "横切って繋がっている場合に1か2を指定する")
    scan.add_argument("--band", default="isobar", choices=sorted(DEFAULT_BANDS),
                      help="記号を探す色。高気圧・低気圧の記号は色付きのことがある")
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

    cluster = sub.add_parser("cluster", help="似た形の候補をまとめ、代表を書き出す")
    cluster.add_argument("--in-dir", required=True)
    cluster.add_argument("--limit", type=int, default=5)
    cluster.add_argument("--out", default="data/templates")
    cluster.add_argument("--size", type=int, nargs=2, metavar=("W", "H"),
                         help="この大きさ付近の候補だけを対象にする。scan の出力から取る")
    cluster.add_argument("--tolerance", type=int, default=3)
    cluster.add_argument("--threshold", type=float, default=0.7,
                         help="同じ山とみなす相関の下限")
    cluster.add_argument("--min-cluster", type=int, default=2,
                         help="これ未満の山は書き出さない")
    cluster.add_argument("--erode", type=int, default=0,
                      help="記号を数える前に何画素細らせるか。等圧線が記号の上を"
                           "横切って繋がっている場合に1か2を指定する")
    cluster.add_argument("--band", default="isobar", choices=sorted(DEFAULT_BANDS),
                      help="記号を探す色。高気圧・低気圧の記号は色付きのことがある")
    cluster.add_argument("--min-side", type=int, default=6)
    cluster.add_argument("--max-side", type=int, default=64)
    cluster.add_argument("--patch-width", type=int, default=24)
    cluster.add_argument("--patch-height", type=int, default=32)
    cluster.set_defaults(func=cmd_cluster)

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
