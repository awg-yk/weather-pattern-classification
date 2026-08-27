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
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.chartsymbols import (
    DEFAULT_BANDS,
    cluster_patches,
    correlation,
    crop_template,
    glyph_candidates,
    touches_border,
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


def oversized_components(rgb, band: str, erode: int, max_side: int):
    """大きさで落とした連結成分の数と、最大のものの枠を返す。

    記号が等圧線と繋がっていると、図の端から端まで伸びる巨大な成分の一部に
    なって落ちる。落とした成分の最大が画像とほぼ同じ大きさなら、等圧線網が
    1つに繋がったままということで、細らせ方が足りない。
    """
    ink = DEFAULT_BANDS[band].mask(to_hsv(rgb))
    if erode:
        kernel = np.ones((2 * erode + 1, 2 * erode + 1), np.uint8)
        ink = cv2.erode(ink.astype(np.uint8), kernel).astype(bool)
    count, _, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), 8)
    rejected, biggest = 0, (0, 0)
    for i in range(1, count):
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        if max(w, h) > max_side:
            rejected += 1
            if w * h > biggest[0] * biggest[1]:
                biggest = (w, h)
    return rejected, biggest


def cmd_scan(args):
    counts = []
    oversize = []
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
        rejected, biggest = oversized_components(rgb, args.band, args.erode,
                                                 args.max_side)
        oversize.append((rejected, biggest))
        print(f"{path.name:24s} 候補 {len(candidates):4d}   "
              f"大きすぎて落とした成分 {rejected:3d} (最大 {biggest[0]}x{biggest[1]})")
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
    #
    # 番号付きのものだけを狙う。"cluster*.png" にすると一覧(clusters.png)も
    # 巻き込み、消して書き直す形になって「消した数」が1多く出る。
    stale = sorted(out_dir.glob("cluster[0-9][0-9].png"))
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
        stamp = datetime.fromtimestamp(sheet.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n一覧: {sheet.resolve()}")
        print(f"      {stamp} に書き出し ({sheet.stat().st_size:,} バイト)  "
              "<- まずこれを開くと全部の山が1枚で見える")
        print("      開いても古いままなら、画像表示ソフトの再読み込みか、"
              "エクスプローラの表示更新(F5)を試すこと。")

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
              f"({best[1]}山)。")
        print("  ただしこれは**字形の数**であって記号の数ではない。候補には気圧の数値の"
              "数字(0〜9で10種)・×印・等圧線の断片が混ざっている。")
        print("  H と L はそのうちの2つなので、一覧(clusters.png)を見て選ぶ。")


def cmd_grid(args):
    """座標のめもりを重ねた天気図を書き出す。cut --box に渡す数を読むため。

    等圧線と繋がった記号は連結成分として取れないので、候補の番号では
    指せない。画素の座標で切り出すしかなく、その座標を読むための道具。
    """
    rgb = np.array(Image.open(args.image).convert("RGB"))
    image = Image.fromarray(rgb).convert("RGB")
    draw = ImageDraw.Draw(image)
    height, width = rgb.shape[:2]
    try:
        font = ImageFont.load_default()
    except OSError:
        font = None

    for x in range(0, width, args.step):
        heavy = (x % (args.step * 5) == 0)
        draw.line((x, 0, x, height), fill=(0, 200, 0) if heavy else (150, 220, 150))
        if heavy:
            draw.text((x + 2, 2), str(x), fill=(0, 140, 0), font=font)
    for y in range(0, height, args.step):
        heavy = (y % (args.step * 5) == 0)
        draw.line((0, y, width, y), fill=(0, 200, 0) if heavy else (150, 220, 150))
        if heavy:
            draw.text((2, y + 2), str(y), fill=(0, 140, 0), font=font)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)
    print(f"{out.resolve()} ({width}x{height}画素、{args.step}画素ごとのめもり)")
    print("記号を囲む枠の左上と右下を読み、cut --box X0 Y0 X1 Y1 に渡すこと。")
    print("枠は記号より少し大きめでよい。多少の余白は一致スコアをあまり下げない。")


def cmd_cut(args):
    rgb = np.array(Image.open(args.image).convert("RGB"))
    if args.box:
        # 候補になっていない場所からでも切り出せる。等圧線と繋がってしまって
        # 連結成分として取れない記号は、この方法でテンプレートにするしかない。
        # 一度テンプレートさえ作れば、match は連結成分を使わずに探すので当たる。
        x0, y0, x1, y1 = args.box
        x0, y0 = max(0, x0 - args.pad), max(0, y0 - args.pad)
        x1 = min(rgb.shape[1], x1 + args.pad)
        y1 = min(rgb.shape[0], y1 + args.pad)
        if not (0 <= x0 < x1 <= rgb.shape[1] and 0 <= y0 < y1 <= rgb.shape[0]):
            raise SystemExit(
                f"--box が画像の外です。画像は {rgb.shape[1]}x{rgb.shape[0]} 画素。")
        template = crop_template(rgb, (x0, y0, x1, y1))
    else:
        candidates = glyph_candidates(rgb, band=args.band, erode=args.erode,
                                      max_side=args.max_side)
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
    edge = touches_border(template)
    if edge > 0.01:
        print(f"★縁の{edge:.1%}に太い線が掛かっている。記号が見切れている見込みが高い。")
        print("  --pad を増やして切り直すこと。部分だけのテンプレートは"
              "完全な記号にうまく当たらない。")
    else:
        print("縁に太い線は掛かっていない。記号は枠に収まっている。")


# cluster が書き出す一覧。テンプレートではないので読み込みから外す。
CONTACT_SHEET = "clusters"


def symbol_of(template_name: str) -> str:
    """テンプレート名から記号名を取り出す。H2 も H_b も「H」として数える。

    1つの個体から作ったテンプレートでは、同じ記号でも汚れ方や重なり方が
    違うと外れる。別の個体からもう1枚作って両方当てられるようにするための
    決まりで、末尾の数字と _ 以降を落とす。
    """
    return re.split(r"[_\d]", template_name, maxsplit=1)[0] or template_name


def load_templates(template_dir: Path) -> dict[str, np.ndarray]:
    """人が名前を付けたテンプレートだけを読む。

    cluster が書き出したものは clusterNN.png のままで、どれがHでどれがLか
    分からない。名前を付けたものが1つでもあればそちらだけを使い、番号のまま
    のものは黙って無視せず、無視したと伝える。1つも名前が付いていなければ
    手順が抜けているので止める。

    一覧(clusters.png)は山の代表を並べた見るための画像で、テンプレートでは
    ない。名前が cluster で始まるので、名指しで外す。
    """
    named: dict[str, np.ndarray] = {}
    unnamed: list[str] = []
    for path in sorted(Path(template_dir).glob("*.png")):
        if path.stem == CONTACT_SHEET:
            continue
        if re.fullmatch(r"cluster\d+", path.stem):
            unnamed.append(path.stem)
            continue
        named[path.stem] = np.array(Image.open(path).convert("L")) > 127

    if named:
        if unnamed:
            print(f"番号のままの山 {len(unnamed)}個は使いません "
                  f"({', '.join(unnamed[:5])}{'...' if len(unnamed) > 5 else ''})。")
            print(f"  消すなら: Remove-Item {template_dir}\\cluster[0-9][0-9].png")
        return named

    if unnamed:
        raise SystemExit(
            f"名前を付けていないテンプレートしかありません: {', '.join(unnamed)}\n"
            f"{template_dir}/ の clusters.png を見て、H なら H.png に、L なら L.png に\n"
            "付け替えてください。記号でない山(数字・目盛)は消してください。\n"
            "どの山がHでどれがLかは機械には分かりません。"
        )
    raise SystemExit(
        f"テンプレートがありません: {template_dir} (先に cluster か cut を実行)")


def report_scores(scores: list[float], threshold: float) -> None:
    """一致スコアの分布を出す。しきい値に張り付いていれば取りこぼしている。

    当たったものの最低スコアがしきい値のすぐ上なら、あと少しで届かなかった
    記号が下に埋もれている。テンプレートを増やすか、しきい値を下げる。
    """
    if not scores:
        return
    arr = np.array(scores)
    print(f"一致スコア: {arr.min():.2f} 〜 {arr.max():.2f} "
          f"(中央値 {np.median(arr):.2f}、しきい値 {threshold:.2f})")
    margin = arr.min() - threshold
    if margin < 0.05:
        print(f"★最低スコアがしきい値の{margin:+.2f}しかない。あと少しで届かなかった"
              "記号が埋もれている見込みが高い。")
        print("  取りこぼした記号を別個体としてテンプレートに足すこと"
              "(cut --box ... --name L3)。")


def report_angles(angles: list[float], angle_range: float) -> None:
    """当たった角度の分布を出す。範囲の端に張り付いていれば広げる合図。"""
    if not angles:
        return
    arr = np.array(angles)
    at_edge = int(np.count_nonzero(np.abs(np.abs(arr) - angle_range) < 1e-6))
    print(f"傾き: {arr.min():+.0f}度 〜 {arr.max():+.0f}度 "
          f"(中央値 {np.median(arr):+.0f}度)")
    if at_edge:
        print(f"★{at_edge}個が範囲の端({angle_range:+.0f}度)で当たっている。"
              f"--angle-range を広げると、まだ見つかる見込みがある。")


# 何割の枚数に出たら「毎回同じ場所」とみなすか。
FIXED_SHARE = 0.6

# 同じ場所とみなす相対座標の差。1453画素なら約7画素にあたる。
SAME_PLACE = 0.005


def report_fixed_detections(placed: list, n_images: int) -> None:
    """毎回ほぼ同じ場所・同じ傾きで出る検出を指摘する。

    高気圧や低気圧は日ごとに動く。**同じ画素に同じ傾きで出続けるものは
    気象ではなく、図郭や経緯度線を記号と読み違えた固定の誤検出**である
    可能性が高い。居座る高気圧でも中心は多少動くので、画素まで一致し続ける
    ことは少ない。

    `chart_palette.py` で色の帯に対して使ったのと同じ考え方を、検出に当てる。

    **これで分かるのは地図の備品との読み違えだけである。**等圧線は日ごとに
    動くので、等圧線を記号と読み違えた誤検出はここをすり抜ける。最後は
    重ね描きを目で見るしかない。
    """
    if n_images < 5 or not placed:
        return

    groups: list[dict] = []
    for symbol, cx, cy, angle in placed:
        for group in groups:
            if (group["symbol"] == symbol
                    and abs(group["cx"] - cx) < SAME_PLACE
                    and abs(group["cy"] - cy) < SAME_PLACE
                    and abs(group["angle"] - angle) < 1e-6):
                group["count"] += 1
                break
        else:
            groups.append({"symbol": symbol, "cx": cx, "cy": cy,
                           "angle": angle, "count": 1})

    fixed = [g for g in groups if g["count"] >= FIXED_SHARE * n_images]
    if not fixed:
        print("画素まで一致し続ける検出はない。地図の備品(図郭・経緯度線)を"
              "記号と読み違えてはいない。")
        print("  ただし等圧線は日ごとに動くので、等圧線を読み違えた誤検出は"
              "これでは分からない。重ね描きで確かめること。")
        return

    print(f"\n★毎回ほぼ同じ場所・同じ傾きで出ている検出 ({n_images}枚中):")
    for g in sorted(fixed, key=lambda g: -g["count"]):
        print(f"    {g['symbol']} 相対座標({g['cx']:.3f}, {g['cy']:.3f}) "
              f"{g['angle']:+.0f}度  {g['count']}枚")
    print("  高気圧・低気圧は日ごとに動くので、画素まで一致し続けるのは不自然。")
    print("  等圧線や図郭を記号と読み違えた固定の誤検出の見込みが高い。")
    print("  重ね描きでこの座標を見て、記号でなければテンプレートを見直すこと。")


def cmd_match(args):
    templates = load_templates(args.templates)
    print("テンプレート: " + ", ".join(
        f"{k}({v.shape[1]}x{v.shape[0]})" for k, v in templates.items()))
    by_symbol = Counter(symbol_of(k) for k in templates)
    if any(n > 1 for n in by_symbol.values()):
        print("  同じ記号の別個体: " + ", ".join(
            f"{k}={n}枚" for k, n in sorted(by_symbol.items()) if n > 1))
    template_sizes = [(v.shape[1], v.shape[0]) for v in templates.values()]
    angles = np.arange(-args.angle_range, args.angle_range + args.angle_step,
                       args.angle_step)
    print(f"テンプレートを {angles[0]:+.0f}度から{angles[-1]:+.0f}度まで "
          f"{args.angle_step}度刻みで回して当てる ({len(angles)}通り)。")

    per_label = Counter()
    all_angles: list[float] = []
    all_scores: list[float] = []
    placed: list[tuple[str, float, float, float]] = []   # (記号, cx, cy, 角度)
    n_images = 0
    n_candidates = 0
    for path in iter_images(args.in_dir, args.limit):
        rgb = np.array(Image.open(path).convert("RGB"))
        hits = match_templates(rgb, templates, threshold=args.threshold,
                               angles=angles)
        found = Counter(symbol_of(h.label) for h in hits)
        per_label.update(found)
        all_angles.extend(h.angle for h in hits)
        all_scores.extend(h.score for h in hits)
        placed.extend((symbol_of(h.label), h.cx, h.cy, h.angle) for h in hits)
        n_images += 1
        # テンプレートと同じくらいの大きさの候補が、そもそも何個あったか
        n_candidates += sum(
            1 for c in glyph_candidates(rgb)
            if any(abs(c.width - w) <= 3 and abs(c.height - h) <= 3
                   for w, h in template_sizes)
        )
        detail = ", ".join(f"{k}={v}" for k, v in sorted(found.items())) or "なし"
        tilt = ("  傾き " + ", ".join(f"{h.label}{h.angle:+.0f}度" for h in hits)
                if hits else "")
        print(f"{path.name:24s} {detail}{tilt}")
        if args.overlay:
            draw_boxes(rgb, hits, Path(args.overlay) / f"{path.stem}_match.png", False)

    total = sum(per_label.values())
    print("\n合計: " + ", ".join(f"{k}={v}" for k, v in sorted(per_label.items())))
    report_angles(all_angles, args.angle_range)
    report_scores(all_scores, args.threshold)
    report_fixed_detections(placed, n_images)
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
    cut.add_argument("--index", type=int, default=-1,
                     help="scan が振った候補の番号から切り出す")
    cut.add_argument("--box", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"),
                     help="画素の座標を直に指定して切り出す。等圧線と繋がっていて"
                          "候補にならない記号は、こちらで切り出す")
    cut.add_argument("--pad", type=int, default=0,
                     help="--box の四方に足す余白(画素)。記号が見切れるときに増やす。"
                          "余分な背景は一致スコアをあまり下げないので、多めでよい")
    cut.add_argument("--band", default="isobar", choices=sorted(DEFAULT_BANDS))
    cut.add_argument("--erode", type=int, default=0)
    cut.add_argument("--max-side", type=int, default=64)
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

    grid = sub.add_parser("grid", help="座標のめもりを重ねた天気図を書き出す")
    grid.add_argument("--image", required=True)
    grid.add_argument("--out", default="reports/grid.png")
    grid.add_argument("--step", type=int, default=50)
    grid.set_defaults(func=cmd_grid)

    match = sub.add_parser("match", help="テンプレートを当てる")
    match.add_argument("--in-dir", required=True)
    match.add_argument("--templates", default="data/templates")
    match.add_argument("--limit", type=int, default=20)
    match.add_argument("--angle-range", type=float, default=50.0,
                       help="テンプレートを何度まで回して当てるか。天気図の記号は"
                            "傾きが揃っていないので、回さないと当たらない")
    match.add_argument("--angle-step", type=float, default=5.0,
                       help="回す刻み。10度にすると一致スコアが0.74まで落ちるので5度が目安")
    match.add_argument("--threshold", type=float, default=0.8,
                       help="テンプレートの一致スコアの下限。誤検出が多ければ上げる")
    match.add_argument("--overlay")
    match.set_defaults(func=cmd_match)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
