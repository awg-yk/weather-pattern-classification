r"""Phase 3: 天気図から検出を行い、分類用の特徴量CSVを作る。

`scripts/cv_features.py`(Phase 4)の入力になる。

    天気図 -> 色マスクで前線 / テンプレートで記号 -> 特徴量 -> CSV

記号のテンプレートは人が用意する。太い等圧線が記号の上を横切ると連結成分では
分けられないので、機械では切り出せない(`docs/2026-08-26-detection-prescreen.md`)。
`data/templates/` に `H.png` `H2.png` ... `L.png` `L2.png` ... と置く。

中心の印を使う場合
------------------
`--marks` に中心の印(×)のテンプレートを置いたディレクトリを渡すと、
**位置は印から、種別は文字から**取るようになる。×は中心そのものなので位置
として正しく、H/L の文字は形が安定していて種別として正しい。それぞれ得意な
ほうだけを使う。名前は自由でよい。

**印の形では高低を見分けない。**丸で囲んだ×(低気圧)を `circle_cross`
テンプレートで拾おうとすると失敗する ― 図の側の丸は大きさも太さもまちまち
なので、丸ごと当てるテンプレートはしきい値を割る一方、丸の内側の×はいつでも
完璧に当たる。結果として**低気圧がすべて高気圧として数えられた**
(実測: 1枚あたり高7.70 / 低0.20)。

印と文字は `--mark-radius`(既定0.08、相対座標)以内で近いものから順に
1対1で組にする。**この半径は当てずっぽうで決めない。**処理のあとに実測の
分布が出るので、それを見て決める。

組にならなかった文字は**中心が枠外の系**である(×が描かれない)。位置は
主張させず、n_edge_high / n_edge_low として数だけ数える。

印は文字と**別の倍率**で当てる(`--mark-scale`、既定1.0=原寸)。印は約31x31
しかなく、文字(約95x117)に効く 0.7 では22x22になって丸と×の細部が潰れる。

速さ
----
テンプレート12枚 x 角度25通りで、原寸だと1枚21秒(2432枚で14時間)かかる。
`--scale 0.7` で1枚7.8秒(同5.3時間)、`--workers` で割ればさらに縮む。
合成図では 0.7 でも検出数は変わらなかったが、0.5 では取りこぼしが出た。

縮小は**色で分けたあとの2値マスクに対して**行う。RGBのまま縮めると
海岸線の赤茶と黒が混ざり、色の切り分けが崩れる。

途中で止めても、同じ `--out` を指定すれば続きから再開する。

使い方:
    python -m scripts.build_features --in-dir data\processed\jma `
        --templates data\templates --out data\features.csv --workers 8
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import time

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from src.chartfeatures import (
    MARK_LETTER_RADIUS,
    ChartDetections,
    assign_marks_to_letters,
    feature_names,
    nearest_letter_distances,
    split_by_edge,
    to_row,
)
from src.chartsymbols import (
    DEFAULT_BANDS,
    FRONT_BANDS,
    clean_mask,
    color_masks,
    match_templates,
    segments,
    stationary_mask,
    to_hsv,
)
from src.regions import load_regions

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")

# 中心の印の名前は自由でよい。**印の形では高低を見分けない**ので、
# cross でも circle_cross でも同じ「中心の印」として扱い、種別はいちばん近い
# H / L の文字から取る(`src.chartfeatures.assign_marks_to_letters`)。

# 記号として扱う名前。これ以外が --templates に入っていると黙って無視される
LETTER_SYMBOLS = ("H", "L")

_WORKER = {}


def ink_image(rgb: np.ndarray, band: str = "isobar") -> np.ndarray:
    """色で分けたあとの2値マスクを、白地に黒のRGB画像として返す。

    縮小するのはこれに対して行う。RGBのまま縮めると海岸線の赤茶と黒が
    混ざり、色の切り分けが崩れる。
    """
    mask = DEFAULT_BANDS[band].mask(to_hsv(rgb))
    out = np.full(rgb.shape, 255, dtype=np.uint8)
    out[mask] = 0
    return out


def shrink(image: np.ndarray, scale: float) -> np.ndarray:
    if scale >= 1.0:
        return image
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def warn_about_misplaced_marks(templates: dict) -> None:
    """記号でない名前が --templates に入っていたら知らせる。

    中心の印を --templates に置くと、H と L しか見ないので**黙って無視される**。
    印は --marks に渡すこと。
    """
    from scripts.extract_symbols import symbol_of

    stray = sorted({symbol_of(name) for name in templates} - set(LETTER_SYMBOLS))
    if stray:
        print(f"★--templates に H/L 以外があります: {', '.join(stray)}")
        print("  これらは使われません。中心の印は --marks に渡してください。")
        print("  (印の名前は自由。種別はいちばん近い H/L の文字から取る)")


def load_templates_scaled(directory: Path, scale: float, quiet: bool = False) -> dict:
    """テンプレートを読み、画像と同じ倍率に縮める。

    `quiet` は並列の子側で使う。同じ知らせを人数ぶん繰り返すと、そのあとに
    出る検出数がスクロールで流れてしまう。
    """
    from scripts.extract_symbols import load_templates

    templates = load_templates(directory, quiet=quiet)
    if scale >= 1.0:
        return templates
    scaled = {}
    for name, template in templates.items():
        small = cv2.resize(template.astype(np.uint8), None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)
        scaled[name] = small.astype(bool)
    return scaled


def analyse_chart(path: Path, letters: dict, marks: dict, scale: float,
                  threshold: float, angle_range: float, angle_step: float,
                  mark_scale: float = 1.0,
                  mark_radius: float = MARK_LETTER_RADIUS) -> tuple:
    """1枚から検出結果を取り出す。位置はすべて相対座標(0〜1)。

    文字と印で倍率を変えられる。**印は文字よりずっと小さい**(印は約31x31、
    H/L の文字は約95x117)ので、文字に効く 0.7 は印には粗すぎる
    (印は22x22になり、丸と×の細部が潰れる)。既定では印は原寸で当てる。
    """
    from scripts.extract_symbols import symbol_of

    rgb = np.array(Image.open(path).convert("RGB"))

    # 前線は原寸のまま。色マスクは安いので縮める必要がない
    masks = color_masks(rgb)
    cleaned = {name: clean_mask(masks[name]) for name in FRONT_BANDS}
    front_segments = {name: segments(cleaned[name], name) for name in FRONT_BANDS}
    stationary = clean_mask(stationary_mask(cleaned["warm_front"], cleaned["cold_front"]))

    ink = ink_image(rgb)
    angles = np.arange(-angle_range, angle_range + angle_step, angle_step)

    def positions(templates: dict, image_scale: float) -> dict:
        if not templates:
            return {}
        # cx / cy は画像の幅・高さで割った相対座標なので、倍率が違っても比べられる
        hits = match_templates(shrink(ink, image_scale), templates,
                               threshold=threshold, angles=angles)
        grouped: dict = {}
        for hit in hits:
            grouped.setdefault(symbol_of(hit.label), []).append((hit.cx, hit.cy))
        return grouped

    letter_pos = positions(letters, scale)
    mark_pos = positions(marks, mark_scale)
    letter_groups = {"H": letter_pos.get("H", []), "L": letter_pos.get("L", [])}
    distances: list = []
    found_marks = 0
    orphans = 0

    if mark_pos:
        # 印は位置だけを担い、種別はいちばん近い文字から取る。
        # **印の形(ただの×か、丸で囲んだ×か)では見分けない。**理由は
        # `src/chartfeatures.assign_marks_to_letters` に書いてある。
        all_marks = [point for points in mark_pos.values() for point in points]
        found_marks = len(all_marks)
        # 文字が1つも無いと距離は空になる。**印の数とは別に数えること**
        distances = nearest_letter_distances(all_marks, letter_groups)
        typed, spare, unmatched = assign_marks_to_letters(
            all_marks, letter_groups, mark_radius)
        highs, lows = typed["H"], typed["L"]
        orphans = len(unmatched)
        # 組にならなかった文字 = 中心が枠外の系。位置は主張させず数だけ数える
        _, edge_highs = split_by_edge(spare["H"])
        _, edge_lows = split_by_edge(spare["L"])
    else:
        # 文字しか無い。文字の位置は中心ではないので、縁のものは位置を主張させない
        highs, edge_highs = split_by_edge(letter_groups["H"])
        lows, edge_lows = split_by_edge(letter_groups["L"])

    return ChartDetections(
        highs=highs, lows=lows, edge_highs=edge_highs, edge_lows=edge_lows,
        front_segments=front_segments, stationary_pixels=int(stationary.sum()),
    ), {"marks": found_marks, "orphan_marks": orphans, "distances": distances}


def _init_worker(template_dir, mark_dir, scale, mark_scale, mark_radius):
    _WORKER["letters"] = load_templates_scaled(Path(template_dir), scale, quiet=True)
    _WORKER["marks"] = (load_templates_scaled(Path(mark_dir), mark_scale, quiet=True)
                        if mark_dir else {})
    _WORKER["regions"] = load_regions()
    _WORKER["scale"] = scale
    _WORKER["mark_scale"] = mark_scale
    _WORKER["mark_radius"] = mark_radius


def _run_one(job) -> tuple:
    path, threshold, angle_range, angle_step = job
    detections, report = analyse_chart(
        Path(path), _WORKER["letters"], _WORKER["marks"], _WORKER["scale"],
        threshold, angle_range, angle_step, _WORKER["mark_scale"],
        _WORKER["mark_radius"],
    )
    report.update(
        high=len(detections.highs), low=len(detections.lows),
        edge_high=len(detections.edge_highs), edge_low=len(detections.edge_lows),
    )
    return Path(path).name, to_row(detections, _WORKER["regions"]), report


def already_done(out_path: Path) -> set:
    """途中まで書けているCSVから、済んだファイル名を読む。"""
    if not out_path.exists():
        return set()
    try:
        return set(pd.read_csv(out_path)["filename"])
    except Exception:
        return set()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--in-dir", required=True)
    parser.add_argument("--templates", default="data/templates",
                        help="H/L の文字のテンプレート")
    parser.add_argument("--marks", default=None,
                        help="中心の印(cross*.png / circled*.png)。位置の主役になる")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0, help="先頭から何枚まで(0で全部)")
    parser.add_argument("--scale", type=float, default=0.7,
                        help="H/L の文字を探すときの縮小率。0.7で速さ2.7倍、検出は変わらなかった")
    parser.add_argument("--mark-scale", type=float, default=1.0,
                        help="中心の印を探すときの縮小率。印は約31x31しかないので既定は原寸")
    parser.add_argument("--mark-radius", type=float, default=MARK_LETTER_RADIUS,
                        help="印と H/L の文字を組にする距離(相対座標)。"
                             "処理のあとに実測の分布が出るので、それを見て決める")
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--angle-range", type=float, default=60.0)
    parser.add_argument("--angle-step", type=float, default=5.0)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args()

    paths = sorted(p for p in Path(args.in_dir).iterdir()
                   if p.suffix.lower() in IMAGE_SUFFIXES)
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        raise SystemExit(f"画像が見つかりません: {args.in_dir}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = already_done(out_path)
    todo = [p for p in paths if p.name not in done]
    if done:
        print(f"{len(done)}件は書き出し済み。残り{len(todo)}件から再開する。")
    if not todo:
        print("すべて済んでいます。")
        return

    columns = ["filename"] + feature_names(load_regions())
    if not out_path.exists():
        out_path.write_text(",".join(columns) + "\n", encoding="utf-8")

    # テンプレートの検分は親で1度だけ。子でやると人数ぶん繰り返される
    letters = load_templates_scaled(Path(args.templates), args.scale)
    warn_about_misplaced_marks(letters)
    print(f"文字 {len(letters)}枚(縮小{args.scale})", end="")
    if args.marks:
        marks = load_templates_scaled(Path(args.marks), args.mark_scale)
        print(f"、印 {len(marks)}枚(縮小{args.mark_scale})", end="")
    print()

    jobs = [(str(p), args.threshold, args.angle_range, args.angle_step) for p in todo]
    print(f"{len(todo)}枚、縮小{args.scale}、{args.workers}並列で処理する。")

    started = time()
    written = 0
    # 何個拾えたかを数える。**印を入れて成績が落ちたときに、それが
    # 「印が当たっていない」せいなのかを、ここの数字で切り分けられる。**
    tally = {"high": 0, "low": 0, "edge_high": 0, "edge_low": 0,
             "marks": 0, "orphan_marks": 0}
    distances: list = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(args.templates, args.marks, args.scale, args.mark_scale,
                  args.mark_radius),
    ) as pool, open(out_path, "a", encoding="utf-8") as handle:
        for name, row, report in pool.map(_run_one, jobs, chunksize=4):
            distances.extend(report.pop("distances"))
            for key, value in report.items():
                tally[key] += value
            handle.write(name + "," + ",".join(_format(v) for v in row) + "\n")
            handle.flush()      # 途中で止めても、ここまでは残す
            written += 1
            if written % 20 == 0 or written == len(todo):
                elapsed = time() - started
                rate = elapsed / written
                left = rate * (len(todo) - written)
                print(f"  {written}/{len(todo)}  "
                      f"{rate:.1f}秒/枚  残り{left / 60:.0f}分", flush=True)

    print(f"\n書き出しました: {out_path.resolve()} ({len(done) + written}件)")
    report_counts(tally, distances, written, args)
    print("次はこれで交差検証する:")
    print(f"  python -m scripts.cv_features --features {args.out} "
          "--years 2023 2024 2025 --out runs\\cv_features")


def report_counts(tally: dict, distances: list, written: int, args) -> None:
    """1枚あたりの検出数と、印から文字までの距離の分布を出す。

    **印を入れて成績が落ちたときに、原因を推測ではなく数字で切り分けるため。**
    高と低の比が偏っていれば種別の付け方が壊れているし、印が文字と組に
    ならないなら --mark-radius が狭すぎる。
    """
    per_chart = max(1, written)
    print("1枚あたりの検出数(この回に処理したぶんだけ):")
    for key in ("high", "low", "edge_high", "edge_low", "marks", "orphan_marks"):
        print(f"  {key:12s} {tally[key] / per_chart:.2f}")

    if not args.marks:
        return

    found = (tally["high"] + tally["low"]) / per_chart
    if found < 1.0:
        print("★印がほとんど組になっていません。--mark-scale を上げるか "
              "--threshold を下げてください。")
    if tally["marks"] and tally["orphan_marks"] / tally["marks"] > 0.3:
        print("★印の3割以上が文字と組になっていません。--mark-radius が狭すぎるか、"
              "文字を取りこぼしています。")

    if distances:
        values = np.sort(np.array(distances))
        marks = [("中央値", 50), ("7割", 70), ("9割", 90), ("最大", 100)]
        line = "  ".join(
            f"{name} {np.percentile(values, q):.3f}" for name, q in marks
        )
        print(f"印から一番近い文字までの距離: {line}")
        print(f"  今の --mark-radius {args.mark_radius:.3f} で "
              f"{100 * (values <= args.mark_radius).mean():.0f}% が届く")


def _format(value) -> str:
    """NaN を空欄にせず 'nan' と書く。pandas が欠測として読み戻せる形。"""
    if isinstance(value, float) and value != value:
        return "nan"
    return repr(value) if isinstance(value, float) else str(value)


if __name__ == "__main__":
    main()
