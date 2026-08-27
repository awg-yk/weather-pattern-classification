"""Phase 3: 天気図から検出を行い、分類用の特徴量CSVを作る。

`scripts/cv_features.py`(Phase 4)の入力になる。

    天気図 -> 色マスクで前線 / テンプレートで記号 -> 特徴量 -> CSV

記号のテンプレートは人が用意する。太い等圧線が記号の上を横切ると連結成分では
分けられないので、機械では切り出せない(`docs/2026-08-26-detection-prescreen.md`)。
`data/templates/` に `H.png` `H2.png` ... `L.png` `L2.png` ... と置く。

中心の印を使う場合
------------------
`--marks` に `cross*.png`(ただの×=高気圧)と `circled*.png`(丸で囲んだ×=
低気圧・台風)を置いたディレクトリを渡すと、そちらを位置の主役に使う。
**×は中心そのものだが、H/L の文字は中心の近くに置かれたラベル**なので、
位置としては×のほうが正しい。

ただし**中心が枠外の系には×が描かれず文字だけになる**。その分は H/L から
拾い、位置を主張しない特徴量(n_edge_high / n_edge_low)として数える。

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

from src.chartfeatures import ChartDetections, feature_names, split_by_edge, to_row
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

# 中心の印のテンプレート名の決まり。
#
# 名前の末尾の数字と _ 以降は落として数えるので(`symbol_of`)、
# cross / cross2 / cross_b は「cross」に、circle_cross / circled / circle2 は
# 「circle」で始まる名前になる。**丸で囲んだ×は circle で始まる名前にすること。**
MARK_HIGH = "cross"        # ただの× = 高気圧
MARK_LOW_PREFIX = "circle"  # 丸で囲んだ× = 低気圧・台風

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
        print(f"  (高気圧は {MARK_HIGH}、低気圧は {MARK_LOW_PREFIX} で始まる名前)")


def load_templates_scaled(directory: Path, scale: float) -> dict:
    """テンプレートを読み、画像と同じ倍率に縮める。"""
    from scripts.extract_symbols import load_templates

    templates = load_templates(directory)
    if scale >= 1.0:
        return templates
    scaled = {}
    for name, template in templates.items():
        small = cv2.resize(template.astype(np.uint8), None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)
        scaled[name] = small.astype(bool)
    return scaled


def analyse_chart(path: Path, letters: dict, marks: dict, scale: float,
                  threshold: float, angle_range: float, angle_step: float) -> ChartDetections:
    """1枚から検出結果を取り出す。位置はすべて相対座標(0〜1)。"""
    from scripts.extract_symbols import symbol_of

    rgb = np.array(Image.open(path).convert("RGB"))

    # 前線は原寸のまま。色マスクは安いので縮める必要がない
    masks = color_masks(rgb)
    cleaned = {name: clean_mask(masks[name]) for name in FRONT_BANDS}
    front_segments = {name: segments(cleaned[name], name) for name in FRONT_BANDS}
    stationary = clean_mask(stationary_mask(cleaned["warm_front"], cleaned["cold_front"]))

    small = shrink(ink_image(rgb), scale)
    angles = np.arange(-angle_range, angle_range + angle_step, angle_step)

    def positions(templates: dict) -> dict:
        if not templates:
            return {}
        hits = match_templates(small, templates, threshold=threshold, angles=angles)
        grouped: dict = {}
        for hit in hits:
            grouped.setdefault(symbol_of(hit.label), []).append((hit.cx, hit.cy))
        return grouped

    letter_pos = positions(letters)
    mark_pos = positions(marks)

    if mark_pos:
        # 中心の印があるならそちらが位置の主役。文字は枠外の系を拾うのに使う
        highs = mark_pos.get(MARK_HIGH, [])
        lows = [p for name, points in mark_pos.items()
                if name.startswith(MARK_LOW_PREFIX) for p in points]
        _, edge_highs = split_by_edge(letter_pos.get("H", []))
        _, edge_lows = split_by_edge(letter_pos.get("L", []))
    else:
        # 文字しか無い。文字の位置は中心ではないので、縁のものは位置を主張させない
        highs, edge_highs = split_by_edge(letter_pos.get("H", []))
        lows, edge_lows = split_by_edge(letter_pos.get("L", []))

    return ChartDetections(
        highs=highs, lows=lows, edge_highs=edge_highs, edge_lows=edge_lows,
        front_segments=front_segments, stationary_pixels=int(stationary.sum()),
    )


def _init_worker(template_dir, mark_dir, scale):
    _WORKER["letters"] = load_templates_scaled(Path(template_dir), scale)
    warn_about_misplaced_marks(_WORKER["letters"])
    _WORKER["marks"] = load_templates_scaled(Path(mark_dir), scale) if mark_dir else {}
    _WORKER["regions"] = load_regions()
    _WORKER["scale"] = scale


def _run_one(job) -> tuple:
    path, threshold, angle_range, angle_step = job
    detections = analyse_chart(
        Path(path), _WORKER["letters"], _WORKER["marks"], _WORKER["scale"],
        threshold, angle_range, angle_step,
    )
    return Path(path).name, to_row(detections, _WORKER["regions"])


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
                        help="記号を探すときの縮小率。0.7で速さ2.7倍、検出は変わらなかった")
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

    jobs = [(str(p), args.threshold, args.angle_range, args.angle_step) for p in todo]
    print(f"{len(todo)}枚、縮小{args.scale}、{args.workers}並列で処理する。")

    started = time()
    written = 0
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(args.templates, args.marks, args.scale),
    ) as pool, open(out_path, "a", encoding="utf-8") as handle:
        for name, row in pool.map(_run_one, jobs, chunksize=4):
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
    print("次はこれで交差検証する:")
    print(f"  python -m scripts.cv_features --features {args.out} "
          "--years 2023 2024 2025 --out runs\\cv_features")


def _format(value) -> str:
    """NaN を空欄にせず 'nan' と書く。pandas が欠測として読み戻せる形。"""
    if isinstance(value, float) and value != value:
        return "nan"
    return repr(value) if isinstance(value, float) else str(value)


if __name__ == "__main__":
    main()
