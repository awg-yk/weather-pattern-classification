"""Phase 2 の下見: 前線をアノテーション無しで、色だけで取れるか確かめる。

`docs/2026-08-26-detection-plan.md`「着手前に1時間で試すこと」の前半。
YOLO Segmentation に300枚の前線アノテーションを与える前に、色マスクだけで
前線の画素が取れるかを見る。取れるなら Phase 2 の手作業は要らないか、
少なくともYOLOの教師データの下地になる。

判断のしかた
------------
数字だけ見ても分からないので、必ず --overlay を付けて重ね描きを目で見ること。
そのうえで:

  合格   前線として数えた区間が、天気図の前線とほぼ一致している。
         海岸線が warm_front に混じっていない(overlap が 0 に近い)。
  不合格 海岸線・等圧線・凡例の色が前線に混じる。あるいは前線が細切れになり、
         1本の前線が10個以上の区間に割れる。

前線の有無はラベルとも突き合わせられる。stationary_front / front_passage の
付いた日に前線が出て、付いていない日に出ないなら、色マスクだけで
「前線が何本あるか」というPhase 3の特徴量が作れるということになる。

使い方:
    python -m scripts.extract_fronts --in-dir data/processed/jma --limit 20 \
        --overlay reports/fronts --labels data/labels_v2.csv
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from src.chartsymbols import (
    FRONT_BANDS,
    band_overlap,
    clean_mask,
    color_masks,
    label_auc,
    segments,
    stationary_mask,
)

# 測定値とラベルの対応を見るときの、最低限の件数。これを下回るとAUCが偶然で動く。
MIN_CASES_FOR_AUC = 3

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")

# 重ね描きの色(RGB)。もとの天気図の上に、拾えた画素だけを塗る。
OVERLAY_COLORS = {
    "warm_front": (255, 0, 0),
    "cold_front": (0, 0, 255),
    "occluded_front": (255, 0, 255),
    "stationary": (0, 160, 0),
}


def load_labels(path: Path | None) -> dict[str, set[str]]:
    if path is None:
        return {}
    with open(path, encoding="utf-8") as f:
        return {
            row["filename"]: set(filter(None, row["label"].split("|")))
            for row in csv.DictReader(f)
        }


def analyse(rgb: np.ndarray) -> dict:
    masks = color_masks(rgb)
    cleaned = {name: clean_mask(masks[name]) for name in FRONT_BANDS}
    stationary = clean_mask(stationary_mask(cleaned["warm_front"], cleaned["cold_front"]))

    result = {"masks": cleaned, "stationary": stationary, "segments": {}}
    for name in FRONT_BANDS:
        found = segments(cleaned[name], name)
        result["segments"][name] = found
    result["overlap"] = band_overlap({
        "warm_front": masks["warm_front"],
        "coastline": masks["coastline"],
        "isobar": masks["isobar"],
    })
    return result


def write_overlay(rgb: np.ndarray, result: dict, out_path: Path) -> None:
    canvas = rgb.copy()
    # 白く飛ばしてから塗ると、拾えなかった前線が抜けとして見える
    canvas = (canvas * 0.35 + 255 * 0.65).astype(np.uint8)
    for name in FRONT_BANDS:
        canvas[result["masks"][name]] = OVERLAY_COLORS[name]
    canvas[result["stationary"]] = OVERLAY_COLORS["stationary"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(out_path)


def report_label_agreement(measured: list[dict]) -> None:
    """測定値がラベルとどれだけ噛み合っているかをAUCで出す。

    色マスクが本物の前線を拾えているなら、stationary_front が付いた日の
    測定値は、付いていない日より大きくなるはず。学習を一切せずに
    ラベルを分けられるなら、その測定値は Phase 3 の特徴量になる。

    これは信号があるかを見るための目安であって、報告用の成績ではない。
    本評価は src/split.py と src/evaluate.py を通すこと。
    """
    measures = ["温暖区間", "寒冷区間", "閉塞区間", "停滞px"]
    all_labels = sorted({label for row in measured for label in row["labels"]})

    rows = []
    for label in all_labels:
        flags = [label in row["labels"] for row in measured]
        n_pos = sum(flags)
        if n_pos < MIN_CASES_FOR_AUC or len(flags) - n_pos < MIN_CASES_FOR_AUC:
            continue
        aucs = {m: label_auc([row[m] for row in measured], flags) for m in measures}
        best = max(measures, key=lambda m: abs(aucs[m] - 0.5))
        rows.append((label, n_pos, len(flags) - n_pos, aucs, best))

    if not rows:
        print(f"\n(ラベルとの突き合わせには、陽性・陰性が各{MIN_CASES_FOR_AUC}件以上要る)")
        return

    print("\n=== 測定値とラベルの対応 (AUC。0.5=当てずっぽう、1.0=完全に分離) ===")
    header = f"{'ラベル':22s} {'陽性':>4s} {'陰性':>4s}  " + "".join(f"{m:>9s}" for m in measures)
    print(header)
    for label, n_pos, n_neg, aucs, best in sorted(rows, key=lambda r: -abs(aucs_best(r))):
        cells = "".join(
            (f"{aucs[m]:8.3f}" + ("*" if m == best else " ")) for m in measures
        )
        print(f"{label:22s} {n_pos:4d} {n_neg:4d}  {cells}")
    print("* はそのラベルを最もよく分けた測定値。")
    print("0.5から離れているほど、学習なしでもそのラベルの手がかりになっている。")
    print("これは信号の有無を見る目安であって報告用の成績ではない"
          "(本評価は src/evaluate.py)。")


def aucs_best(row) -> float:
    """並べ替え用。そのラベルで最も0.5から離れたAUCの符号付きのずれ。"""
    _, _, _, aucs, best = row
    return aucs[best] - 0.5


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--in-dir", required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--overlay", help="重ね描きPNGの出力先ディレクトリ")
    parser.add_argument("--labels", help="data/labels_v2.csv。付いているラベルと突き合わせる")
    args = parser.parse_args()

    labels = load_labels(Path(args.labels) if args.labels else None)
    in_dir = Path(args.in_dir)
    paths = sorted(p for p in in_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise SystemExit(f"画像が見つかりません: {in_dir}")
    paths = paths[:args.limit]

    print(f"{'画像':24s} {'温暖':>6s} {'寒冷':>6s} {'閉塞':>6s} {'停滞':>6s}  "
          f"{'海岸線混入':>10s}  ラベル")
    totals = defaultdict(int)
    with_front = 0
    measured: list[dict] = []
    for path in paths:
        rgb = np.array(Image.open(path).convert("RGB"))
        result = analyse(rgb)
        counts = {
            name: sum(1 for s in result["segments"][name] if s.is_frontlike)
            for name in FRONT_BANDS
        }
        stationary_px = int(result["stationary"].sum())
        contamination = result["overlap"].get(("coastline", "warm_front"), 0)
        for name, n in counts.items():
            totals[name] += n
        if sum(counts.values()) or stationary_px:
            with_front += 1
        measured.append({
            "labels": labels.get(path.name, set()),
            "温暖区間": counts["warm_front"],
            "寒冷区間": counts["cold_front"],
            "閉塞区間": counts["occluded_front"],
            "停滞px": stationary_px,
        })
        tag = "|".join(sorted(labels.get(path.name, []))) if labels else ""
        print(f"{path.name:24s} {counts['warm_front']:6d} {counts['cold_front']:6d} "
              f"{counts['occluded_front']:6d} {stationary_px:6d}  {contamination:10d}  {tag}")

        if args.overlay:
            write_overlay(rgb, result, Path(args.overlay) / f"{path.stem}_fronts.png")

    if labels:
        report_label_agreement(measured)

    print(f"\n{len(paths)}枚中 {with_front}枚で前線らしい区間を検出。")
    print("区間の合計: " + ", ".join(f"{k}={v}" for k, v in sorted(totals.items())))
    if args.overlay:
        print(f"重ね描き: {args.overlay}/ — 必ず目で確認すること。")
    print("\n海岸線混入が0でない画像が多いなら、まず scripts/chart_palette.py で色を測り直すこと。")


if __name__ == "__main__":
    main()
