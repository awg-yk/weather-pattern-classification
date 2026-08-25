"""ラベルごとに「見るべき領域」をどれだけ見ているかを測る。

モデルがそのラベルを判断するときのGrad-CAMを、data/regions.csv の矩形と
突き合わせる。天気図を1枚ずつ目で見て「オホーツク海高気圧と答えたのに
本州を見ている」と指摘していたものが、ラベル別の1つの数字になる。

測る対象は2通りある。

    --on record    記録にそのラベルが付いている天気図(既定)
                   「正解のときに正しい場所を見ているか」
    --on predicted モデルがそのラベルを主張した天気図(確信度がしきい値超え)
                   「そう答えたとき、どこを見て答えたか」。誤検出も含むので、
                   適合率が低いラベルの原因を見るのはこちら

読み方
------
lift が1に近いラベルは、画像全体に一様に注目しているのと変わらない
= その気圧配置を位置で捉えられていない。lift が大きいほど良い。
mass だけを見てはいけない -- 矩形が広いラベル(西高東低など)は
何もしなくても mass が高く出る。

使い方:
    python -m scripts.attention_check \\
        --images-dir ..\\weather-pattern-classification-data\\processed \\
        --labels data/labels_v2.csv --weights weights/model.pt \\
        --years 2025 --out reports/attention_2025.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from scripts.gradcam import GradCAM, _load_model
from src import calibration as calib
from src.labels import LABELS, LABEL_JA, parse_labels
from src.regions import attention_mass, load_regions, peak_in_region
from src.train import get_transforms


def select_rows(labels_csv, years=None, months=None, limit=None, seed=42) -> pd.DataFrame:
    """対象の天気図を選ぶ。scripts/predict_charts.py と同じ絞り込み方。"""
    frame = pd.read_csv(labels_csv)
    stamps = pd.to_datetime(
        frame["filename"].str.extract(r"(\d{10})")[0], format="%Y%m%d%H", errors="coerce"
    )
    keep = stamps.notna()
    if years:
        keep &= stamps.dt.year.isin(set(years))
    if months:
        keep &= stamps.dt.month.isin(set(months))
    frame = frame[keep].copy()
    frame["日付"] = stamps[keep].dt.strftime("%Y-%m-%d %HZ")
    if frame.empty:
        raise SystemExit(f"年{years}・月{months} に該当する天気図がありません。")
    if limit and limit < len(frame):
        frame = frame.sample(n=limit, random_state=seed)
    return frame


def summarize(records: pd.DataFrame, regions: dict) -> pd.DataFrame:
    """1枚ごとの結果を、ラベル別の表にまとめる。"""
    rows = []
    for label, group in records.groupby("ラベル"):
        region = regions[label]
        mass = float(group["mass"].mean())
        rows.append({
            "ラベル": label,
            "気圧配置": LABEL_JA[label],
            "枚数": len(group),
            "mass": round(mass, 3),
            "area": round(region.area, 3),
            "lift": round(mass / region.area, 2),
            "peak的中率": round(float(group["peak"].mean()), 3),
            "確信度": round(float(group["確信度"].mean()), 3),
        })
    return pd.DataFrame(rows).sort_values("lift").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--regions", default=None, help="既定は data/regions.csv")
    parser.add_argument("--on", choices=("record", "predicted"), default="record",
                        help="record=記録にラベルがある天気図、predicted=モデルが主張した天気図")
    parser.add_argument("--only", nargs="+", default=None, choices=list(LABELS),
                        help="このラベルだけ測る。省略すると領域が定義されている全ラベル")
    parser.add_argument("--years", type=int, nargs="+", default=None,
                        help="leave-one-year-out の重みなら、その重みのテスト年だけを指定する")
    parser.add_argument("--months", type=int, nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None, help="この枚数だけ無作為に選ぶ")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None, help="1枚ごとの結果を書き出すCSV")
    args = parser.parse_args()

    regions = load_regions(args.regions)
    wanted = [label for label in (args.only or LABELS) if label in regions]
    if not wanted:
        raise SystemExit("測れるラベルがありません。data/regions.csv を確認してください。")
    skipped = [label for label in (args.only or LABELS) if label not in regions]
    if skipped:
        print(f"領域が未定義なので飛ばします: {[LABEL_JA[s] for s in skipped]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = _load_model(args.weights, device)
    gradcam = GradCAM(model)
    calibration = calib.load_for_weights_cli(args.weights)
    if not calibration.is_fitted:
        print("警告: 校正ファイルがありません。--on predicted のしきい値は一律0.5になります。")

    frame = select_rows(args.labels, args.years, args.months, args.limit, args.seed)
    images_dir = Path(args.images_dir)
    missing = [f for f in frame["filename"] if not (images_dir / f).exists()]
    if missing:
        raise SystemExit(
            f"{images_dir} に{len(missing)}枚がありません(例: {missing[:3]})。"
            "--images-dir を確認してください。"
        )
    print(f"対象: {len(frame)}枚(入力解像度 {meta['image_size']}、判定基準 --on {args.on})")

    transform = get_transforms(train=False, image_size=meta["image_size"])
    records = []
    for i, row in enumerate(frame.itertuples(), start=1):
        image = Image.open(images_dir / row.filename).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)[0].cpu().numpy()
        probs = calibration.probabilities(logits)

        if args.on == "record":
            targets = [label for label in parse_labels(row.label) if label in wanted]
        else:
            targets = [
                label for label in wanted
                if probs[LABELS.index(label)] > calibration[label].threshold
            ]

        for label in targets:
            cam = gradcam.generate(tensor, LABELS.index(label))
            region = regions[label]
            records.append({
                "filename": row.filename,
                "日付": row.日付,
                "ラベル": label,
                "気圧配置": LABEL_JA[label],
                "確信度": round(float(probs[LABELS.index(label)]), 4),
                "mass": round(attention_mass(cam, region), 4),
                "peak": bool(peak_in_region(cam, region)),
            })
        if i % 50 == 0:
            print(f"  {i}/{len(frame)}")

    if not records:
        raise SystemExit("測る対象がありませんでした。--on や --years を確認してください。")

    per_image = pd.DataFrame(records)
    table = summarize(per_image, regions)
    print()
    print(table.drop(columns=["ラベル"]).to_string(index=False))
    print()
    print("lift = mass / area。1なら画像全体に一様に注目しているのと同じで、"
          "その気圧配置を位置で捉えられていない。大きいほど良い。")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        per_image.to_csv(out_path, index=False, encoding="utf-8-sig")
        summary_path = out_path.with_name(out_path.stem + "_summary.csv")
        table.to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"\n1枚ごと: {out_path}\nラベル別: {summary_path}")


if __name__ == "__main__":
    main()
