"""ラベル済みの天気図に対してモデルを走らせ、1枚ずつの判定を書き出す。

scripts/classify_dates.py は日付を受け取ってアーカイブから天気図を取りに行くが、
こちらは既に手元にある学習用の画像(data/processed 相当)をそのまま使う。
学習・評価に使ったのと同じ画像で、モデルの答えを1枚ずつ確認したいときのもの。

出力は scripts/grade_predictions.py --images-dir がそのまま読める形式。
正解ラベルは書き出さない —— 採点のときに見えてしまうと、判断が引きずられる。

leave-one-year-out で学習した重みを使う場合、--years にはその重みのテスト年だけを
指定すること。学習に使った年を混ぜると、実力より良い判定を採点することになる。

使い方:
    python -m scripts.predict_charts \\
        --images-dir ..\\weather-pattern-classification-data\\processed \\
        --labels data/labels_v2.csv --weights runs/v2_chart_spread/model_test2023.pt \\
        --years 2023 --limit 60 --out data/predictions_2023.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src import calibration as calib
from src.labels import INDEX_TO_LABEL, LABEL_JA
from src.model import load_model
from src.train import get_transforms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--labels", required=True, help="対象の天気図を選ぶために読む。答えは書き出さない")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--years", type=int, nargs="+", default=None,
                        help="leave-one-year-out の重みなら、その重みのテスト年だけを指定する")
    parser.add_argument("--months", type=int, nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None, help="この枚数だけ無作為に選ぶ")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--all-probabilities", action="store_true",
                        help="全ラベルの確信度も列として出力する")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_model(args.weights, map_location=device)
    model.to(device).eval()
    calibration = calib.load_for_weights_cli(args.weights)
    if not calibration.is_fitted:
        print("警告: 校正ファイルがありません。確信度は未校正の生の値です。")

    df = pd.read_csv(args.labels)
    stamps = pd.to_datetime(
        df["filename"].str.extract(r"(\d{10})")[0], format="%Y%m%d%H", errors="coerce"
    )
    keep = stamps.notna()
    if args.years:
        keep &= stamps.dt.year.isin(set(args.years))
    if args.months:
        keep &= stamps.dt.month.isin(set(args.months))
    df = df[keep].copy()
    df["日付"] = stamps[keep].dt.strftime("%Y-%m-%d %HZ")
    if df.empty:
        raise SystemExit(f"年{args.years}・月{args.months} に該当する天気図がありません。")

    if args.limit and args.limit < len(df):
        df = df.sample(n=args.limit, random_state=args.seed)
    print(f"対象: {len(df)}枚(入力解像度 {meta['image_size']})")

    images_dir = Path(args.images_dir)
    missing = [f for f in df["filename"] if not (images_dir / f).exists()]
    if missing:
        raise SystemExit(
            f"{images_dir} に{len(missing)}枚がありません(例: {missing[:3]})。"
            "--images-dir を確認してください。"
        )

    transform = get_transforms(train=False, image_size=meta["image_size"])
    records = []
    for i, row in enumerate(df.itertuples(), start=1):
        image = Image.open(images_dir / row.filename).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)[0].cpu().numpy()
        probs = calibration.probabilities(logits)
        best = int(np.argmax(probs))
        record = {
            "filename": row.filename,
            "日付": row.日付,
            "気圧配置": LABEL_JA[INDEX_TO_LABEL[best]],
            "ラベル": INDEX_TO_LABEL[best],
            "確信度": round(float(probs[best]), 4),
            "しきい値": round(float(calibration[INDEX_TO_LABEL[best]].threshold), 4),
        }
        if args.all_probabilities:
            for idx, name in INDEX_TO_LABEL.items():
                record[f"p_{name}"] = round(float(probs[idx]), 4)
        records.append(record)
        if i % 20 == 0:
            print(f"  {i}/{len(df)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n書き出しました: {out_path}")
    print("次: python -m scripts.grade_predictions --predictions "
          f"{out_path} --images-dir {args.images_dir} --out <採点結果.csv>")


if __name__ == "__main__":
    main()
