"""日付の一覧を受け取り、その日の天気図を取得して気圧配置を判定する。

事例日(風替わりが起きた日など)のリストに対して、モデルが最も高い確信度を
出した気圧配置を付け、パターンごとの件数を数える。

天気図は日付に応じて自動的に取得元が切り替わる:
    2022-10-01以降 : 気象庁JSMAPアーカイブ
    それ以前        : 手動アーカイブ(weather-pattern-classification-data)

一度取得した画像はキャッシュされるので、2回目以降は通信しない。

使い方:
    python -m scripts.classify_dates \
        --dates-csv 風替わり.csv --weights runs/loyo/model_test2023.pt \
        --out 風替わり_気圧配置.csv
"""

import argparse
import collections
from datetime import date
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from scripts.fetch_and_predict import fetch_chart
from scripts.preprocess_jma import DEFAULT_STAMP_BOX, autocrop_to_content, mask_stamp_box
from src.labels import INDEX_TO_LABEL, LABEL_JA
from src.model import load_model
from src.train import get_transforms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dates-csv", required=True, help="日付が入ったCSV")
    parser.add_argument("--date-column", default="発生日", help="日付が入っている列名")
    parser.add_argument(
        "--weights",
        required=True,
        nargs="+",
        help="重みファイル。複数渡すと予測確率を平均する"
        "(交差検証で作った各foldのモデルを平均すると、1つだけ使うより安定する)",
    )
    parser.add_argument("--out", required=True, help="結果を書き出すCSV")
    parser.add_argument(
        "--hour",
        type=int,
        default=0,
        choices=[0, 12],
        help="使う観測時刻(UTC)。0Zは日本時間9時、12Zは21時",
    )
    parser.add_argument(
        "--cache-dir",
        default="data/raw/date_lookup",
        help="取得した天気図の保存先。同じ日付を再実行しても通信しない",
    )
    parser.add_argument(
        "--poppler-path",
        default=None,
        help=r"popplerのbinフォルダ。2022-10-01以降の日付はPDFで配信されるため変換に必要"
        "(それ以前は手動アーカイブのJPEGなので不要)",
    )
    parser.add_argument(
        "--all-probabilities",
        action="store_true",
        help="全ラベルの確信度も列として出力する",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, image_size = [], None
    for path in args.weights:
        model, meta = load_model(path, map_location=device)
        model.to(device).eval()
        if image_size is None:
            image_size = meta["image_size"]
        elif meta["image_size"] != image_size:
            raise SystemExit(
                f"入力解像度が揃っていません({path} は {meta['image_size']}、"
                f"他は {image_size})。同じ条件で学習した重みを渡してください。"
            )
        if meta["num_features"]:
            raise SystemExit(
                f"{path} はERA5特徴量を前提にした重みです。"
                "このスクリプトは天気図の画像だけで判定するため使えません。"
            )
        models.append(model)
    transform = get_transforms(train=False, image_size=image_size)
    if len(models) == 1:
        print(f"モデル: {args.weights[0]}(入力解像度 {image_size})")
    else:
        print(f"モデル{len(models)}個の平均(入力解像度 {image_size}):")
        for path in args.weights:
            print(f"  - {path}")

    df = pd.read_csv(args.dates_csv)
    if args.date_column not in df.columns:
        raise SystemExit(f"列 '{args.date_column}' がありません。列: {list(df.columns)}")
    df[args.date_column] = pd.to_datetime(df[args.date_column], errors="coerce")
    df = df.dropna(subset=[args.date_column]).reset_index(drop=True)
    print(f"対象: {len(df)}件({df[args.date_column].min():%Y-%m-%d} 〜 {df[args.date_column].max():%Y-%m-%d})\n")

    records = []
    failures = []
    for i, stamp in enumerate(df[args.date_column], start=1):
        target = stamp.date()
        row = {args.date_column: target.isoformat()}
        try:
            image_path = fetch_chart(
                target.isoformat(),
                hour=args.hour,
                cache_dir=args.cache_dir,
                poppler_path=args.poppler_path,
            )
            image = Image.open(image_path).convert("RGB")
            image = mask_stamp_box(autocrop_to_content(image), DEFAULT_STAMP_BOX)
            tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = torch.stack(
                    [torch.sigmoid(m(tensor))[0].cpu() for m in models]
                ).mean(dim=0)

            best = int(torch.argmax(probs))
            label = INDEX_TO_LABEL[best]
            row["気圧配置"] = LABEL_JA[label]
            row["ラベル"] = label
            row["確信度"] = round(float(probs[best]), 4)
            if args.all_probabilities:
                for idx, name in INDEX_TO_LABEL.items():
                    row[f"p_{name}"] = round(float(probs[idx]), 4)
            print(f"[{i:>3}/{len(df)}] {target} -> {LABEL_JA[label]} ({probs[best] * 100:.1f}%)")
        except Exception as e:
            row["気圧配置"] = ""
            row["ラベル"] = ""
            row["確信度"] = ""
            row["備考"] = f"取得失敗: {e}"
            failures.append((target, str(e)))
            print(f"[{i:>3}/{len(df)}] {target} -> 取得できませんでした")
        records.append(row)

    out = pd.DataFrame(records)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Excelでそのまま開けるようにBOM付きUTF-8で書く
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n書き出しました: {out_path}")

    # ---- 集計 ----
    classified = out[out["ラベル"] != ""]
    counts = collections.Counter(classified["気圧配置"])
    print(f"\n{'=' * 46}\n気圧配置ごとの件数(最も確信度が高いもの)\n{'=' * 46}")
    total = len(classified)
    for name, count in counts.most_common():
        bar = "█" * round(count / total * 30)
        print(f"  {name:<22}{count:>4}件 ({count / total * 100:>5.1f}%) {bar}")
    print(f"  {'合計':<22}{total:>4}件")

    if failures:
        print(f"\n取得できなかった日付: {len(failures)}件")
        for target, reason in failures[:10]:
            print(f"  {target}: {reason[:90]}")
        if len(failures) > 10:
            print(f"  ... 他{len(failures) - 10}件")


if __name__ == "__main__":
    main()
