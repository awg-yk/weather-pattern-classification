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

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.fetch_and_predict import fetch_chart
from scripts.preprocess_jma import DEFAULT_STAMP_BOX, autocrop_to_content, mask_stamp_box
from src.dataset import parse_labels
from src.labels import INDEX_TO_LABEL, LABEL_JA, LABELS
from src.model import load_model
from src.train import get_transforms


def _build_era5_model(features_csv: str, labels_csv: str) -> dict:
    """ERA5特徴量からラベルを予測するロジスティック回帰を、ラベル済みの年で学習する。

    天気図のCNNとは別に動かし、出力する確率だけを混ぜる(scripts/ensemble_era5.py と
    同じ考え方)。特徴ベクトルに連結する方式は、画像側1280次元に対しERA5側が
    26次元と差が大きく、効果が出なかった。
    """
    feats = pd.read_csv(features_csv)
    columns = [c for c in feats.columns if c not in ("filename", "datetime")]

    labels = pd.read_csv(labels_csv)
    labels["parsed"] = labels["label"].apply(parse_labels)
    labels = labels[labels["parsed"].apply(len) > 0]
    train = labels.merge(feats, on="filename", how="inner")
    if train.empty:
        raise SystemExit(
            "labels.csv とERA5特徴量が1件も結合できませんでした。"
            "filenameの形式が揃っているか確認してください。"
        )

    X = train[columns].to_numpy(dtype="float64")
    fitted = []
    for label in LABELS:
        y = train["parsed"].apply(lambda ls, l=label: l in ls).to_numpy(dtype=int)
        if y.sum() < 2 or y.sum() == len(y):
            fitted.append(None)   # 学習できないラベルは確率0を返す
            continue
        fitted.append(
            make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, class_weight="balanced"),
            ).fit(X, y)
        )

    def predict(vector):
        row = vector.reshape(1, -1)
        return np.array([0.0 if m is None else m.predict_proba(row)[0, 1] for m in fitted])

    # 日時(YYYYMMDDHH)で引けるようにしておく
    stamps = feats["filename"].str.extract(r"(\d{10})")[0]
    values = feats[columns].to_numpy(dtype="float64")
    by_time = {stamp: values[i] for i, stamp in enumerate(stamps) if isinstance(stamp, str)}

    return {"columns": columns, "rows": len(train), "predict": predict, "by_time": by_time}


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
        default="0",
        choices=["0", "12", "both"],
        help="使う観測時刻(UTC)。0Zは日本時間9時、12Zは21時。"
        "bothは両方を見て1つにまとめる(--combineで方法を選ぶ)",
    )
    parser.add_argument(
        "--combine",
        default="max",
        choices=["max", "mean"],
        help="--hour both のときの統合方法。"
        "max(既定)=確信度が高い方の時刻の判定をそのまま採用するので、"
        "結果は必ずどちらかの天気図が支持する気圧配置になる。"
        "mean=2時刻の確率を平均する。統計的には安定だが、両方で2位だったラベルが"
        "逆転し、どちらの天気図も1位に選ばなかった気圧配置が出ることがある",
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
        "--era5-features",
        default=None,
        help="scripts/era5_features.py の出力。対象日を含む全期間ぶん。"
        "指定するとERA5のロジスティック回帰を併用する(--labelsも必要)",
    )
    parser.add_argument(
        "--labels",
        default=None,
        help="ERA5側のモデルを学習するためのlabels.csv。--era5-featuresと一緒に使う",
    )
    parser.add_argument(
        "--blend-weight",
        type=float,
        default=0.25,
        help="ERA5側の重み。0=天気図のみ、1=ERA5のみ。"
        "既定の0.25は交差検証で選ばれた値(0.25/0.20/0.35)の代表値",
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

    era5 = None
    if args.era5_features:
        if not args.labels:
            raise SystemExit("--era5-features を使うには --labels も指定してください")
        era5 = _build_era5_model(args.era5_features, args.labels)
        print(f"ERA5併用: 特徴量{len(era5['columns'])}個 / "
              f"学習{era5['rows']}件 / 重み {args.blend_weight}")

    df = pd.read_csv(args.dates_csv)
    if args.date_column not in df.columns:
        raise SystemExit(f"列 '{args.date_column}' がありません。列: {list(df.columns)}")
    df[args.date_column] = pd.to_datetime(df[args.date_column], errors="coerce")
    df = df.dropna(subset=[args.date_column]).reset_index(drop=True)
    print(f"対象: {len(df)}件({df[args.date_column].min():%Y-%m-%d} 〜 {df[args.date_column].max():%Y-%m-%d})\n")

    def predict(target, hour):
        """1つの日時の天気図(と、あればERA5)から全ラベルの確率を返す。"""
        image_path = fetch_chart(
            target.isoformat(), hour=hour,
            cache_dir=args.cache_dir, poppler_path=args.poppler_path,
        )
        image = Image.open(image_path).convert("RGB")
        image = mask_stamp_box(autocrop_to_content(image), DEFAULT_STAMP_BOX)
        tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.stack([torch.sigmoid(m(tensor))[0].cpu() for m in models]).mean(dim=0)

        if era5 is None:
            return probs
        key = f"{target:%Y%m%d}{hour:02d}"
        if key not in era5["by_time"]:
            raise KeyError(
                f"{target} {hour:02d}Z のERA5特徴量がありません。"
                f"scripts/download_era5.py と era5_features.py を{target.year}年について実行してください。"
            )
        era5_probs = torch.tensor(era5["predict"](era5["by_time"][key]), dtype=probs.dtype)
        return (1 - args.blend_weight) * probs + args.blend_weight * era5_probs

    hours = [0, 12] if args.hour == "both" else [int(args.hour)]
    records = []
    failures = []
    disagreements = 0
    both_hours = 0
    third_label = 0   # 統合結果がどちらの時刻の判定とも違ってしまった件数
    for i, stamp in enumerate(df[args.date_column], start=1):
        target = stamp.date()
        row = {args.date_column: target.isoformat()}
        try:
            by_hour = {}
            errors = []
            for hour in hours:
                try:
                    by_hour[hour] = predict(target, hour)
                except Exception as e:
                    errors.append(f"{hour:02d}Z: {e}")
            if not by_hour:
                raise RuntimeError(" / ".join(errors))

            if len(by_hour) == 1:
                probs = next(iter(by_hour.values()))
            elif args.combine == "mean":
                probs = torch.stack(list(by_hour.values())).mean(dim=0)
            else:
                # 各時刻の最大確率を比べ、高い方の時刻の予測をそのまま使う
                probs = max(by_hour.values(), key=lambda p: float(p.max()))

            best = int(torch.argmax(probs))
            label = INDEX_TO_LABEL[best]

            note = ""
            if len(by_hour) == 2:
                both_hours += 1
                tops = {h: INDEX_TO_LABEL[int(torch.argmax(p))] for h, p in by_hour.items()}
                for hour, p in by_hour.items():
                    best_h = int(torch.argmax(p))
                    row[f"気圧配置_{hour:02d}Z"] = LABEL_JA[INDEX_TO_LABEL[best_h]]
                    row[f"確信度_{hour:02d}Z"] = round(float(p[best_h]), 4)
                if tops[0] != tops[12]:
                    disagreements += 1
                    row["備考"] = f"00Zと12Zで判定が異なる({LABEL_JA[tops[0]]} / {LABEL_JA[tops[12]]})"
                    note = "  ※" + row["備考"]
                if label not in tops.values():
                    third_label += 1
                    note += "  ⚠どちらの天気図も1位に選ばなかったラベル"
            elif hours == [0, 12]:
                row["備考"] = f"{'12' if 0 in by_hour else '00'}Zは取得できず片方のみ"
                note = "  ※" + row["備考"]
            row["気圧配置"] = LABEL_JA[label]
            row["ラベル"] = label
            row["確信度"] = round(float(probs[best]), 4)
            if args.all_probabilities:
                for idx, name in INDEX_TO_LABEL.items():
                    row[f"p_{name}"] = round(float(probs[idx]), 4)
            print(f"[{i:>3}/{len(df)}] {target} -> {LABEL_JA[label]} "
                  f"({probs[best] * 100:.1f}%){note}")
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

    if len(hours) == 2 and both_hours:
        rate = disagreements / both_hours
        print(f"\n00Zと12Zで判定が分かれた日: {disagreements}件 / {both_hours}件 ({rate * 100:.0f}%)")
        # モデルの1位正解率をaとすると、真の気圧配置が同じでも 2a(1-a) 程度は
        # 食い違う。この水準と比べないと「天気が変わった割合」と誤読してしまう。
        for accuracy in (0.65, 0.70, 0.75):
            print(f"    参考: 1位正解率が{accuracy:.0%}なら、"
                  f"両時刻の正解が同一でも約{2 * accuracy * (1 - accuracy) * 100:.0f}%は食い違う")
        print("  この割合の大部分はモデルの不安定さで説明され、"
              "天気が実際に変化した割合ではない。")
        if third_label:
            print(f"\n  ⚠ 統合結果がどちらの時刻の判定とも異なった日: {third_label}件"
                  f"({third_label / both_hours * 100:.0f}%)")
            print("    --combine mean で、両時刻とも2位だったラベルが逆転したもの。"
                  "--combine max なら必ずどちらかの判定になる。")

    if failures:
        print(f"\n取得できなかった日付: {len(failures)}件")
        for target, reason in failures[:10]:
            print(f"  {target}: {reason[:90]}")
        if len(failures) > 10:
            print(f"  ... 他{len(failures) - 10}件")


if __name__ == "__main__":
    main()
