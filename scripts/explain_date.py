"""日付を1つ受け取り、その日の天気図・Grad-CAM・全ラベルの確信度を書き出す。

Colabのノートブックで手作業でやっていた「日付を入れて結果を見る」を、
そのままファイル出力にしたもの。GitHub Actions(.github/workflows/explain-date.yml)
から呼ぶことを想定している。

天気図の取得元は日付に応じて自動で切り替わる(scripts/fetch_and_predict.py)。
    2022-10-01以降 : 気象庁JSMAPアーカイブ(PDF配信。popplerが要る)
    それ以前        : 手動アーカイブ(weather-pattern-classification-data)

使い方:
    python -m scripts.explain_date --date 2025-08-10 --hour 0 \\
        --weights weights/model.pt --out-dir reports/explain
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pandas as pd

from scripts.fetch_and_predict import fetch_chart
from scripts.gradcam import gradcam_figure
from src import calibration as calib
from src.labels import LABEL_JA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--hour", type=int, default=0, choices=[0, 12],
                        help="UTC時刻。0Zは日本時間9時、12Zは21時")
    parser.add_argument("--weights", default="weights/model.pt")
    parser.add_argument("--out-dir", default="reports/explain")
    parser.add_argument("--top-k", type=int, default=3, help="Grad-CAMを描くラベル数")
    parser.add_argument("--no-regions", action="store_true",
                        help="「見るべき領域」の枠を重ねない(data/regions.csv)")
    parser.add_argument("--cache-dir", default="data/raw/date_lookup")
    parser.add_argument("--poppler-path", default=None,
                        help="Windowsでpopplerのbinフォルダを直接指定する場合")
    args = parser.parse_args()

    chart_path = fetch_chart(args.date, hour=args.hour, cache_dir=args.cache_dir,
                             poppler_path=args.poppler_path)
    print(f"天気図: {chart_path}")

    calibration = calib.load_for_weights_cli(args.weights)
    if not calibration.is_fitted:
        print("警告: 校正ファイルがありません。確信度は未校正の生の値です。")

    fig, ranked = gradcam_figure(
        str(chart_path), args.weights, top_k=args.top_k,
        apply_preprocess=True, calibration=calibration,
        show_regions=not args.no_regions,
    )

    rows = []
    for label, probability in ranked:
        threshold = calibration[label].threshold
        rows.append({
            "気圧配置": LABEL_JA[label],
            "ラベル": label,
            "確信度": round(float(probability), 4),
            "しきい値": round(float(threshold), 4),
            "判定": "○" if probability > threshold else "―",
        })
    table = pd.DataFrame(rows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.date}_{args.hour:02d}Z"
    figure_path = out_dir / f"{stem}_gradcam.png"
    table_path = out_dir / f"{stem}_confidence.csv"
    fig.savefig(figure_path, dpi=140, bbox_inches="tight")
    table.to_csv(table_path, index=False, encoding="utf-8-sig")

    print(f"\n{args.date} {args.hour:02d}Z(日本時間 {args.hour + 9:02d}時)")
    print(table.drop(columns=["ラベル"]).to_string(index=False))
    asserted = table[table["判定"] == "○"]["気圧配置"].tolist()
    print(f"\nしきい値を超えた気圧配置: {'、'.join(asserted) if asserted else 'なし'}")
    print(f"\n書き出し: {figure_path}\n          {table_path}")


if __name__ == "__main__":
    main()
