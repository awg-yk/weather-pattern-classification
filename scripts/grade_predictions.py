"""モデルが出した判定に、人が○×を付ける。

■ 何のために

「人が見れば明らかに違うのに確信度が高い」を減らしたい、というのが出発点だった。
校正(src/calibration.py)は表示する%を実態に合わせるが、その「実態」は
保存済みのラベルで測ったものだった。本当に知りたいのは、**あなたの目で見て**
確信度90%の判定が何割正しいか である。

このツールは、scripts/classify_dates.py の出力を1件ずつ見せて○×を受け取り、
確信度の帯ごとに正解率を集計する。ラベルCSVもベストトラックも介さない、
直接の測定になる。

■ 注意: モデルの答えを見せるので、判断は引きずられる

○×方式は速いが、モデルの答えを先に見るぶん「そう言われればそう見える」方向に
寄りやすい。明らかな誤りを見つける用途では実害は小さいが、際どい事例の
一致率はやや高めに出る。引きずられたくない場合は --blind を使うこと。
天気図と日付だけを見せ、先に自分の答えを選んでから照合する。

■ 途中でやめられる

1件ごとに追記するので、q で中断して後日続きから再開できる。

使い方:
    python -m scripts.grade_predictions \\
        --predictions data/kazegawari_気圧配置_v6aa.csv \\
        --out data/grade_v6aa.csv

    # 集計だけ見る(判定済みのぶん)
    python -m scripts.grade_predictions --predictions ... --out ... --summary-only
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.labels import LABEL_JA, LABELS

BANDS = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]


def open_image(path: Path) -> None:
    """既定の画像ビューアで開く。開けなくても採点は続けられるようにする。"""
    try:
        if sys.platform == "win32":
            os.startfile(path)  # noqa: S606
        else:
            subprocess.run(["xdg-open" if sys.platform.startswith("linux") else "open", path],
                           check=False)
    except OSError as error:
        print(f"  (画像を開けませんでした: {error})")
        print(f"  手動で開いてください: {path}")


def band_of(confidence: float) -> str:
    for low, high in BANDS:
        if low <= confidence < high:
            return f"{low * 100:.0f}〜{min(high, 1.0) * 100:.0f}%"
    return "不明"


def summarise(graded: pd.DataFrame) -> None:
    """確信度の帯ごとに、人の目で見た正解率を出す。"""
    decided = graded[graded["判定"].isin(["正しい", "誤り"])].copy()
    if decided.empty:
        print("まだ採点された件がありません。")
        return
    decided["正解"] = (decided["判定"] == "正しい").astype(int)
    decided["帯"] = decided["確信度"].apply(band_of)

    print(f"\n{'=' * 62}")
    print(f"確信度と、人の目で見た正解率({len(decided)}件)")
    print("=" * 62)
    print(f"  {'確信度の帯':<14}{'件数':>6}{'平均確信度':>12}{'実際の正解率':>14}{'ズレ':>10}")
    order = [f"{l * 100:.0f}〜{min(h, 1.0) * 100:.0f}%" for l, h in BANDS]
    total_gap = 0.0
    for name in order:
        rows = decided[decided["帯"] == name]
        if rows.empty:
            continue
        mean_conf = rows["確信度"].mean()
        accuracy = rows["正解"].mean()
        gap = mean_conf - accuracy
        total_gap += len(rows) * abs(gap)
        flag = "  ←表示が高すぎる" if gap > 0.10 else ""
        print(f"  {name:<14}{len(rows):>6}{mean_conf * 100:>11.1f}%"
              f"{accuracy * 100:>13.1f}%{gap * 100:>+9.1f}pt{flag}")
    print(f"\n  全体の正解率: {decided['正解'].mean():.1%}")
    print(f"  ECE(表示%と実際のズレの平均): {total_gap / len(decided):.3f}")

    # 出発点の問題そのもの: 高い確信度で外している件
    confident_errors = decided[(decided["確信度"] >= 0.8) & (decided["正解"] == 0)]
    print(f"\n  確信度80%以上で誤っていた件: {len(confident_errors)}件"
          f"({len(confident_errors) / max(len(decided[decided['確信度'] >= 0.8]), 1):.1%})")
    if not confident_errors.empty:
        print("  --- その一覧(天気図を見直す候補) ---")
        for _, row in confident_errors.sort_values("確信度", ascending=False).iterrows():
            correct = row.get("正しいラベル", "")
            correct = f" → 正しくは {correct}" if isinstance(correct, str) and correct else ""
            print(f"    {row['日付']}  {row['気圧配置']} ({row['確信度'] * 100:.1f}%){correct}")

    unsure = graded[graded["判定"] == "わからない"]
    if not unsure.empty:
        print(f"\n  「わからない」{len(unsure)}件は集計から除いています。")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True,
                        help="scripts/classify_dates.py が書き出したCSV")
    parser.add_argument("--out", required=True, help="採点結果の書き出し先CSV")
    parser.add_argument("--date-column", default="発生日")
    parser.add_argument("--hour", type=int, default=0, choices=[0, 12],
                        help="classify_dates を動かしたときと同じ時刻を指定すること")
    parser.add_argument("--cache-dir", default="data/raw/date_lookup",
                        help="classify_dates と同じ場所。取得済みなら通信しない")
    parser.add_argument("--poppler-path", default=None)
    parser.add_argument("--per-band", type=int, default=None,
                        help="確信度の帯ごとに何件まで採点するか。既定は全件")
    parser.add_argument("--seed", type=int, default=42, help="並び順と抽出の乱数")
    parser.add_argument("--blind", action="store_true",
                        help="モデルの答えを先に見せない。自分の答えを選んでから照合する"
                        "(引きずられを避けたいとき)")
    parser.add_argument("--summary-only", action="store_true",
                        help="採点せず、これまでの結果の集計だけを表示する")
    args = parser.parse_args()

    out_path = Path(args.out)
    if args.summary_only:
        if not out_path.exists():
            raise SystemExit(f"{out_path} がありません。まず採点してください。")
        summarise(pd.read_csv(out_path))
        return

    preds = pd.read_csv(args.predictions)
    for column in (args.date_column, "気圧配置", "確信度"):
        if column not in preds.columns:
            raise SystemExit(
                f"列 '{column}' がありません。scripts/classify_dates.py の出力を渡してください。\n"
                f"  この表の列: {list(preds.columns)}"
            )
    preds = preds[preds["気圧配置"].notna() & (preds["気圧配置"] != "")]
    preds["確信度"] = pd.to_numeric(preds["確信度"], errors="coerce")
    preds = preds[preds["確信度"].notna()]

    rng = np.random.default_rng(args.seed)
    if args.per_band:
        preds["帯"] = preds["確信度"].apply(band_of)
        preds = (preds.groupby("帯", group_keys=False)
                 .apply(lambda g: g.sample(n=min(len(g), args.per_band),
                                           random_state=int(rng.integers(2**31)))))
    # 確信度順に並んでいると、高い帯が固まって現れて判断が偏る。混ぜる。
    preds = preds.sample(frac=1.0, random_state=int(rng.integers(2**31)))

    done = set(pd.read_csv(out_path)["日付"]) if out_path.exists() else set()
    remaining = [r for _, r in preds.iterrows() if r[args.date_column] not in done]
    if not remaining:
        print("すべて採点済みです。")
        summarise(pd.read_csv(out_path))
        return

    from scripts.fetch_and_predict import fetch_chart

    print(f"{len(remaining)}件を採点します(採点済み{len(done)}件)。")
    print("  ○=1(正しい) / ×=2(誤り) / u=わからない / s=とばす / q=中断して保存\n")

    for i, row in enumerate(remaining, start=1):
        date_str = str(row[args.date_column])
        try:
            image = fetch_chart(date_str, hour=args.hour, cache_dir=args.cache_dir,
                                poppler_path=args.poppler_path)
        except Exception as error:
            print(f"[{i}/{len(remaining)}] {date_str}: 天気図を取得できません({error})")
            continue

        print(f"[{i}/{len(remaining)}] {date_str}")
        open_image(Path(image))
        if not args.blind:
            print(f"  モデルの判定: {row['気圧配置']}(確信度 {row['確信度'] * 100:.1f}%)")
        else:
            print("  (モデルの判定は伏せています)")

        answer = ""
        while answer not in {"1", "2", "u", "s", "q"}:
            answer = input("  ○=1 / ×=2 / u / s / q > ").strip().lower()
        if answer == "q":
            print("中断しました。同じコマンドで続きから再開できます。")
            break
        if answer == "s":
            continue

        if args.blind:
            print(f"  モデルの判定は {row['気圧配置']}({row['確信度'] * 100:.1f}%) でした。")

        judgement = {"1": "正しい", "2": "誤り", "u": "わからない"}[answer]
        correct_label = ""
        if answer == "2":
            print("  正しい気圧配置は？(番号。分からなければ空欄でEnter)")
            for n, label in enumerate(LABELS, start=1):
                print(f"    {n}. {LABEL_JA[label]}")
            choice = input("  > ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(LABELS):
                correct_label = LABEL_JA[LABELS[int(choice) - 1]]

        record = pd.DataFrame([{
            "日付": date_str,
            "気圧配置": row["気圧配置"],
            "確信度": row["確信度"],
            "判定": judgement,
            "正しいラベル": correct_label,
        }])
        record.to_csv(out_path, mode="a", header=not out_path.exists(),
                      index=False, encoding="utf-8-sig")

    if out_path.exists():
        summarise(pd.read_csv(out_path))
        print(f"\n採点結果: {out_path}")


if __name__ == "__main__":
    main()
