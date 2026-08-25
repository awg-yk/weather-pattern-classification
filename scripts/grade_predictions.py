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
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.labels import LABEL_JA, LABELS, parse_labels

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


def summarise(graded: pd.DataFrame, against_labels=None, predictions_path=None) -> None:
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

    if against_labels:
        _compare_with_stored_labels(decided, against_labels)

    if predictions_path:
        meta_path = Path(predictions_path).with_suffix(".meta.json")
        if meta_path.exists():
            _report_corrected_precision(decided, meta_path)



def _compare_with_stored_labels(decided: pd.DataFrame, labels_csv: str) -> None:
    """あなたの○×と、保存済みラベルCSVの判定を突き合わせる。

    知りたいのは「あなたが○としたのに、ラベル上は不正解と数えられる」件数。
    それが多ければ、ラベルに正解の取りこぼしがあり、報告している F1 は
    実力より低く出ていることになる。
    """
    if "対象" not in decided.columns:
        print("\n  --against-labels は、ファイル名で採点した結果にのみ使えます"
              "(scripts/predict_charts.py の出力を採点した場合)。")
        return

    labels = pd.read_csv(labels_csv)
    labels["parsed"] = labels["label"].apply(parse_labels)
    stored = dict(zip(labels["filename"], labels["parsed"]))

    rows = decided[decided["対象"].isin(stored)].copy()
    if rows.empty:
        print(f"\n  {labels_csv} に、採点した天気図が1枚も見つかりませんでした。")
        return

    # モデルの1位ラベルが、保存済みのラベル集合に入っているか
    ja_to_key = {v: k for k, v in LABEL_JA.items()}
    rows["ラベル上は正解"] = [
        ja_to_key.get(name) in stored[f]
        for f, name in zip(rows["対象"], rows["気圧配置"])
    ]
    rows["人が○"] = rows["判定"] == "正しい"

    both = int((rows["人が○"] & rows["ラベル上は正解"]).sum())
    human_only = int((rows["人が○"] & ~rows["ラベル上は正解"]).sum())
    stored_only = int((~rows["人が○"] & rows["ラベル上は正解"]).sum())
    neither = int((~rows["人が○"] & ~rows["ラベル上は正解"]).sum())

    print(f"\n{'=' * 62}")
    print(f"あなたの○× と 保存済みラベル({Path(labels_csv).name})の食い違い({len(rows)}枚)")
    print("=" * 62)
    print(f"  {'':<20}{'ラベル上も正解':>16}{'ラベル上は不正解':>18}")
    print(f"  {'あなたが ○':<20}{both:>16}{human_only:>18}")
    print(f"  {'あなたが ×':<20}{stored_only:>16}{neither:>18}")

    print(f"\n  ラベルで測った正解率: {(both + stored_only) / len(rows):.1%}")
    print(f"  あなたの目での正解率  : {rows['人が○'].mean():.1%}")

    if human_only:
        share = human_only / len(rows)
        print(f"\n  ★ あなたは○なのに、ラベル上は不正解と数えられた: {human_only}枚({share:.1%})")
        print("     これがラベルの取りこぼし。報告しているF1は、この分だけ実力より")
        print("     低く出ていることになります。")
        print("\n     --- 該当する天気図(ラベルを見直す候補) ---")
        for _, row in rows[rows["人が○"] & ~rows["ラベル上は正解"]].head(20).iterrows():
            recorded = "|".join(LABEL_JA[l] for l in stored[row["対象"]])
            print(f"       {row['日付']}  モデル={row['気圧配置']}"
                  f"({row['確信度'] * 100:.0f}%) / 記録={recorded}")
    else:
        print("\n  あなたが○としたものは、すべてラベル上も正解でした。"
              "ラベルの取りこぼしは見つかりません。")

    if stored_only:
        print(f"\n  逆に、ラベル上は正解だがあなたが×とした: {stored_only}枚")
        print("     こちらは、以前のラベル付けが緩かった可能性を示します。")



def _report_corrected_precision(decided: pd.DataFrame, meta_path: Path) -> None:
    """誤検出だけを採点した結果から、適合率を組み直す。

    記録で測った適合率は「モデルが主張した件数のうち、記録にもあった割合」。
    誤検出のうち人が妥当と認めたものを正しい検出に数え直すと、本来の適合率になる。
    """
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    label_ja = LABEL_JA.get(meta["focus_label"], meta["focus_label"])
    tp, fp = meta["n_true_positive"], meta["n_false_positive"]
    graded = len(decided)
    accepted = int((decided["判定"] == "正しい").sum())

    print(f"\n{'=' * 62}")
    print(f"{label_ja} の適合率を、採点結果で組み直す")
    print("=" * 62)
    print(f"  モデルが主張した           : {tp + fp}枚")
    print(f"    記録にもあった           : {tp}枚")
    print(f"    記録に無い(誤検出扱い)   : {fp}枚")
    print(f"  そのうち採点した           : {graded}枚")
    print(f"    人が「妥当」とした       : {accepted}枚 ({accepted / graded:.1%})")

    stored_precision = tp / (tp + fp) if tp + fp else 0.0
    # 採点した割合から、誤検出全体のうち妥当なものの数を見積もる
    valid_rate = accepted / graded
    corrected = (tp + fp * valid_rate) / (tp + fp) if tp + fp else 0.0
    print(f"\n  記録で測った適合率         : {stored_precision:.3f}")
    print(f"  人の目で組み直した適合率   : {corrected:.3f}")

    if graded < fp:
        print(f"\n  ※ 誤検出{fp}枚のうち{graded}枚を採点した結果からの推定です。"
              "残りも同じ割合とみなしています。")
    if corrected - stored_precision > 0.1:
        print(f"\n  ★ 適合率が {corrected - stored_precision:+.3f} 動きます。"
              f"{label_ja} のF1は、報告値より高いことになります。")
        print("     ラベルの取りこぼしが、このラベルの評価を押し下げていました。")
    elif corrected - stored_precision < 0.05:
        print(f"\n  誤検出のほとんどは本当に誤りでした。"
              f"{label_ja} はモデルが弱いラベルで、改善対象として正しいことになります。")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True,
                        help="scripts/classify_dates.py が書き出したCSV")
    parser.add_argument("--out", required=True, help="採点結果の書き出し先CSV")
    parser.add_argument("--date-column", default="発生日")
    parser.add_argument(
        "--images-dir", default=None,
        help="scripts/predict_charts.py の出力(filename列を持つ)を採点するときに指定する。"
        "指定すると日付からの取得ではなく、このフォルダの画像を直接開く",
    )
    parser.add_argument(
        "--against-labels", default=None,
        help="集計時に、あなたの○×と保存済みラベルCSVを突き合わせる。"
        "「あなたは○だがラベル上は不正解」がラベルの取りこぼしにあたる",
    )
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
        summarise(pd.read_csv(out_path), args.against_labels, args.predictions)
        return

    preds = pd.read_csv(args.predictions)
    key = "filename" if args.images_dir else args.date_column
    for column in (key, "気圧配置", "確信度"):
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

    done = set(pd.read_csv(out_path)["対象"]) if out_path.exists() else set()
    remaining = [r for _, r in preds.iterrows() if str(r[key]) not in done]
    if not remaining:
        print("すべて採点済みです。")
        summarise(pd.read_csv(out_path), args.against_labels, args.predictions)
        return

    if not args.images_dir:
        from scripts.fetch_and_predict import fetch_chart

    print(f"{len(remaining)}件を採点します(採点済み{len(done)}件)。")
    print("  ○=1(正しい) / ×=2(誤り) / u=わからない / s=とばす / q=中断して保存\n")

    for i, row in enumerate(remaining, start=1):
        target = str(row[key])
        shown = str(row.get("日付", target))
        if args.images_dir:
            image = Path(args.images_dir) / target
            if not image.exists():
                print(f"[{i}/{len(remaining)}] {target}: 画像がありません")
                continue
        else:
            try:
                image = fetch_chart(target, hour=args.hour, cache_dir=args.cache_dir,
                                    poppler_path=args.poppler_path)
            except Exception as error:
                print(f"[{i}/{len(remaining)}] {target}: 天気図を取得できません({error})")
                continue

        print(f"[{i}/{len(remaining)}] {shown}")
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
            "対象": target,
            "日付": shown,
            "気圧配置": row["気圧配置"],
            "確信度": row["確信度"],
            "判定": judgement,
            "正しいラベル": correct_label,
        }])
        record.to_csv(out_path, mode="a", header=not out_path.exists(),
                      index=False, encoding="utf-8-sig")

    if out_path.exists():
        summarise(pd.read_csv(out_path), args.against_labels, args.predictions)
        print(f"\n採点結果: {out_path}")


if __name__ == "__main__":
    main()
