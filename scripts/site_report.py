r"""指定した地点の周辺だけを見て、高気圧・低気圧が何個あったかを日付ごとに数える。

なぜこれが要るか
----------------
モデルは日本全体の天気図を見て気圧配置を判断するため、その地点に影響しない
遠方の気圧配置まで拾ってしまう。小松のおろし風167日の検証では、誤りのうち
17件がこれに当たった。ここで「地点の周辺に系があったか」を実測すれば、
その17件が本当に「近くに何も無いのに遠くを見ていた」のかを数字で言える。

**これは新しい分類器ではない。**学習し直しは伴わず、検出結果を距離で
絞り込んで数えるだけなので、macro F1 や1位正解率は一切変わらない。

使い方:
    # 1日だけ
    python -m scripts.site_report --site komatsu --date 2023-01-01

    # 日付の一覧(おろし風167日など)をまとめて
    python -m scripts.site_report --site komatsu ^
        --dates-csv data\oroshi_dates.csv --date-column 発生日 ^
        --out reports\komatsu_nearby.csv --workers 4

    # 判定結果と突き合わせる(classify_dates.py の出力など)
    python -m scripts.site_report --site komatsu --dates-csv ... ^
        --join reports\小松_気圧配置.csv --join-date-column 発生日 ^
        --out reports\komatsu_nearby.csv

**先に scripts/site_preview.py で地点の位置を目視確認すること。**
円がずれていれば、この集計はすべて別の場所を測ったものになる。
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

# **annotate_charts の worker をそのまま使う。**倍率の掛け方には
# 「文字のテンプレートにだけ letter_size を掛ける(画像側には掛けない)」
# という過去に踏んだ落とし穴があるので、書き写さず1か所から使う
from scripts.annotate_charts import _WORKER, _init_worker
from scripts.build_features import analyse_chart
from src.quicklook import DETECTION, MARKS_DIR, PROCESSED_DIR, TEMPLATES_DIR
from src.sites import get_site, summarize


def _detect_one(job) -> dict:
    """1枚から、地点の周辺にある系を数える。worker で動く。"""
    path, stamp, threshold, angle_range, angle_step = job
    detections, _report = analyse_chart(
        Path(path), _WORKER["letters"], _WORKER["marks"], _WORKER["scale"],
        threshold, angle_range, angle_step, _WORKER["mark_scale"],
        _WORKER["mark_radius"], _WORKER["letter_threshold"],
        overlay_dir=None, want_masks=True,
    )
    found = summarize(detections, _WORKER["site"])
    nearest = found["nearest"]
    return {
        "stamp": stamp,
        "filename": Path(path).name,
        "周辺の高気圧": found["n_high"],
        "周辺の低気圧": found["n_low"],
        "全体の高気圧": found["n_high_all"],
        "全体の低気圧": found["n_low_all"],
        "最も近い系": nearest["kind"] if nearest else "",
        "最も近い距離": round(nearest["distance"], 4) if nearest else "",
    }


def _init(site, *args):
    _init_worker(*args)
    _WORKER["site"] = site


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--site", required=True, help="data/sites.csv の地点名")
    parser.add_argument("--sites", default=None, help="既定は data/sites.csv")
    parser.add_argument("--radius", type=float, default=None,
                        help="半径を一時的に変える(CSVは書き換えない)")
    parser.add_argument("--date", default=None, help="YYYY-MM-DD。1日だけ調べる")
    parser.add_argument("--dates-csv", default=None, help="日付が入ったCSV")
    parser.add_argument("--date-column", default="発生日")
    parser.add_argument("--hour", type=int, default=0, choices=[0, 12])
    parser.add_argument("--images-dir", default=str(PROCESSED_DIR))
    parser.add_argument("--templates", default=str(TEMPLATES_DIR))
    parser.add_argument("--marks", default=str(MARKS_DIR))
    parser.add_argument("--out", default=None, help="結果を書き出すCSV")
    parser.add_argument("--join", default=None,
                        help="日付で突き合わせるCSV(classify_dates.py の出力など)")
    parser.add_argument("--join-date-column", default=None,
                        help="--join 側の日付の列名。既定は --date-column と同じ")
    parser.add_argument("--label-column", default="気圧配置",
                        help="--join 側の判定ラベルの列名。ラベル別の内訳に使う")
    parser.add_argument("--reuse", default=None,
                        help="前回の出力CSVを読み直して集計だけやり直す。"
                        "検出をもう一度回さないので数秒で終わる。"
                        "--join を後から足したいときに使う")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--scale", type=float, default=0.7)
    parser.add_argument("--letter-size", type=float, default=DETECTION["letter_size"])
    parser.add_argument("--threshold", type=float, default=DETECTION["detect_threshold"])
    parser.add_argument("--mark-scale", type=float, default=1.0)
    parser.add_argument("--angle-range", type=float, default=60.0)
    parser.add_argument("--angle-step", type=float, default=5.0)
    args = parser.parse_args()

    if not args.date and not args.dates_csv and not args.reuse:
        parser.error("--date か --dates-csv か --reuse を指定してください")

    site = get_site(args.site, args.sites)
    if args.radius is not None:
        site = type(site)(name=site.name, x=site.x, y=site.y,
                          radius=args.radius, note=site.note)
    print(f"地点: {site.name}  中心 ({site.x}, {site.y})  半径 {site.radius}")
    if site.note:
        print(f"  {site.note}")
    print("  ★位置は scripts/site_preview.py で目視確認済みであること\n")

    if args.reuse:
        result = pd.read_csv(args.reuse)
        needed = ["周辺の高気圧", "周辺の低気圧"]
        missing_columns = [c for c in needed if c not in result.columns]
        if missing_columns:
            raise SystemExit(
                f"{args.reuse} に列がありません: {missing_columns}\n"
                "  site_report.py が書き出したCSVを渡してください")
        print(f"前回の出力を読み直しました: {args.reuse}({len(result)}行)")
        print("  検出はやり直していません。集計と突き合わせだけ行います。\n")
        _finish(result, site, args)
        return

    from src.split import index_images_by_stamp

    index = index_images_by_stamp(args.images_dir)
    if not index:
        raise SystemExit(
            f"天気図が1枚もありません: {args.images_dir}\n"
            "  python -m scripts.preprocess_jma --in-dir data/raw/new_png "
            f"--out-dir {args.images_dir} で作れます")

    if args.date:
        dates = [str(args.date)]
    else:
        table = pd.read_csv(args.dates_csv)
        if args.date_column not in table.columns:
            raise SystemExit(
                f"列 '{args.date_column}' がありません。列: {list(table.columns)}")
        parsed = pd.to_datetime(table[args.date_column], errors="coerce").dropna()
        dates = [d.strftime("%Y-%m-%d") for d in parsed]
    print(f"対象: {len(dates)}日")

    jobs, missing = [], []
    for date in dates:
        stamp = f"{str(date).replace('-', '').replace('/', '')[:8]}{args.hour:02d}"
        path = index.get(stamp)
        if path is None:
            missing.append(date)
            continue
        jobs.append((str(path), stamp, args.threshold,
                     args.angle_range, args.angle_step))
    if missing:
        print(f"  天気図が無い日: {len(missing)}日(集計から外します)")
    print(f"  調べる: {len(jobs)}枚\n")
    if not jobs:
        raise SystemExit("調べられる天気図がありません")

    options = {"boxes": True, "fronts": False, "thickness": 3}
    init_args = (site, args.templates,
                 args.marks if args.marks and Path(args.marks).exists() else None,
                 args.scale, args.letter_size, args.mark_scale, 1.6, 0.55, options)

    rows = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                                 initargs=init_args) as pool:
            for done, row in enumerate(pool.map(_detect_one, jobs, chunksize=4), 1):
                rows.append(row)
                if done % 20 == 0 or done == len(jobs):
                    print(f"  {done}/{len(jobs)}", flush=True)
    else:
        _init(*init_args)
        for done, job in enumerate(jobs, 1):
            rows.append(_detect_one(job))
            if done % 20 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)}", flush=True)

    result = pd.DataFrame(rows)
    result.insert(0, args.date_column,
                  [f"{s[:4]}-{s[4:6]}-{s[6:8]}" for s in result["stamp"]])
    result = result.drop(columns=["stamp"])
    _finish(result, site, args)


def _finish(result, site, args):
    """集計・突き合わせ・書き出し。検出したときと --reuse の両方から呼ぶ。"""
    import pandas as pd

    if args.join:
        other = pd.read_csv(args.join)
        join_col = args.join_date_column or args.date_column
        if join_col not in other.columns:
            raise SystemExit(
                f"--join のCSVに列 '{join_col}' がありません。列: {list(other.columns)}")
        other[join_col] = pd.to_datetime(other[join_col], errors="coerce")
        other = other.dropna(subset=[join_col])
        other[join_col] = other[join_col].dt.strftime("%Y-%m-%d")
        result = result.merge(other, left_on=args.date_column, right_on=join_col,
                              how="left", suffixes=("", "_判定"))
        print(f"\n{args.join} と突き合わせました({len(other)}行)")

    # ---- 集計 ----
    near_total = result["周辺の高気圧"] + result["周辺の低気圧"]
    empty = int((near_total == 0).sum())
    print(f"\n{'=' * 52}\n{site.name} の周辺(半径 {site.radius})\n{'=' * 52}")
    print(f"  調べた日数            {len(result):>5}日")
    print(f"  周辺に系が1つも無い日 {empty:>5}日 "
          f"({empty / max(len(result), 1) * 100:.1f}%)")
    print(f"  周辺の高気圧 平均     {result['周辺の高気圧'].mean():>5.2f}個"
          f"(全体 {result['全体の高気圧'].mean():.2f}個)")
    print(f"  周辺の低気圧 平均     {result['周辺の低気圧'].mean():>5.2f}個"
          f"(全体 {result['全体の低気圧'].mean():.2f}個)")
    print("\n  ※**周辺に系が無い=誤り、ではない。**冬型は西の高気圧と東の低気圧の"
          "\n    配置で決まるので、地点の近くにあるのは混んだ等圧線であって中心では"
          "\n    ない。前線通過・停滞前線も、前線は中心ではないので0になる。"
          "\n    下のラベル別の内訳と併せて読むこと。"
          "\n    検出漏れの可能性もあるので、数枚は必ず目で確かめること"
          "(notebooks/predict_local.ipynb の show_site)。")

    if args.join and args.label_column in result.columns:
        # **ラベルごとに分けて初めて読める。**「周辺に系が無い」が多いラベルが
        # 冬型や前線なら当たり前、日本海低気圧なら怪しい、と読み分けられる
        labelled = result[result[args.label_column].notna()]
        if len(labelled):
            print(f"\n  ラベル別の内訳({args.label_column})")
            print(f"  {'ラベル':<24}{'日数':>6}{'周辺に系なし':>14}")
            print("  " + "-" * 46)
            grouped = labelled.groupby(args.label_column, sort=False)
            for name, rows in grouped:
                none_near = int(((rows["周辺の高気圧"] + rows["周辺の低気圧"]) == 0).sum())
                print(f"  {str(name):<24}{len(rows):>6}{none_near:>10}"
                      f"({none_near / len(rows) * 100:>3.0f}%)")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Excelでそのまま開けるようBOM付きUTF-8
        result.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"\n書き出しました: {out_path}")


if __name__ == "__main__":
    main()
