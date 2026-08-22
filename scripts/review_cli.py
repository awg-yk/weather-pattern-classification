"""1つのラベルの「あり/なし」を、端末から判定する。ノートブックを使わない版。

scripts/label_tool.py の盲検レビューと同じことをするが、ipywidgets を必要としない。
天気図は既定の画像ビューアで開き、答えは端末で受け取る。少数の見直し(数十枚程度)なら
こちらのほうが早い。

label_tool.py と同じ約束を守る:
  - 元のラベルは表示しない。見えていると、以前の自分の答えに引きずられる。
  - 日付は表示する。季節が分からないと判断できないため。
  - 1枚ごとに追記するので、途中で止めても続きから再開できる。
  - 元のラベルCSVには一切書き込まない。突き合わせは別の工程。

使い方:
    python -m scripts.review_cli --images-dir <画像> --label futatsudama_low \\
        --candidates data/review_futatsudama_candidates.csv \\
        --out-csv data/review_futatsudama.csv
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.labels import LABEL_JA, LABELS

ANSWERS = {"y": "yes", "n": "no", "u": "unsure"}


def open_image(path: Path) -> None:
    """既定の画像ビューアで開く。開けなくても判定は続けられるようにする。"""
    try:
        if sys.platform == "win32":
            os.startfile(path)  # noqa: S606
        else:
            subprocess.run(["xdg-open" if sys.platform.startswith("linux") else "open", path],
                           check=False)
    except OSError as error:
        print(f"  (画像を開けませんでした: {error})")
        print(f"  手動で開いてください: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument(
        "--candidates", default=None,
        help="scripts/label_stats.py --out-csv が書いたCSV。kind列で絞り込む",
    )
    parser.add_argument(
        "--kind", default="needs_judgement",
        help="--candidates のうち、この kind の行だけを対象にする",
    )
    parser.add_argument("--filenames", nargs="+", default=None, help="対象を直接指定する")
    parser.add_argument(
        "--labels-csv", default=None,
        help="--years/--months で絞るときに読むラベルCSV。答えは書き込まない",
    )
    parser.add_argument("--years", type=int, nargs="+", default=None, help="対象の年")
    parser.add_argument(
        "--months", type=int, nargs="+", default=None,
        help="対象の月。付け忘れが疑われる月だけに絞れる"
        "(scripts/label_stats.py --by-month で見当を付ける)",
    )
    args = parser.parse_args()

    if args.label not in LABELS:
        raise SystemExit(f"知らないラベルです: {args.label}\n使えるのは: {LABELS}")

    if args.candidates:
        candidates = pd.read_csv(args.candidates)
        targets = candidates.loc[candidates["kind"] == args.kind, "filename"].tolist()
    elif args.filenames:
        targets = list(args.filenames)
    elif args.labels_csv:
        # 年・月で絞る。付け忘れを拾うには陰性も見る必要があるので、
        # そのラベルが付いている行だけに絞ったりはしない。
        frame = pd.read_csv(args.labels_csv)
        stamps = frame["filename"].str.extract(r"(\d{10})")[0]
        parsed = pd.to_datetime(stamps, format="%Y%m%d%H", errors="coerce")
        keep = parsed.notna()
        if args.years:
            keep &= parsed.dt.year.isin(set(args.years))
        if args.months:
            keep &= parsed.dt.month.isin(set(args.months))
        targets = frame.loc[keep, "filename"].tolist()
        if not targets:
            raise SystemExit(f"年{args.years}・月{args.months} に該当する行がありません")
    else:
        raise SystemExit(
            "対象の指定が必要です:\n"
            "  --candidates <CSV>            … label_stats --out-csv が書いた候補\n"
            "  --filenames <名前...>          … 直接指定\n"
            "  --labels-csv <CSV> --years/--months … 年・月で絞る"
        )

    images_dir = Path(args.images_dir)
    out_path = Path(args.out_csv)
    done = set(pd.read_csv(out_path)["filename"]) if out_path.exists() else set()
    remaining = [name for name in sorted(targets) if name not in done]

    missing = [name for name in remaining if not (images_dir / name).exists()]
    if missing:
        raise SystemExit(
            f"{images_dir} に{len(missing)}件の画像がありません(例: {missing[:3]})。\n"
            "--images-dir が正しいか、ファイル名の付け方が合っているか確認してください。"
        )

    if not remaining:
        print(f"対象{len(targets)}件はすべて判定済みです: {out_path}")
        return

    print(f"{LABEL_JA[args.label]} の判定: 残り{len(remaining)}件"
          f"(判定済み{len(done)}件)")
    print("  y = あり / n = なし / u = わからない / q = 中断(ここまでは保存済み)")
    print("  以前付けた答えは表示しません。天気図だけを見て判断してください。\n")

    for position, filename in enumerate(remaining, start=1):
        stamp = "".join(ch for ch in filename if ch.isdigit())[:10]
        print(f"[{position}/{len(remaining)}] {stamp[:4]}-{stamp[4:6]}-{stamp[6:8]} {stamp[8:10]}Z"
              f"  ({filename})")
        open_image(images_dir / filename)

        while True:
            answer = input("  y / n / u / q > ").strip().lower()
            if answer == "q":
                print(f"\n中断しました。ここまでの判定は {out_path} にあります。")
                return
            if answer in ANSWERS:
                break
            print("  y, n, u, q のいずれかを入力してください。")

        pd.DataFrame([{"filename": filename, "answer": ANSWERS[answer]}]).to_csv(
            out_path, mode="a", header=not out_path.exists(), index=False, encoding="utf-8"
        )

    print(f"\n完了しました: {out_path}")
    result = pd.read_csv(out_path)
    print(result["answer"].value_counts().to_string())


if __name__ == "__main__":
    main()
