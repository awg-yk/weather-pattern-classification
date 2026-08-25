"""ラベルの併用ルールを、既存のCSVに機械的に当てる。

同じ気圧配置に違うラベルが付いていると、モデルは矛盾した答えを要求されることに
なり、学習しようがない。実測では futatsudama_low(二つ玉低気圧)でこれが起きていた
-- 106件のうち56件は単独で付き、japan_sea_low と nankigan_low の両方を伴う例は
0件だったのに、片方だけを伴う例が9件あった。しかもその9件は2025年3月以降に集中して
おり、途中で基準が変わったことを示している。

判定そのもの(この天気図は二つ玉か)は人が決めるしかないが、
「二つ玉なら個別の低気圧ラベルは付けない」のような規約は機械的に当てられる。

逆向き(--add)もできる。二つ玉低気圧を「日本海低気圧と南岸低気圧の同時発生」を
指す合成ラベルとみなすなら、構成要素も併記されているべき、という立場になる。
どちらが正しいかは気象学的な定義の問題で、このスクリプトは決めない。

    # 何が変わるかだけ見る(既定。ファイルは書き換えない)
    python -m scripts.apply_label_rule --labels data/labels_v2.csv \\
        --when futatsudama_low --drop japan_sea_low nankigan_low

    # 逆に、構成要素を併記する側に寄せる
    python -m scripts.apply_label_rule --labels data/labels_v2.csv \\
        --when futatsudama_low --add japan_sea_low nankigan_low

    # 元を壊さず別ファイルに書き出す(規約を変えたときの影響を測るとき)
    python -m scripts.apply_label_rule --labels data/labels_v2.csv \\
        --when futatsudama_low --add japan_sea_low nankigan_low \\
        --out data/labels_v2_composite.csv
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd

from src.labels import LABEL_JA, LABELS, parse_labels


def apply_rule(labels: list, when: str, drop: list = None, add: list = None) -> list:
    """whenが付いている行に、dropの除去またはaddの追加を適用した結果を返す。"""
    if when not in labels:
        return labels
    result = [label for label in labels if label not in set(drop or ())]
    for label in add or ():
        if label not in result:
            result.append(label)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--when", required=True, help="このラベルが付いている行に適用する")
    parser.add_argument("--drop", nargs="+", default=None, help="取り除くラベル")
    parser.add_argument("--add", nargs="+", default=None,
                        help="付け足すラベル。--when を合成ラベルとみなす場合に使う")
    parser.add_argument("--out", default=None,
                        help="書き出し先。指定すると元のファイルを変更せず別ファイルに出す"
                        "(規約を変えたときの影響を測るときはこちら)")
    parser.add_argument(
        "--apply", action="store_true",
        help="実際に書き換える。指定しなければ、何が変わるかを表示するだけ",
    )
    args = parser.parse_args()

    if not args.drop and not args.add:
        raise SystemExit("--drop か --add のどちらかを指定してください")
    unknown = [l for l in [args.when, *(args.drop or []), *(args.add or [])] if l not in LABELS]
    if unknown:
        raise SystemExit(f"知らないラベルです: {unknown}\n使えるのは: {LABELS}")
    overlap = set(args.drop or []) & set(args.add or [])
    if overlap:
        raise SystemExit(f"--drop と --add に同じラベルは指定できません: {sorted(overlap)}")
    if args.when in (args.drop or []) or args.when in (args.add or []):
        raise SystemExit(f"--when と同じラベル({args.when})は --drop/--add に指定できません")

    path = Path(args.labels)
    df = pd.read_csv(path)
    if "label" not in df.columns:
        raise SystemExit(f"{path} に label 列がありません(列: {list(df.columns)})")

    changes = []
    new_values = []
    for filename, value in zip(df["filename"], df["label"]):
        before = parse_labels(value)
        after = apply_rule(before, args.when, args.drop, args.add)
        # 認識できないラベルは parse_labels が落とすので、変更が無い行は原文のまま残す
        if before == after:
            new_values.append(value)
            continue
        changes.append((filename, before, after))
        new_values.append("|".join(after))

    parts = []
    if args.drop:
        parts.append(f"{'・'.join(LABEL_JA[l] for l in args.drop)} を外す")
    if args.add:
        parts.append(f"{'・'.join(LABEL_JA[l] for l in args.add)} を付け足す")
    rule = f"{LABEL_JA[args.when]} が付いていたら " + "、".join(parts)
    print(f"規約: {rule}")
    print(f"対象: {path} ({len(df)}行)")
    if not changes:
        print("変わる行はありません。")
        return

    print(f"\n変わる行 {len(changes)}件:")
    for filename, before, after in changes:
        print(f"  {filename}")
        print(f"    前: {'|'.join(sorted(before))}")
        print(f"    後: {'|'.join(sorted(after))}")

    if args.out:
        # 元のファイルには触れず、別ファイルに出す。規約を変えたときに評価が
        # どれだけ動くかを、ラベルを壊さずに測るための経路。
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result = df.copy()
        result["label"] = new_values
        result.to_csv(out_path, index=False, encoding="utf-8")
        print(f"\n{len(changes)}件を変えたものを書き出しました: {out_path}")
        print(f"元の {path} は変更していません。")
        return

    if not args.apply:
        print(f"\n何も書き換えていません。"
              "実行するには --apply、別ファイルに出すには --out を付けてください。")
        return

    backup = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, backup)
    df["label"] = new_values
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"\n{len(changes)}件を書き換えました: {path}")
    print(f"元のファイルは {backup} に残しています。")


if __name__ == "__main__":
    main()
