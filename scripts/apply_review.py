"""見直しの結果を、ラベルCSVに反映する。

scripts/review_cli.py と scripts/label_tool.py の盲検レビューは、答えを別のファイルに
書き、元のラベルには触れない。突き合わせて納得してから反映する、という順序を守るため。
この工程がそれにあたる。

    yes    … そのラベルを付ける。--replaces を指定すれば、置き換えられる側を外す
    no     … 何もしない
    unsure … 何もしない。判断がつかなかったものを機械的に倒すと、
             「決めなかった」という情報が失われる

使い方:
    # 何が変わるか見る(既定。ファイルは書き換えない)
    python -m scripts.apply_review --labels data/labels_v2.csv \\
        --review data/review_futatsudama.csv --label futatsudama_low \\
        --replaces japan_sea_low nankigan_low

    # 実際に書き換える(元のファイルは .bak に残す)
    python -m scripts.apply_review ... --apply
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd

from src.labels import LABEL_JA, LABELS, parse_labels


def updated_labels(labels: list, label: str, replaces: list) -> list:
    """yesと判定された行のラベルを、規約に合わせて作り直す。"""
    kept = [l for l in labels if l not in set(replaces) and l != label]
    return sorted([*kept, label])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--review", required=True, help="filename,answer 列を持つCSV")
    parser.add_argument("--label", required=True, help="yesのときに付けるラベル")
    parser.add_argument(
        "--replaces", nargs="*", default=[],
        help="--label を付けるときに外すラベル(置き換えの運用のとき)",
    )
    parser.add_argument("--apply", action="store_true", help="実際に書き換える")
    args = parser.parse_args()

    unknown = [l for l in [args.label, *args.replaces] if l not in LABELS]
    if unknown:
        raise SystemExit(f"知らないラベルです: {unknown}\n使えるのは: {LABELS}")

    review = pd.read_csv(args.review)
    for column in ("filename", "answer"):
        if column not in review.columns:
            raise SystemExit(f"{args.review} に {column} 列がありません")
    counts = review["answer"].value_counts().to_dict()
    accepted = set(review.loc[review["answer"] == "yes", "filename"])

    path = Path(args.labels)
    df = pd.read_csv(path)

    not_found = accepted - set(df["filename"])
    if not_found:
        raise SystemExit(
            f"{path} に無いファイル名が見直し結果に{len(not_found)}件あります"
            f"(例: {sorted(not_found)[:3]})。同じラベルCSVを見直したか確認してください。"
        )

    changes, new_values = [], []
    for filename, value in zip(df["filename"], df["label"]):
        before = parse_labels(value)
        if filename not in accepted:
            new_values.append(value)
            continue
        after = updated_labels(before, args.label, args.replaces)
        if sorted(before) == after:
            new_values.append(value)
            continue
        changes.append((filename, before, after))
        new_values.append("|".join(after))

    print(f"見直しの結果: " + " / ".join(f"{k} {v}件" for k, v in sorted(counts.items())))
    print(f"反映するのは yes の{len(accepted)}件だけです"
          f"(no・unsure は元のまま)")
    if args.replaces:
        print(f"規約: {LABEL_JA[args.label]} を付けるとき "
              f"{'・'.join(LABEL_JA[l] for l in args.replaces)} を外す")

    if not changes:
        print("\n変わる行はありません(既に反映済みの可能性があります)。")
        return

    print(f"\n変わる行 {len(changes)}件:")
    for filename, before, after in changes:
        print(f"  {filename}")
        print(f"    前: {'|'.join(sorted(before))}")
        print(f"    後: {'|'.join(after)}")

    if not args.apply:
        print("\n何も書き換えていません。実行するには --apply を付けてください。")
        return

    backup = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, backup)
    df["label"] = new_values
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"\n{len(changes)}件を書き換えました: {path}")
    print(f"元のファイルは {backup} に残しています。")


if __name__ == "__main__":
    main()
