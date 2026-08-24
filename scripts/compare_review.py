"""見直しの結果と元のラベルを突き合わせ、判断の揺れの大きさを測る。

同じ人が同じ天気図を、答えを伏せた状態で2回判定したときの一致率は、
そのラベルの「決めやすさ」を表す。人間でも揺れるなら、モデルがその揺れを
超えて正解し続けることは原理的にできない。つまりこの一致率が、
達成可能なF1のおおよその上限になる。

一致率だけでなくCohenのκも出す。オホーツク海高気圧のように陽性が少ない
ラベルでは、両方「なし」と答えるだけで一致率が9割を超えてしまい、
一致率だけでは実態が見えないため。κは偶然の一致を差し引いた値。

使い方:
    python -m scripts.compare_review --review data/review_okhotsk.csv \
        --labels data/labels.csv --label okhotsk_high
"""

import argparse
from pathlib import Path

import pandas as pd

from src.labels import LABEL_JA, LABELS, parse_labels


def cohen_kappa(a, b) -> float:
    """2値の一致度。0=偶然と同じ、1=完全一致。"""
    n = len(a)
    observed = (a == b).mean()
    # 偶然だけで一致する確率
    expected = sum((a == v).mean() * (b == v).mean() for v in (0, 1))
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", required=True, help="run_binary_review_session の出力")
    parser.add_argument("--labels", required=True)
    parser.add_argument(
        "--baseline-review",
        default=None,
        help="比較相手を別の見直し結果にする(既定はlabels.csv)。"
        "2回目と3回目のように、見直しどうしを比べたいときに使う",
    )
    parser.add_argument(
        "--universe-months",
        type=int,
        nargs="+",
        default=None,
        help="比較の対象範囲を月で指定する。範囲内にあって見直しに出てこない画像は"
        "「なし」として扱う。選抜された候補だけで比べるとκが意味を失うため、"
        "元の範囲全体に戻して比べるときに使う",
    )
    parser.add_argument("--label", default="okhotsk_high")
    parser.add_argument(
        "--apply",
        default=None,
        help="見直し結果を反映したlabels.csvの書き出し先。指定しなければ集計のみで、元のファイルは変更しない",
    )
    args = parser.parse_args()

    if args.label not in LABELS:
        raise SystemExit(f"--label は {LABELS} のいずれかを指定してください")

    review = pd.read_csv(args.review)
    labels = pd.read_csv(args.labels)
    labels["parsed"] = labels["label"].apply(parse_labels)
    labels["original"] = labels["parsed"].apply(lambda ls: int(args.label in ls))

    if args.baseline_review:
        baseline = pd.read_csv(args.baseline_review)
        baseline = baseline[baseline["answer"].isin(["yes", "no"])]
        labels["original"] = labels["filename"].map(
            dict(zip(baseline["filename"], (baseline["answer"] == "yes").astype(int)))
        )
        # 比較相手に出てこない画像は「なし」として扱う
        labels["original"] = labels["original"].fillna(0).astype(int)
        base_name = Path(args.baseline_review).name
    else:
        base_name = Path(args.labels).name

    if args.universe_months:
        # 範囲内にあって--reviewに出てこない画像も「なし」として加える。
        # 候補だけで比べると陽性ばかりの集合になり、κが意味を失う。
        months = {int(m) for m in args.universe_months}
        month_of = pd.to_numeric(
            labels["filename"].str.extract(r"\d{4}(\d{2})")[0], errors="coerce"
        )
        universe = labels[month_of.isin(months)]
        answered = dict(zip(review["filename"], review["answer"]))
        review = pd.DataFrame({
            "filename": universe["filename"],
            "answer": [answered.get(f, "no") for f in universe["filename"]],
        })
        print(f"比較範囲を{sorted(months)}月の{len(review)}枚に広げました"
              f"(見直しに出てこない画像は「なし」として扱います)")

    merged = review.merge(labels[["filename", "original"]], on="filename", how="inner")
    unsure = merged[merged["answer"] == "unsure"]
    decided = merged[merged["answer"].isin(["yes", "no"])].copy()
    decided["review"] = (decided["answer"] == "yes").astype(int)

    name = LABEL_JA[args.label]
    print(f"{'=' * 56}\n{name} の判断の揺れ\n{'=' * 56}")
    print(f"  比較相手            : {base_name}")
    print(f"  見直した枚数        : {len(merged)}枚"
          f"(うち「わからない」{len(unsure)}枚は集計から除外)")
    if decided.empty:
        raise SystemExit("集計できる判定がありません。")

    agree = (decided["review"] == decided["original"]).mean()
    kappa = cohen_kappa(decided["original"].to_numpy(), decided["review"].to_numpy())
    print(f"  一致率              : {agree:.1%}")
    positive_rate = decided["original"].mean()
    if positive_rate > 0.7 or positive_rate < 0.02:
        print("  ⚠ 比較対象の陽性率が偏っているため、κと一致率は解釈できません。")
        print("    候補だけを抜き出した集合ではこうなります。"
              "--universe-months で元の範囲に戻して測り直してください。")
    print(f"  Cohenのκ            : {kappa:.3f}"
          f"({'ほぼ完全' if kappa > 0.8 else 'かなり高い' if kappa > 0.6 else '中程度' if kappa > 0.4 else '低い'})")

    both = ((decided["original"] == 1) & (decided["review"] == 1)).sum()
    only_first = ((decided["original"] == 1) & (decided["review"] == 0)).sum()
    only_second = ((decided["original"] == 0) & (decided["review"] == 1)).sum()
    neither = ((decided["original"] == 0) & (decided["review"] == 0)).sum()

    print(f"\n  {'':<16}{'今回 あり':>12}{'今回 なし':>12}")
    print(f"  {'比較相手 あり':<16}{both:>12}{only_first:>12}")
    print(f"  {'比較相手 なし':<16}{only_second:>12}{neither:>12}")

    # 1回目を正解とみなしたときの2回目のF1。同じ人の2回の判定なので、
    # モデルがこれを超えることは期待できない = 実質的な上限。
    if both:
        precision = both / (both + only_second)
        recall = both / (both + only_first)
        ceiling = 2 * precision * recall / (precision + recall)
        print(f"\n  人間どうしのF1(達成可能な上限の目安): {ceiling:.3f}")
        print(f"    適合率 {precision:.3f} / 再現率 {recall:.3f}")
    else:
        print("\n  両方が「あり」とした事例が無いため、上限は推定できません。")

    disagreed = decided[decided["review"] != decided["original"]]
    if not disagreed.empty:
        print(f"\n--- 判断が変わった{len(disagreed)}枚 ---")
        for _, row in disagreed.head(30).iterrows():
            direction = "なし→あり" if row["review"] else "あり→なし"
            print(f"  {row['filename']:<28}{direction}")
        if len(disagreed) > 30:
            print(f"  ... 他{len(disagreed) - 30}枚")

    if args.apply:
        # 見直した行だけ、2回目の判定でラベルを差し替える
        answer = dict(zip(decided["filename"], decided["review"]))
        updated = 0
        emptied = []

        def rewrite(row):
            nonlocal updated
            if row["filename"] not in answer:
                return row["label"]
            current = list(row["parsed"])
            want = answer[row["filename"]]
            has = args.label in current
            if want == has:
                return row["label"]
            if want:
                current.append(args.label)
            else:
                current.remove(args.label)
            updated += 1
            if not current:
                # このラベルだけが付いていた行。外すとラベルが無くなるので、
                # 学習から除外される。改めて付け直す必要がある。
                emptied.append(row["filename"])
                return "unclassified"
            return "|".join(current)

        labels["label"] = labels.apply(rewrite, axis=1)
        out_path = Path(args.apply)
        labels.drop(columns=["parsed", "original"]).to_csv(out_path, index=False)
        print(f"\n{updated}行を更新して書き出しました: {out_path}")
        print("  元のlabels.csvは変更していません。内容を確認してから差し替えてください。")
        if emptied:
            print(f"\n  ⚠ {len(emptied)}行は、このラベルを外した結果ラベルが無くなり"
                  "unclassifiedになりました。")
            print("    学習から除外されてしまうため、別のラベルを付け直してください:")
            for filename in emptied[:10]:
                print(f"      {filename}")


if __name__ == "__main__":
    main()
