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

import numpy as np
import pandas as pd

from src.labels import LABEL_JA, LABELS, parse_labels


def cohen_kappa(a, b) -> float:
    """2値の一致度。0=偶然と同じ、1=完全一致。"""
    n = len(a)
    observed = (a == b).mean()
    # 偶然だけで一致する確率
    expected = sum((a == v).mean() * (b == v).mean() for v in (0, 1))
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0



def reconstruct_at_prevalence(sensitivity: float, specificity: float, prevalence: float) -> dict:
    """感度・特異度・出現率から、本来の出現率での混同行列と指標を組み直す。

    層別抽出(陽性と陰性を同数ずつ)で測った結果は、そのままではκもF1も
    出現率が人為的なままで、他のラベルと比べられない。感度と特異度は
    出現率に依存しないので、この2つと本来の出現率から組み直す。

      あり・あり   = p × 感度
      あり・なし   = p × (1 - 感度)
      なし・あり   = (1-p) × (1 - 特異度)
      なし・なし   = (1-p) × 特異度
    """
    p = prevalence
    both = p * sensitivity
    only_first = p * (1.0 - sensitivity)
    only_second = (1.0 - p) * (1.0 - specificity)
    neither = (1.0 - p) * specificity

    precision = both / (both + only_second) if both + only_second else 0.0
    recall = sensitivity
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    observed = both + neither
    expected = (both + only_first) * (both + only_second) + (only_second + neither) * (only_first + neither)
    kappa = (observed - expected) / (1 - expected) if expected < 1 else 1.0
    return {"both": both, "only_first": only_first, "only_second": only_second,
            "neither": neither, "precision": precision, "recall": recall,
            "f1": f1, "kappa": kappa, "agreement": observed}


def uncertainty(n_pos_correct, n_pos, n_neg_correct, n_neg, prevalence,
                draws: int = 20000, seed: int = 0) -> dict:
    """感度・特異度の推定誤差が、F1とκにどれだけ効くかを見る。

    30枚ずつしか見ていないので、感度・特異度そのものに幅がある。その幅を
    Beta分布(Jeffreys事前分布)から取り直して、F1とκの分布を作る。
    「測った」と言える精度が出ているかを、数字で判断するために出す。
    """
    rng = np.random.default_rng(seed)
    sens = rng.beta(n_pos_correct + 0.5, n_pos - n_pos_correct + 0.5, draws)
    spec = rng.beta(n_neg_correct + 0.5, n_neg - n_neg_correct + 0.5, draws)
    f1s, kappas = [], []
    for a, b in zip(sens, spec):
        r = reconstruct_at_prevalence(a, b, prevalence)
        f1s.append(r["f1"]); kappas.append(r["kappa"])
    return {
        "f1_low": float(np.percentile(f1s, 2.5)), "f1_high": float(np.percentile(f1s, 97.5)),
        "kappa_low": float(np.percentile(kappas, 2.5)), "kappa_high": float(np.percentile(kappas, 97.5)),
    }



def _report_stratified(args, labels, decided, name) -> None:
    """層別抽出した見直し結果を、本来の出現率に戻して報告する。"""
    stamps = pd.to_datetime(
        labels["filename"].str.extract(r"(\d{10})")[0], format="%Y%m%d%H", errors="coerce"
    )
    keep = stamps.notna()
    if args.prevalence_years:
        keep &= stamps.dt.year.isin(set(args.prevalence_years))
    if args.prevalence_months:
        keep &= stamps.dt.month.isin(set(args.prevalence_months))
    pool = labels[keep]
    prevalence = float(pool["original"].mean())

    pos = decided[decided["original"] == 1]
    neg = decided[decided["original"] == 0]
    if pos.empty or neg.empty:
        raise SystemExit(
            "層別の集計には陽性・陰性の両方が要ります"
            f"(陽性{len(pos)}枚 / 陰性{len(neg)}枚)。"
            "--stratified を外すか、scripts/sample_review.py で抜き直してください。"
        )

    n_pos_correct = int((pos["review"] == 1).sum())
    n_neg_correct = int((neg["review"] == 0).sum())
    sensitivity = n_pos_correct / len(pos)
    specificity = n_neg_correct / len(neg)

    print(f"  層別抽出            : 陽性{len(pos)}枚 / 陰性{len(neg)}枚")
    print(f"  感度(あり→あり)     : {sensitivity:.3f} ({n_pos_correct}/{len(pos)})")
    print(f"  特異度(なし→なし)   : {specificity:.3f} ({n_neg_correct}/{len(neg)})")
    print(f"  本来の出現率        : {prevalence:.3%}"
          f"({int(pool['original'].sum())}/{len(pool)}件)")

    r = reconstruct_at_prevalence(sensitivity, specificity, prevalence)
    u = uncertainty(n_pos_correct, len(pos), n_neg_correct, len(neg), prevalence)

    print(f"\n  --- 本来の出現率に戻した推定 ---")
    print(f"  一致率              : {r['agreement']:.1%}")
    print(f"  Cohenのκ            : {r['kappa']:.3f}  [95%区間 {u['kappa_low']:.3f}〜{u['kappa_high']:.3f}]")
    print(f"  人間どうしのF1      : {r['f1']:.3f}  [95%区間 {u['f1_low']:.3f}〜{u['f1_high']:.3f}]")
    print(f"    適合率 {r['precision']:.3f} / 再現率 {r['recall']:.3f}")
    print("\n  この F1 が、そのラベルでモデルに期待できる上限の目安。"
          "\n  区間が広いときは枚数が足りていない —— sample_review.py の --n-positive を増やす。")

    width = u["f1_high"] - u["f1_low"]
    if width > 0.25:
        # どちらの層を増やすべきかは出現率で変わる。少数ラベルでは適合率が
        # 偽陽性 =(1-出現率)×(1-特異度) に支配されるため、陽性をいくら増やしても
        # 区間は狭まらない。実際にどちらが効くかを、この場で試算して示す。
        more_pos = uncertainty(int(round(len(pos) * 2 * sensitivity)), len(pos) * 2,
                               n_neg_correct, len(neg), prevalence, draws=4000, seed=1)
        more_neg = uncertainty(n_pos_correct, len(pos),
                               int(round(len(neg) * 2 * specificity)), len(neg) * 2,
                               prevalence, draws=4000, seed=1)
        gain_pos = width - (more_pos["f1_high"] - more_pos["f1_low"])
        gain_neg = width - (more_neg["f1_high"] - more_neg["f1_low"])
        target = "陰性" if gain_neg > gain_pos else "陽性"
        print(f"\n  ⚠ F1の区間幅が{width:.2f}あります。この枚数では『測った』と言える"
              "精度に達していません。")
        print(f"    枚数を倍にしたときの改善: 陽性{len(pos)}→{len(pos)*2}枚で {gain_pos:+.3f} / "
              f"陰性{len(neg)}→{len(neg)*2}枚で {gain_neg:+.3f}")
        print(f"    → **{target}**を増やすほうが効きます"
              f"(sample_review.py の --n-{'negative' if target == '陰性' else 'positive'})")
        if prevalence < 0.15 and target == "陰性":
            print(f"    出現率が{prevalence:.1%}と低いため、適合率は特異度のわずかな差で"
                  "大きく動きます。")
            print("    F1の上限を精度よく出すには枚数がかさむので、"
                  "感度(再現率の上限)だけを報告する選択もあります。")


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
        "--stratified",
        action="store_true",
        help="scripts/sample_review.py で陽性・陰性を同数ずつ抜いた結果を集計する。"
        "感度と特異度を別々に測り、--labels 全体の出現率で組み直す。"
        "付け忘れると、陽性を多く含む集合のままのκ・F1になり比較できない",
    )
    parser.add_argument(
        "--prevalence-years", type=int, nargs="+", default=None,
        help="--stratified で出現率を測る範囲の年。抽出時に --years を使ったなら同じ値を指定する",
    )
    parser.add_argument(
        "--prevalence-months", type=int, nargs="+", default=None,
        help="--stratified で出現率を測る範囲の月。抽出時に --months を使ったなら同じ値を指定する",
    )
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

    if args.stratified:
        _report_stratified(args, labels, decided, name)
        return

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
