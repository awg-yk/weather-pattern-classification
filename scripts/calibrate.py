"""確信度を較正し、どれだけずれていたかを図にして、重みの隣に保存する。

学習したモデルは「95%の確信度」と言いながら実際には65%しか当たらない。
判定そのものは変わらなくても、確信度を根拠に何かを判断するなら困る
(風替わりの分析で「確信度が高い方の時刻を採る」方式が偏ったのも、これが原因)。

■ 温度スケーリングから、ラベルごとの当てはめへ

以前はロジットを1つの数Tで割る温度スケーリングを使っていた:
    p = sigmoid(logit / T)
これは全体の自信過剰をならすが、**pos_weight によるかさ上げは原理的に消せない**。
sigmoid(z/T) は T が何であっても z=0 を必ず 0.5 に写す。しかし pos_weight=8 の
ラベルでは生の出力 0.5 の実体は 1/(1+8) = 11% しかない。割り算では、この
「引き算」の歪みを表現できない。しかも pos_weight はラベルごとに違うので、
共通のTでは「ラベルによってかさ上げ量が違う」部分にも届かない。

そこでラベルごとに2つの数を当てはめる(Platt scaling):
    p = sigmoid(a · logit + b)
b が加法シフトなので pos_weight の補正を含む(a=1, b=-log w がその解析解)。
検証データに陽性が少ないラベルは当てはめが不安定になるため、その場合は
解析解 b=-log w だけを使う。詳しくは src/calibration.py を参照。

■ 結果は保存され、推論側が自動で読む

<重み名>.calib.json を書き出す。scripts/predict.py・webapp・
scripts/classify_dates.py・Grad-CAM はこれを自動的に読み込む。
重みの指紋を記録するので、学習し直したのに古い校正が残っていれば検出できる。

使い方:
    python -m scripts.calibrate --data-dir <画像> --labels data/labels_v2.csv \
        --weights runs/v2_chart/model_test2023.pt --years 2023 2024 2025 \
        --split-mode loyo --test-year 2023 --out-dir reports
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from src import calibration as calib
from src.dataset import WeatherMapDataset
from src.labels import LABEL_JA, LABELS
from src.model import load_model
from src.split import SPLIT_MODES, VAL_MODES, make_splits, min_days_to_other_split
from src.train import compute_pos_weight, get_transforms

# 信頼度図を描くときの区切り
BINS = np.linspace(0.0, 1.0, 11)


def reliability(probs: np.ndarray, targets: np.ndarray):
    """確信度の区間ごとに、実際の的中率と件数を返す。"""
    rows = []
    for low, high in zip(BINS[:-1], BINS[1:]):
        mask = (probs >= low) & (probs < high)
        if mask.sum() == 0:
            rows.append((low, high, np.nan, np.nan, 0))
            continue
        rows.append((low, high, probs[mask].mean(), targets[mask].mean(), int(mask.sum())))
    return rows


def expected_calibration_error(rows, total: int) -> float:
    """各区間の |確信度 − 的中率| を件数で重み付けして平均したもの。0に近いほど良い。"""
    return sum(
        n * abs(mean_p - mean_y) for _, _, mean_p, mean_y, n in rows if n > 0
    ) / total


def report_boundary_leak(df, splits, gap_days: int) -> None:
    """valとtestが、学習データからどれだけ時間的に離れているかを数える。

    temporal分割は日付順に切るだけなので、境界の前後は1日しか離れていない。
    天気図は隣り合う日が極めて似ているため、その数件は学習済みの画像を当てて
    いるのに近く、校正パラメータもECEもその分だけ良い方向に出る。
    近すぎる行があっても勝手に除外はしない(除外するとこれまでの実験と分割が
    変わって数値を比較できなくなる)。件数を出して判断材料にする。
    """
    for name in ("val", "test"):
        rows = splits.get(name) or []
        if not rows or not splits["train"]:
            continue
        distance = min_days_to_other_split(df, rows, splits["train"])
        close = int((distance <= gap_days).sum())
        if close:
            print(f"  時間的リークの目安: {name} {len(rows)}件のうち{close}件"
                  f"({close / len(rows) * 100:.1f}%)が学習データと{gap_days}日以内。"
                  f"最短{distance.min():.0f}日")
        else:
            print(f"  時間的リークの目安: {name}は学習データから{gap_days}日以上離れています")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--era5-features", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--years", type=int, nargs="+", default=None)
    parser.add_argument("--split-mode", default="loyo", choices=SPLIT_MODES)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.0)
    parser.add_argument("--test-year", type=int, default=None)
    parser.add_argument("--gap-days", type=int, default=3)
    parser.add_argument("--val-mode", default="tail", choices=VAL_MODES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="reports")
    parser.add_argument(
        "--pos-weight-cap",
        type=float,
        default=8.0,
        help="重みにpos_weightが記録されていない場合に、学習時の値を学習データから"
        "計算し直すための上限。学習時の --pos-weight-cap と同じ値を指定すること",
    )
    parser.add_argument(
        "--min-positives",
        type=int,
        default=calib.MIN_POSITIVES_FOR_FIT,
        help="検証データの陽性がこの数に満たないラベルは、当てはめずpos_weightの"
        "解析的な補正だけを使う",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="校正の書き出し先。既定は重みと同じ場所の <重み名>.calib.json",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_model(args.weights, map_location=device)
    model.to(device).eval()

    dataset = WeatherMapDataset(
        args.data_dir, args.labels,
        transform=get_transforms(train=False, image_size=meta["image_size"]),
        years=args.years, features_csv=args.era5_features,
    )
    splits = make_splits(
        dataset.df, mode=args.split_mode, val_ratio=args.val_ratio,
        test_ratio=args.test_ratio, seed=args.seed, test_year=args.test_year,
        gap_days=args.gap_days, val_mode=args.val_mode,
    )
    if not splits["val"]:
        raise SystemExit("valが0件です。--val-ratio / --split-mode を学習時と揃えてください。")

    pos_weight = meta.get("pos_weight")
    pos_weight_source = "checkpoint" if pos_weight is not None else "recomputed"
    if pos_weight is not None:
        pos_weight = np.asarray(pos_weight, dtype="float64")
        print("学習時のpos_weight(重みに記録された値):",
              {l: round(w, 2) for l, w in zip(LABELS, pos_weight)})
    else:
        # pos_weightを記録する前に学習された重み。compute_pos_weight() は学習用
        # サブセットのラベル分布だけで決まる決定的な計算なので、同じ分割と同じ
        # --pos-weight-cap を与えれば学習時の値をそのまま再現できる。
        from torch.utils.data import Subset

        pos_weight = compute_pos_weight(
            Subset(dataset, splits["train"]), num_classes=len(LABELS), cap=args.pos_weight_cap
        ).numpy().astype("float64")
        print(f"この重みにはpos_weightが記録されていないため、学習データ"
              f"({len(splits['train'])}件)から再計算しました"
              f"(--pos-weight-cap {args.pos_weight_cap})。学習時と違う値を使っていた場合は、"
              "その値を --pos-weight-cap に指定してください。")
        print("再計算したpos_weight:", {l: round(w, 2) for l, w in zip(LABELS, pos_weight)})

    report_boundary_leak(dataset.df, splits, args.gap_days)

    # 当てはめは必ずvalで行う。テストに当てはめて同じテストで報告すると、
    # 校正の効き目が実際より良く見える。
    val_logits, val_targets = calib.collect_logits(
        model, dataset, splits["val"], device, args.batch_size
    )
    print(f"校正の当てはめ: val {len(splits['val'])}件")

    calibration = calib.fit(
        val_logits, val_targets, pos_weight=pos_weight, min_positives=args.min_positives
    )
    calibration.source = calib.build_source(
        weights_path=args.weights,
        labels_csv=args.labels,
        image_size=meta["image_size"],
        pos_weight=pos_weight,
        pos_weight_source=pos_weight_source,
        pos_weight_cap=args.pos_weight_cap if pos_weight_source == "recomputed" else None,
        split={
            "mode": args.split_mode, "val_mode": args.val_mode,
            "val_ratio": args.val_ratio, "test_ratio": args.test_ratio,
            "seed": args.seed, "years": args.years,
            "test_year": args.test_year, "gap_days": args.gap_days,
        },
        fitted_on="val",
        n_fit=len(splits["val"]),
    )

    print("\nラベルごとの校正")
    print(f"  {'ラベル':<20}{'方法':>8}{'陽性数':>7}{'a':>8}{'b':>9}{'しきい値':>10}")
    degenerate = []
    for label in LABELS:
        c = calibration[label]
        print(f"  {LABEL_JA[label]:<20}{c.method:>8}{c.n_positive:>7}"
              f"{c.a:>8.2f}{c.b:>9.2f}{c.threshold:>10.3f}")
        if c.method == "platt" and (c.threshold <= 0.03 or c.threshold >= 0.97):
            degenerate.append(label)
    if degenerate:
        print("\n  警告: 次のラベルはしきい値が端に張り付いています"
              "(事実上『常に陽性』または『常に陰性』):\n    "
              + ", ".join(LABEL_JA[l] for l in degenerate)
              + "\n  検証データの件数が足りないか、そのラベルをモデルが区別できていません。"
              "確信度で絞り込む運用には使えないので、そのラベルは人の確認に回してください。")

    # ---- 効き目はテストで測る(当てはめに使っていない側) ----
    report_rows = splits["test"] or splits["val"]
    on_val = not splits["test"]
    if on_val:
        print("\n注意: テストセットが空のため、当てはめたvalで報告します。"
              "この数値は実際より良く出ます。--test-ratio か --test-year を指定してください。")
        test_logits, test_targets = val_logits, val_targets
    else:
        test_logits, test_targets = calib.collect_logits(
            model, dataset, report_rows, device, args.batch_size
        )

    # 図は従来と同じ「全ラベルの確率をならべた」信頼度図。過去の図と比較できるようにする。
    before = calib.sigmoid(test_logits).ravel()
    after = calibration.probabilities(test_logits).ravel()
    truth = test_targets.ravel()

    rows_before = reliability(before, truth)
    rows_after = reliability(after, truth)
    ece_before = expected_calibration_error(rows_before, len(truth))
    ece_after = expected_calibration_error(rows_after, len(truth))

    where = "val" if on_val else "テストデータ"
    print(f"\n{'=' * 62}\n確信度と実際の的中率({where}・全ラベル)\n{'=' * 62}")
    print(f"  {'確信度の範囲':<14}{'件数':>8}{'平均確信度':>12}{'実際の的中率':>14}{'ずれ':>9}")
    print("  " + "-" * 58)
    for low, high, mean_p, mean_y, n in rows_before:
        if n == 0:
            continue
        print(f"  {low:.1f}〜{high:.1f}{'':<7}{n:>8}{mean_p:>12.3f}{mean_y:>14.3f}{mean_p - mean_y:>+9.3f}")

    print(f"\n  較正前のずれ(ECE): {ece_before:.4f}")
    print(f"  較正後のずれ(ECE): {ece_after:.4f}"
          f"  ({'改善' if ece_after < ece_before else '悪化'} {abs(ece_after - ece_before):.4f})")

    # 利用者が実際に目にするのは「1位ラベルの確信度」なので、そちらも別に測る。
    summary = calib.summarize(test_logits, test_targets, calibration)
    print(f"\n{'=' * 62}\n1位ラベルの確信度と、実際の的中率\n{'=' * 62}")
    print(f"1位ラベルが正解に含まれていた割合: {summary['top1_accuracy'] * 100:.1f}%")
    print(f"\n[校正前] 平均確信度 {summary['raw']['mean_confidence'] * 100:.1f}% / "
          f"ECE {summary['raw']['ece']:.3f}")
    print(calib.format_reliability(summary["raw"]["reliability"]))
    print(f"\n[校正後] 平均確信度 {summary['calibrated']['mean_confidence'] * 100:.1f}% / "
          f"ECE {summary['calibrated']['ece']:.3f}")
    print(calib.format_reliability(summary["calibrated"]["reliability"]))
    print("\n  判定そのものは変わらないので、しきい値を選び直すかぎり"
          "F1やmacro APは較正しても変化しない。")

    calibration.metrics = {
        "report_split": "val" if on_val else "test",
        "n": summary["n"],
        "top1_accuracy": summary["top1_accuracy"],
        "ece_raw": summary["raw"]["ece"],
        "ece_calibrated": summary["calibrated"]["ece"],
        "ece_all_labels_raw": ece_before,
        "ece_all_labels_calibrated": ece_after,
    }

    out_path = Path(args.out) if args.out else calib.default_path(args.weights)
    calibration.save(out_path)
    print(f"\n校正を書き出しました: {out_path}")
    print("これ以降、predict / webapp / classify_dates / Grad-CAM はこのファイルを自動で読みます。")

    _plot(rows_before, rows_after, ece_before, ece_after, Path(args.out_dir))


def _plot(rows_before, rows_after, ece_before, ece_after, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from scripts.plot_learning_curve import _style, _use_japanese_font, BLUE, INK, INK2, MUTED, SURFACE

    _use_japanese_font()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 6.0), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)

    ax.plot([0, 1], [0, 1], color=MUTED, lw=1.5, ls=(0, (4, 3)), zorder=1)
    ax.annotate("完全に較正された状態", (0.97, 0.97), textcoords="offset points",
                xytext=(-6, -16), ha="right", fontsize=9, color=INK2)

    for rows, color, label in ((rows_before, "#d92626", f"較正前 (ECE {ece_before:.3f})"),
                              (rows_after, BLUE, f"較正後 (ECE {ece_after:.3f})")):
        xs = [p for _, _, p, y, n in rows if n > 0]
        ys = [y for _, _, p, y, n in rows if n > 0]
        ax.plot(xs, ys, color=color, lw=2, marker="o", ms=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=label, zorder=3)

    ax.set_xlabel("モデルが出した確信度", fontsize=11, color=INK2)
    ax.set_ylabel("実際に当たっていた割合", fontsize=11, color=INK2)
    ax.set_title("確信度は実態と合っているか\n(ラベルごとの当てはめ p = sigmoid(a·z + b))",
                 fontsize=13, color=INK, loc="left", pad=14)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    fig.tight_layout()
    path = out_dir / "calibration.png"
    fig.savefig(path, facecolor=SURFACE)
    print(f"\n図を書き出しました: {path}")
    print("  対角線より下にあるほど自信過剰(確信度ほどには当たっていない)。")


if __name__ == "__main__":
    main()
