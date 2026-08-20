"""モデルの確信度を較正し、どれだけずれていたかを図にする。

学習したモデルは「95%の確信度」と言いながら実際には65%しか当たらない。
判定そのものは変わらなくても、確信度を根拠に何かを判断するなら困る
(風替わりの分析で「確信度が高い方の時刻を採る」方式が偏ったのも、これが原因)。

温度スケーリングは、ロジットを1つの数Tで割ってから確率に直すだけの手法。
    p = sigmoid(logit / T)
Tが1より大きければ確信度は全体に下がる。ラベルの順位もモデルの予測も変えず、
確信度の目盛りだけを実態に合わせる。Tは検証データから求める(テストは使わない)。

使い方:
    python -m scripts.calibrate --data-dir <画像> --labels data/labels.csv \
        --weights runs/loyo_coordconv/model_test2023.pt --years 2023 2024 2025 \
        --split-mode loyo --test-year 2023 --out-dir reports
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.dataset import WeatherMapDataset
from src.labels import LABELS
from src.model import load_model
from src.split import SPLIT_MODES, VAL_MODES, make_splits
from src.train import get_transforms

# 信頼度図を描くときの区切り
BINS = np.linspace(0.0, 1.0, 11)


def collect(model, dataset, rows, device, batch_size):
    """指定した行のロジット(sigmoidをかける前の値)と正解を集める。"""
    loader = DataLoader(Subset(dataset, rows), batch_size=batch_size)
    logits, targets = [], []
    with torch.no_grad():
        for images, features, labels in loader:
            images = images.to(device)
            out = model(images, features.to(device)) if features.numel() else model(images)
            logits.append(out.cpu())
            targets.append(labels)
    return torch.cat(logits), torch.cat(targets)


def fit_temperature(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """検証データの損失が最小になる温度Tを探す。

    Tはラベル共通の1つの値。ラベルごとに別々のTを当てる方法もあるが、
    少数ラベルでは推定が不安定になるため、まずは共通の1つで揃える。
    """
    log_t = torch.zeros(1, requires_grad=True)  # T = exp(log_t) として正の値に保つ
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        optimizer.zero_grad()
        loss = criterion(logits / log_t.exp(), targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.exp().item())


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

    val_logits, val_targets = collect(model, dataset, splits["val"], device, args.batch_size)
    temperature = fit_temperature(val_logits, val_targets)
    print(f"\n検証データから求めた温度 T = {temperature:.3f}")
    if temperature > 1:
        print("  T > 1 なので、このモデルは自信過剰でした。確信度を下げる方向に補正します。")
    else:
        print("  T < 1 なので、このモデルは自信不足でした。確信度を上げる方向に補正します。")

    # 較正はテストデータで評価する(Tを求めるのに使っていない側で測る)
    test_logits, test_targets = collect(model, dataset, splits["test"], device, args.batch_size)
    before = torch.sigmoid(test_logits).numpy().ravel()
    after = torch.sigmoid(test_logits / temperature).numpy().ravel()
    truth = test_targets.numpy().ravel()

    rows_before = reliability(before, truth)
    rows_after = reliability(after, truth)
    ece_before = expected_calibration_error(rows_before, len(truth))
    ece_after = expected_calibration_error(rows_after, len(truth))

    print(f"\n{'=' * 62}\n確信度と実際の的中率(テストデータ)\n{'=' * 62}")
    print(f"  {'確信度の範囲':<14}{'件数':>8}{'平均確信度':>12}{'実際の的中率':>14}{'ずれ':>9}")
    print("  " + "-" * 58)
    for low, high, mean_p, mean_y, n in rows_before:
        if n == 0:
            continue
        print(f"  {low:.1f}〜{high:.1f}{'':<7}{n:>8}{mean_p:>12.3f}{mean_y:>14.3f}{mean_p - mean_y:>+9.3f}")

    print(f"\n  較正前のずれ(ECE): {ece_before:.4f}")
    print(f"  較正後のずれ(ECE): {ece_after:.4f}"
          f"  ({'改善' if ece_after < ece_before else '悪化'} {abs(ece_after - ece_before):.4f})")
    print("\n  ECEは各区間の|確信度 − 的中率|を件数で重み付け平均したもの。0に近いほど良い。")
    print("  判定そのものは変わらないので、F1などの成績は較正しても変化しない。")

    _plot(rows_before, rows_after, temperature, ece_before, ece_after, Path(args.out_dir))


def _plot(rows_before, rows_after, temperature, ece_before, ece_after, out_dir: Path):
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
    ax.set_title(f"確信度は実態と合っているか\n(温度スケーリング T = {temperature:.2f})",
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
