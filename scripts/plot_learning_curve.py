"""scripts/learning_curve.py の結果(summary.json)を図にする。

数値を貼り直さずに済むよう、必ずJSONから読む。学習をやり直したら
このスクリプトを再実行するだけで図が更新される。

使い方:
    python -m scripts.plot_learning_curve --summary runs/curve/summary.json --out-dir reports
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from src.labels import LABEL_JA, LABELS

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#b8b7b0"
BLUE, RED = "#2a78d6", "#d92626"
# F1がこれ未満のラベルは「件数を増やしても伸びない」ものとして色を変える
WEAK_F1 = 0.30


def _use_japanese_font():
    """日本語が豆腐(□)にならないよう、環境にあるゴシック体を探して使う。"""
    candidates = [
        "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",  # Linux
        "C:/Windows/Fonts/YuGothM.ttc", "C:/Windows/Fonts/meiryo.ttc",
        "C:/Windows/Fonts/msgothic.ttc",
    ]
    for path in candidates:
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
            break
    else:
        for name in ("Yu Gothic", "Meiryo", "MS Gothic", "IPAGothic", "Noto Sans CJK JP"):
            if any(f.name == name for f in font_manager.fontManager.ttflist):
                plt.rcParams["font.family"] = name
                break
    plt.rcParams["axes.unicode_minus"] = False


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color="#e6e5e0", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d8d7d1")
    ax.tick_params(colors=INK2, length=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="runs/curve/summary.json")
    parser.add_argument("--out-dir", default="reports")
    parser.add_argument("--test-year", type=int, default=None, help="図の副題に入れるテスト年")
    args = parser.parse_args()

    _use_japanese_font()
    results = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # train_limitがNoneの行が「全件」。実際の件数はJSONに入っていないことがあるため、
    # 最大の指定件数より少し大きい位置に置いて右端であることが分かるようにする。
    limits = [r["train_limit"] for r in results]
    known = [l for l in limits if l is not None]
    n = np.array([l if l is not None else max(known) + 255 for l in limits], dtype=float)
    f = np.array([r["macro_f1_evaluable"] for r in results])
    ticks = [str(l) if l is not None else "全件" for l in limits]

    subtitle = "leave-one-year-out"
    if args.test_year:
        subtitle += f" / テスト={args.test_year}年通年"

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)

    a, b = np.polyfit(np.log(n), f, 1)
    fitted = a * np.log(n) + b
    r2 = 1 - ((f - fitted) ** 2).sum() / ((f - f.mean()) ** 2).sum()
    xs = np.linspace(n.min() * 0.9, n.max() * 1.08, 200)
    ax.plot(xs, a * np.log(xs) + b, color=MUTED, lw=1.5, ls=(0, (4, 3)), zorder=1)
    ax.plot(n, f, color=BLUE, lw=2, marker="o", ms=8, zorder=3,
            markerfacecolor=BLUE, markeredgecolor=SURFACE, markeredgewidth=2)
    for x, y in zip(n, f):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=10, color=INK)
    mid = n[len(n) // 2 + 1]
    ax.annotate(f"対数近似 (R²={r2:.2f})", (mid, a * np.log(mid) + b),
                textcoords="offset points", xytext=(6, -26), ha="left", fontsize=9, color=INK2)

    ax.set_xlabel("学習データ件数", fontsize=11, color=INK2)
    ax.set_ylabel("macro F1", fontsize=11, color=INK2)
    ax.set_title(f"学習データ件数と分類性能\n({subtitle})", fontsize=13, color=INK,
                 loc="left", pad=14)
    ax.set_ylim(max(0.0, f.min() - 0.08), f.max() + 0.07)
    ax.set_xticks(n)
    ax.set_xticklabels(ticks)
    fig.tight_layout()
    path = out_dir / "learning_curve.png"
    fig.savefig(path, facecolor=SURFACE)
    print(f"保存しました: {path}")

    # ラベルごとの推移。10本を1枚に重ねると読めないので小さなグラフに分ける。
    per = {label: [r["per_label"][label]["f1"] for r in results] for label in LABELS}
    order = sorted(LABELS, key=lambda l: -per[l][-1])
    fig, axes = plt.subplots(2, 5, figsize=(13, 5.4), dpi=200, sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, label in zip(axes.ravel(), order):
        values = per[label]
        weak = values[-1] < WEAK_F1
        color = RED if weak else BLUE
        _style(ax)
        ax.plot(n, values, color=color, lw=2, marker="o", ms=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.set_title(f"{LABEL_JA[label]}\n{values[-1]:.2f}", fontsize=9.5,
                     color=RED if weak else INK)
        ax.set_ylim(0, 0.9)
        ax.tick_params(labelsize=8)
        ax.set_xticks([n[0], n[len(n) // 2], n[-1]])
        ax.set_xticklabels([ticks[0], ticks[len(n) // 2], ticks[-1]])
    fig.suptitle("ラベル別 F1 と学習件数", fontsize=12, color=INK, x=0.01, ha="left", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = out_dir / "learning_curve_by_label.png"
    fig.savefig(path, facecolor=SURFACE)
    print(f"保存しました: {path}")

    print(f"\nF1 = {a:.4f}*ln(n) + {b:.4f}")
    for gain in (0.02, 0.05, 0.10):
        print(f"  +{gain:.2f} には現在の{np.exp(gain / a):.2f}倍の学習データが必要")


if __name__ == "__main__":
    main()
