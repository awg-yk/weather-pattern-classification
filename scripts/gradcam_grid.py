"""ERA5格子モデルが気圧場のどこを見ているかを可視化する。

天気図画像用のscripts/gradcam.pyと同じGrad-CAMを、2チャンネルの格子入力に
適用する。オホーツク海高気圧だけERA5格子が天気図画像に勝った(0.312 vs 0.240)
理由を確かめるのが目的で、南部の気圧を見ているなら、これまでの分析
(領域を南北に分けたら効いた、地衡風の東西成分が効いた)と一致する。

背景には気圧場の等値線を引くので、モデルの注目箇所と気圧配置を重ねて読める。

使い方:
    python -m scripts.gradcam_grid --weights runs/loyo_grid/model_test2023.pt \
        --labels data/labels.csv --era5-grid-dir data/raw/era5 \
        --label okhotsk_high --years 2023 2024 2025 --test-year 2023 --out-dir reports/gradcam_grid
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from scripts.gradcam import GradCAM
from src.era5_grid import ERA5GridDataset, compute_grid_stats
from src.labels import LABEL_JA, LABELS
from src.model import load_model
from src.split import SPLIT_MODES, VAL_MODES, make_splits

# 描画範囲(src/era5_grid.pyが読み込むnetCDFの範囲と揃えること)
AREA_NORTH, AREA_SOUTH = 60.0, 15.0
AREA_WEST, AREA_EAST = 110.0, 165.0


def _plot(grids, cams, titles, out_path: Path, label_ja: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patheffects as path_effects
    import matplotlib.pyplot as plt

    from scripts.plot_learning_curve import _use_japanese_font

    _use_japanese_font()
    columns = len(grids)
    fig, axes = plt.subplots(1, columns, figsize=(4.2 * columns, 4.6), dpi=180, squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")

    extent = (AREA_WEST, AREA_EAST, AREA_SOUTH, AREA_NORTH)
    for ax, grid, cam, title in zip(axes[0], grids, cams, titles):
        # 注目箇所を下地に敷き、その上に気圧の等値線を白で重ねる。
        # 逆順(等値線が下)だと熱マップに埋もれて気圧配置が読めない。
        image = ax.imshow(cam, cmap="magma", extent=extent, origin="upper",
                          aspect="auto", vmin=0, vmax=1)
        contours = ax.contour(grid[0], levels=10, colors="white", linewidths=0.9,
                              extent=extent, origin="upper", alpha=0.85)
        # 等値線に細い暗い縁を付け、熱マップが明るい部分でも見えるようにする。
        # matplotlib 3.8以降 QuadContourSet.collections は廃止されたので、
        # ContourSet自体にpath_effectsを設定する(旧版には collections を使う)。
        targets = getattr(contours, "collections", [contours])
        for target in targets:
            target.set_path_effects([
                path_effects.Stroke(linewidth=2.0, foreground="#00000055"),
                path_effects.Normal(),
            ])
        ax.set_title(title, fontsize=10.5, color="#0b0b0b")
        ax.set_xlabel("東経", fontsize=8.5, color="#52514e")
        ax.set_ylabel("北緯", fontsize=8.5, color="#52514e")
        ax.tick_params(labelsize=8, colors="#52514e")

    bar = fig.colorbar(image, ax=axes[0], fraction=0.025, pad=0.015)
    bar.set_label("注目の強さ", fontsize=8.5, color="#52514e")
    bar.ax.tick_params(labelsize=7.5, colors="#52514e")

    fig.suptitle(f"{label_ja} — ERA5格子モデルの注目箇所(明るいほど強い)",
                 fontsize=12.5, color="#0b0b0b", x=0.01, ha="left")
    fig.subplots_adjust(top=0.86, bottom=0.12, left=0.05, right=0.9, wspace=0.22)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor="#fcfcfb")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--era5-grid-dir", default="data/raw/era5")
    parser.add_argument("--label", default="okhotsk_high", choices=list(LABELS))
    parser.add_argument("--years", type=int, nargs="+", default=None)
    parser.add_argument("--split-mode", default="loyo", choices=SPLIT_MODES)
    parser.add_argument("--test-year", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--gap-days", type=int, default=3)
    parser.add_argument("--val-mode", default="tail", choices=VAL_MODES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--count", type=int, default=3, help="正解例・見逃し例をそれぞれ何枚出すか")
    parser.add_argument("--out-dir", default="reports/gradcam_grid")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_model(args.weights, map_location=device)
    model.to(device).eval()
    if meta["in_channels"] != 2:
        raise SystemExit(
            f"この重みは{meta['in_channels']}チャンネル入力です。"
            "ERA5格子(2チャンネル)の重みを指定してください"
            "(天気図画像の重みには scripts/gradcam.py を使います)。"
        )

    dataset = ERA5GridDataset(
        args.labels, args.era5_grid_dir, years=args.years,
        grid_size=meta["image_size"], augment=False,
    )
    splits = make_splits(
        dataset.df, mode=args.split_mode, val_ratio=args.val_ratio, test_ratio=0.0,
        seed=args.seed, test_year=args.test_year, gap_days=args.gap_days, val_mode=args.val_mode,
    )
    # 学習時と同じ手順で正規化する(統計は学習用サブセットだけから)
    mean, std = compute_grid_stats(dataset, splits["train"])
    dataset.set_stats(mean, std)

    index = LABELS.index(args.label)
    cam = GradCAM(model)

    hits, misses = [], []
    for row in splits["test"]:
        grid, _features, target = dataset[row]
        if not target[index]:
            continue
        tensor = grid.unsqueeze(0).to(device)
        with torch.no_grad():
            probability = float(torch.sigmoid(model(tensor))[0, index])
        record = (row, grid.numpy(), tensor, probability)
        (hits if probability >= 0.5 else misses).append(record)

    print(f"{LABEL_JA[args.label]}: テストセット中の陽性 {len(hits) + len(misses)}件"
          f"(検出 {len(hits)}件 / 見逃し {len(misses)}件)")
    if not hits and not misses:
        raise SystemExit("この年のテストセットに、そのラベルの陽性がありません")

    out_dir = Path(args.out_dir)
    for name, records, japanese in (("hit", hits, "検出できた例"), ("miss", misses, "見逃した例")):
        # 確信度の高い順・低い順に選び、両極端を見る
        records = sorted(records, key=lambda r: -r[3])[: args.count]
        if not records:
            print(f"  {japanese}: 該当なし")
            continue
        grids, cams, titles = [], [], []
        for row, grid, tensor, probability in records:
            heat = cam.generate(tensor, index)
            # CAMは畳み込みの解像度なので、格子の解像度に引き伸ばす
            heat = torch.nn.functional.interpolate(
                torch.from_numpy(heat)[None, None], size=grid.shape[1:],
                mode="bilinear", align_corners=False,
            )[0, 0].numpy()
            grids.append(grid)
            cams.append(heat)
            stamp = dataset.df.at[row, "parsed_datetime"]
            titles.append(f"{stamp:%Y-%m-%d %HZ}  確信度 {probability:.2f}")
        path = out_dir / f"{args.label}_{name}.png"
        _plot(grids, cams, titles, path, f"{LABEL_JA[args.label]} / {japanese}")
        print(f"  {japanese}: {path}")


if __name__ == "__main__":
    main()
