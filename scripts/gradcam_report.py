"""指定したラベルについて、テストセットの実例でGrad-CAMの図を作る。

既存の gradcam.py は「1枚の画像で確信度の高い上位ラベル」を可視化するもので、
論文の図には向かない。こちらは逆に、

  * ラベルを1つ指定し(モデルが得意な台風、苦手なオホーツク海高気圧など)
  * そのラベルが実際に正解となっているテスト画像を選び
  * 当てられた例(成功)と外した例(失敗)を並べる

という作りにしてある。成功例と失敗例を並べると「何を見て当てているのか」
「外すときは何を見ているのか」が対比でき、考察の材料になる。

交差検証で作った重みと、そのfoldのテストセットをそのまま使う:

    python -m scripts.gradcam_report \
        --data-dir data/processed --labels data/labels.csv \
        --weights runs/loyo/model_test2025.pt \
        --years 2023 2024 2025 --split-mode loyo --test-year 2025 \
        --label typhoon --out runs/loyo/gradcam_typhoon.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset

from scripts.gradcam import GradCAM, _load_model, _overlay_heatmap
from src.dataset import WeatherMapDataset
from src.labels import LABEL_JA, LABEL_TO_INDEX, LABELS
from src.split import SPLIT_MODES, make_splits
from src.train import get_transforms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--label", required=True, choices=list(LABELS), help="可視化したいラベル")
    parser.add_argument("--n", type=int, default=3, help="成功例・失敗例をそれぞれ何枚並べるか")
    parser.add_argument("--out", required=True, help="出力する画像ファイル")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.5, help="成功/失敗の判定に使う閾値")
    # 分割の再現用(train.pyと同じ値を渡す)
    parser.add_argument("--years", type=int, nargs="+", default=None)
    parser.add_argument("--split-mode", choices=list(SPLIT_MODES), default="temporal")
    parser.add_argument("--test-year", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.0)
    parser.add_argument("--gap-days", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = _load_model(args.weights, device)

    dataset = WeatherMapDataset(
        args.data_dir,
        args.labels,
        transform=get_transforms(train=False, image_size=meta["image_size"]),
        years=args.years,
    )
    splits = make_splits(
        dataset.df,
        mode=args.split_mode,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        test_year=args.test_year,
        gap_days=args.gap_days,
    )
    test_rows = splits["test"]
    if not test_rows:
        raise SystemExit("テストセットが0件です。--test-year / --test-ratio を確認してください。")

    class_idx = LABEL_TO_INDEX[args.label]

    # テストセット全体を推論し、対象ラベルの確信度と正解を集める
    loader = DataLoader(Subset(dataset, test_rows), batch_size=args.batch_size)
    probs, truths = [], []
    with torch.no_grad():
        for images, targets in loader:
            probs.append(torch.sigmoid(model(images.to(device)))[:, class_idx].cpu())
            truths.append(targets[:, class_idx])
    probs = torch.cat(probs).numpy()
    truths = torch.cat(truths).numpy()

    positive = np.where(truths == 1)[0]
    if len(positive) == 0:
        raise SystemExit(f"テストセットに「{LABEL_JA[args.label]}」の正解例がありません。")

    # 正解例のうち、確信度が高い順=当てられた例、低い順=見逃した例
    ordered = positive[np.argsort(-probs[positive])]
    hits = [i for i in ordered if probs[i] > args.threshold][: args.n]
    misses = [i for i in ordered[::-1] if probs[i] <= args.threshold][: args.n]

    chosen = [("正解", i) for i in hits] + [("見逃し", i) for i in misses]
    if not chosen:
        raise SystemExit("表示できる例がありませんでした。")

    print(
        f"{LABEL_JA[args.label]}: テスト{len(test_rows)}件中 正解ラベルあり{len(positive)}件 "
        f"→ 当てられた{sum(probs[positive] > args.threshold)}件 / 見逃し{sum(probs[positive] <= args.threshold)}件"
    )

    gradcam = GradCAM(model)
    fig, axes = plt.subplots(2, len(chosen), figsize=(3.2 * len(chosen), 6.8), squeeze=False)

    for col, (kind, local_idx) in enumerate(chosen):
        row_idx = test_rows[local_idx]
        record = dataset.df.iloc[row_idx]

        # モデルに入れるのは前処理済みテンソル、表示するのは元の見た目の画像。
        # CAMは入力サイズの格子で出るが、_overlay_heatmapが元画像サイズへ
        # 一様に拡大するので、位置の対応はずれない。
        tensor, _ = dataset[row_idx]
        cam = gradcam.generate(tensor.unsqueeze(0).to(device), class_idx)

        base = Image.open(Path(args.data_dir) / record["filename"]).convert("RGB")
        overlay = _overlay_heatmap(base, cam)

        stamp = record["parsed_datetime"]
        date_text = stamp.strftime("%Y-%m-%d %HZ") if pd.notna(stamp) else record["filename"]

        axes[0][col].imshow(base)
        axes[0][col].set_title(
            f"{kind}  {date_text}\n確信度 {probs[local_idx] * 100:.1f}%", fontsize=10
        )
        axes[1][col].imshow(overlay)
        for row in (0, 1):
            axes[row][col].set_xticks([])
            axes[row][col].set_yticks([])

    # axis("off")にすると行ラベルも消えるので、目盛りだけ消して左端にだけ文字を出す
    axes[0][0].set_ylabel("入力天気図", fontsize=11)
    axes[1][0].set_ylabel("Grad-CAM", fontsize=11)
    fig.suptitle(
        f"{LABEL_JA[args.label]}({args.label})  —  モデルが注目した領域", fontsize=13
    )
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"保存しました: {out_path}")


if __name__ == "__main__":
    main()
