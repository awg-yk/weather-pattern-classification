"""
リポジトリに同梱された学習済みモデル(weights/model.pt)を使って、
天気図画像1枚を分類するコマンドラインツール。学習は行わない。

使い方:
    python scripts/predict.py path/to/chart.png

    # 既に前処理済み(余白クロップ・日時スタンプ消去済み)の画像の場合
    python scripts/predict.py path/to/chart.png --no-preprocess
"""

import argparse
from pathlib import Path

import torch
from PIL import Image

from scripts.preprocess_jma import DEFAULT_STAMP_BOX, autocrop_to_content, mask_stamp_box
from src.labels import INDEX_TO_LABEL, LABEL_JA, LABELS
from src.model import build_model
from src.train import get_transforms

DEFAULT_WEIGHTS = Path(__file__).resolve().parent.parent / "weights" / "model.pt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="分類したい天気図画像のパス")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="モデルの重みファイル")
    parser.add_argument("--threshold", type=float, default=0.5, help="このしきい値を超えたラベルを表示")
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="気象庁の生画像向け前処理(余白クロップ・日時スタンプ消去)をスキップする",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=len(LABELS))
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)
    model.eval()

    image = Image.open(args.image).convert("RGB")
    if not args.no_preprocess:
        image = autocrop_to_content(image)
        image = mask_stamp_box(image, DEFAULT_STAMP_BOX)

    transform = get_transforms(train=False)
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.sigmoid(model(input_tensor))[0].cpu()

    ranked = sorted(
        ((INDEX_TO_LABEL[i], p.item()) for i, p in enumerate(probs)),
        key=lambda x: x[1],
        reverse=True,
    )

    predicted = [label for label, p in ranked if p > args.threshold]
    print("予測:", " / ".join(LABEL_JA[l] for l in predicted) if predicted else "該当なし")
    print()
    print("--- 全ラベルの確信度 ---")
    for label, p in ranked:
        print(f"{LABEL_JA[label]}: {p * 100:.1f}%")


if __name__ == "__main__":
    main()
