"""
リポジトリに同梱された学習済みモデル(weights/model.pt)を使って、
天気図画像1枚を分類するコマンドラインツール。学習は行わない。

使い方:
    python scripts/predict.py path/to/chart.png

    # 既に前処理済み(余白クロップ・日時スタンプ消去済み)の画像の場合
    python scripts/predict.py path/to/chart.png --no-preprocess

    # 確信度上位3件について、モデルが注目した箇所をヒートマップ画像として保存する
    python scripts/predict.py path/to/chart.png --save-gradcam out_dir/

    # 画像を用意する代わりに、気象庁アーカイブから日付指定で直接取得して分類する
    python scripts/predict.py --date 2025-01-01 --hour 0
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
    parser.add_argument("image", nargs="?", help="分類したい天気図画像のパス(--dateを使う場合は不要)")
    parser.add_argument("--date", help="YYYY-MM-DD形式。指定すると気象庁アーカイブから直接取得する")
    parser.add_argument("--hour", type=int, default=0, choices=[0, 12], help="--date指定時のUTC時刻(0または12)")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="モデルの重みファイル")
    parser.add_argument("--threshold", type=float, default=0.5, help="このしきい値を超えたラベルを表示")
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="気象庁の生画像向け前処理(余白クロップ・日時スタンプ消去)をスキップする",
    )
    parser.add_argument(
        "--save-gradcam",
        metavar="OUT_DIR",
        help="指定したディレクトリに、確信度上位3件のGrad-CAMヒートマップ画像を保存する",
    )
    parser.add_argument("--top-k", type=int, default=3, help="--save-gradcam で保存する件数")
    args = parser.parse_args()

    if args.date:
        from scripts.fetch_and_predict import fetch_chart

        args.image = str(fetch_chart(args.date, hour=args.hour))
        print(f"取得: {args.image}\n")
    elif not args.image:
        parser.error("画像パスまたは --date のどちらかを指定してください")

    if args.save_gradcam:
        from scripts.gradcam import explain_top_predictions

        out_dir = Path(args.save_gradcam)
        out_dir.mkdir(parents=True, exist_ok=True)

        display_image, top_overlays, ranked = explain_top_predictions(
            image_path=args.image,
            weights_path=args.weights,
            top_k=args.top_k,
            apply_preprocess=not args.no_preprocess,
        )
        for rank, (label, prob, overlay) in enumerate(top_overlays, start=1):
            out_path = out_dir / f"{rank}_{label}.png"
            overlay.save(out_path)
            print(f"saved: {out_path} ({LABEL_JA[label]}, {prob * 100:.1f}%)")
        print()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # pretrained=False: どうせ直後に自前の学習済み重みで上書きするので、
        # ImageNet事前学習済み重みのダウンロードは不要(ネット接続なしでも動かせる)
        model = build_model(num_classes=len(LABELS), pretrained=False)
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
