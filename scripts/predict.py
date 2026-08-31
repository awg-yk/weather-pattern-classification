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
import sys
from pathlib import Path

# `python scripts/predict.py ...` で起動された場合、sys.path に入るのは scripts/
# だけなので、この下の `from scripts...` / `from src...` が解決できない。
# リポジトリのルートを自分で足しておく(`python -m scripts.predict` なら既に
# 入っているので、この行は何もしない)。
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
from PIL import Image

from scripts.preprocess_jma import DEFAULT_STAMP_BOX, autocrop_to_content, mask_stamp_box
from src import calibration as calib
from src.labels import INDEX_TO_LABEL, LABEL_JA
from src.chartscale import letter_size_arg
from src.model import load_model
from src.train import get_transforms

DEFAULT_WEIGHTS = _ROOT / "weights" / "model.pt"
# テンプレートと印はリポジトリに同梱されている。カレントディレクトリが
# どこであっても見つかるよう、絶対パスを既定にする。
DEFAULT_TEMPLATES = _ROOT / "data" / "templates"
DEFAULT_MARKS = _ROOT / "data" / "marks"


def maybe_annotate(image, args):
    """--annotate が指定されていれば、検出した枠を描き込んだ画像を返す。

    **描き方は学習に使ったものと揃える必要がある。**`runs/cv_annot_boxes` は
    `--no-fronts`(枠のみ)で作った画像で学習したので、ここも枠のみにする。
    食い違うと、モデルは見たことのない絵を渡されて静かに成績が落ちる。
    """
    if not args.annotate:
        return image

    import numpy as np

    from scripts.annotate_charts import annotate_one

    marked, detections = annotate_one(
        np.array(image), args.templates, args.marks, boxes=True, fronts=False,
        letter_size=args.letter_size,
    )
    from src.chartsymbols import ink_mask

    # 固定の色帯が読めずに控えへ切り替わったなら、黙っていてはいけない。
    # 色を見ないぶん海岸線も拾うので、枠が増えたり外れたりしうる
    _, fell_back = ink_mask(np.array(image))
    if fell_back:
        print("※ 等圧線の色帯が読めませんでした(紙のスキャンなど)。"
              "濃さのしきい値に切り替えています。")
        print("  色を見ないので海岸線も線として拾います。枠の位置を必ず目で確かめてください。")
    print(f"検出: 高気圧 {len(detections.highs)}個 / 低気圧 {len(detections.lows)}個"
          f"(枠外の系 {len(detections.edge_highs) + len(detections.edge_lows)}個)")
    out = Image.fromarray(marked)
    if args.save_annotated:
        path = Path(args.save_annotated)
        path.parent.mkdir(parents=True, exist_ok=True)
        out.save(path)
        print(f"注釈付き画像: {path}  ← 枠が本物の高低気圧に付いているか確かめること")
    return out


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?", help="分類したい天気図画像のパス(--dateを使う場合は不要)")
    parser.add_argument("--date", help="YYYY-MM-DD形式。指定すると気象庁アーカイブから直接取得する")
    parser.add_argument("--hour", type=int, default=0, choices=[0, 12], help="--date指定時のUTC時刻(0または12)")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="モデルの重みファイル")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="このしきい値を超えたラベルを表示。既定は校正ファイルに入っている"
        "ラベルごとのしきい値(校正ファイルが無い場合は一律0.5)",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="校正前の生の確信度を表示する。校正の前後を見比べるとき以外は不要"
        "(生の値は学習時のpos_weightのぶん高く出る。src/calibration.py を参照)",
    )
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
    parser.add_argument(
        "--annotate", action="store_true",
        help="検出した高低気圧の枠を描き込んでから分類する。"
             "**注釈付き画像で学習した重みを使うときは必須**(--weights weights/model_annot.pt)",
    )
    parser.add_argument("--templates", default=str(DEFAULT_TEMPLATES),
                        help="--annotate で使う H/L のテンプレート")
    parser.add_argument("--marks", default=str(DEFAULT_MARKS),
                        help="--annotate で使う中心の印。無ければ検出が減る")
    parser.add_argument("--letter-size", type=letter_size_arg, default=1.0,
                        help="H/Lのテンプレートを縮める倍率。auto で天気図の幅から"
                             "自動で決める(data/templates/reference.json が要る)。"
                             "**解像度の違いには効くが、記号の線の太さが違う天気図"
                             "(国会図書館由来の2000〜2022年)には効かない。**"
                             "そちらはテンプレートの切り直しが要る")
    parser.add_argument("--save-annotated", default=None,
                        help="注釈付き画像の保存先。検出が当たっているか目で確かめられる")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.date:
        from scripts.fetch_and_predict import fetch_chart

        args.image = str(fetch_chart(args.date, hour=args.hour))
        print(f"取得: {args.image}\n")
    elif not args.image:
        parser.error("画像パスまたは --date のどちらかを指定してください")

    # 確信度の校正(<重み名>.calib.json)。無ければ生の値のまま動く。
    # --raw のときは校正なしの Calibration を使うので、以降の経路は同じ形になる。
    calibration = calib.Calibration.identity() if args.raw else calib.load_for_weights_cli(args.weights)
    thresholds = {
        label: (args.threshold if args.threshold is not None else calibration[label].threshold)
        for label in INDEX_TO_LABEL.values()
    }

    if args.save_gradcam:
        from scripts.gradcam import explain_top_predictions

        out_dir = Path(args.save_gradcam)
        out_dir.mkdir(parents=True, exist_ok=True)

        # --annotate と併用するときは、先に描き込んだ画像を作ってから渡す。
        # Grad-CAM は「モデルがどこを見たか」、注釈は「検出が当たったか」で
        # 別のことを示すので、両方あると読み解きやすい
        gradcam_source = args.image
        if args.annotate:
            source = Image.open(args.image).convert("RGB")
            if not args.no_preprocess:
                source = autocrop_to_content(source)
                source = mask_stamp_box(source, DEFAULT_STAMP_BOX)
            marked = maybe_annotate(source, args)
            gradcam_source = str(Path(args.save_gradcam) / "_annotated.png")
            Path(args.save_gradcam).mkdir(parents=True, exist_ok=True)
            marked.save(gradcam_source)

        display_image, top_overlays, ranked = explain_top_predictions(
            image_path=gradcam_source,
            weights_path=args.weights,
            top_k=args.top_k,
            # 描き込み済みの画像には前処理を二重にかけない
            apply_preprocess=(not args.no_preprocess) and not args.annotate,
            calibration=calibration,
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
        model, meta = load_model(args.weights, map_location=device)
        model.to(device)
        model.eval()

        image = Image.open(args.image).convert("RGB")
        if not args.no_preprocess:
            image = autocrop_to_content(image)
            image = mask_stamp_box(image, DEFAULT_STAMP_BOX)
        # 描き込みは前処理のあと。学習に使った画像も前処理済みのものから作った
        image = maybe_annotate(image, args)

        transform = get_transforms(train=False, image_size=meta["image_size"])
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(input_tensor)[0].cpu().numpy()

        probs = calibration.probabilities(logits)

        ranked = sorted(
            ((INDEX_TO_LABEL[i], float(p)) for i, p in enumerate(probs)),
            key=lambda x: x[1],
            reverse=True,
        )

    predicted = [label for label, p in ranked if p > thresholds[label]]
    print("予測:", " / ".join(LABEL_JA[l] for l in predicted) if predicted else "該当なし")
    if not predicted:
        top_label, top_prob = ranked[0]
        print(
            f"  どのラベルもしきい値に届きませんでした。最も高いのは"
            f"{LABEL_JA[top_label]}({top_prob * 100:.1f}% / しきい値"
            f"{thresholds[top_label] * 100:.1f}%)ですが、この確信度では"
            "自動判定に使わず人が見た方が確実です。"
        )
    print()
    print("--- 全ラベルの確信度" + ("(校正前の生の値)" if args.raw else "") + " ---")
    for label, p in ranked:
        # cp932(日本語Windowsのコンソール)で出せる文字だけを使う。
        # U+2713 のチェックマークはそこで UnicodeEncodeError になる
        mark = " <-該当" if p > thresholds[label] else ""
        print(f"{LABEL_JA[label]}: {p * 100:.1f}%(しきい値 {thresholds[label] * 100:.1f}%){mark}")


if __name__ == "__main__":
    main()
