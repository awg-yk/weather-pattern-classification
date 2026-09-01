"""天気図1枚を分類して、根拠の絵と一緒に見るための一式。

Colabのノートブックと、手元(VS Code)のノートブックの**両方から同じものを
呼ぶ**ためにここに置いてある。以前はノートブックの中に関数を書き写していたが、
それだと片方だけ直して食い違う。実際、この計画では「学習に使った描き方と
推論の描き方が食い違うと成績が静かに落ちる」という失敗をしているので、
描き方を決める場所は1つにしておく。

手元での使い方(notebooks/predict_local.ipynb):

    from src.quicklook import classify_and_show
    classify_and_show("path/to/chart.png", threshold=0.5, annotate=True)

古い天気図(2000〜2022年)を渡すときは、検出の設定を変える:

    classify_and_show(path, annotate=True, letter_size="auto", detect_threshold=0.55)

理由は README の「検出が0個になるとき」を参照。テンプレートは2023年以降の
天気図から切り出したものなので、古い天気図では大きさが3.2%違い、スコアも
少し下がる。
"""

import os
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_WEIGHTS = REPO_ROOT / "weights" / "model.pt"
# 検出した枠を描き込んだ画像で学習した重み。入力の見た目が違うので別名。
# **必ず annotate=True と組にすること。**素の天気図を渡すと、モデルは
# 見たことのない絵を受け取ることになり、成績が静かに落ちる。
ANNOT_WEIGHTS = REPO_ROOT / "weights" / "model_annot.pt"
TEMPLATES_DIR = REPO_ROOT / "data" / "templates"
MARKS_DIR = REPO_ROOT / "data" / "marks"


def annotation_available(annot_weights=ANNOT_WEIGHTS, templates=TEMPLATES_DIR):
    """注釈方式が使える状態か(重みとテンプレートが揃っているか)を返す。"""
    missing = [str(p) for p in (annot_weights, templates) if not os.path.exists(p)]
    return (not missing), missing


def make_annotated(image_path, out_path=None, *, templates=TEMPLATES_DIR,
                   marks=MARKS_DIR, letter_size=1.0, detect_threshold=0.65,
                   quiet=False):
    """検出した枠を描き込んだ画像を作り、そのパスを返す。

    **描き方は学習に使ったものと揃える。**同梱の重みは枠のみ(前線の縁取り
    なし)で作った画像で学習してあるので、ここも枠のみにする。
    """
    from scripts.annotate_charts import annotate_one
    from scripts.preprocess_jma import (DEFAULT_STAMP_BOX, autocrop_to_content,
                                        mask_stamp_box)

    image = Image.open(image_path).convert("RGB")
    image = mask_stamp_box(autocrop_to_content(image), DEFAULT_STAMP_BOX)
    marked, detections = annotate_one(
        np.array(image), templates,
        marks if marks and os.path.exists(marks) else None,
        letter_size=letter_size, threshold=detect_threshold,
        boxes=True, fronts=False,
    )
    out_path = Path(out_path) if out_path else Path(image_path).with_name(
        Path(image_path).stem + "_annotated.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(marked).save(out_path)
    if not quiet:
        edge = len(detections.edge_highs) + len(detections.edge_lows)
        print(f"検出: 高気圧 {len(detections.highs)}個 / "
              f"低気圧 {len(detections.lows)}個(中心が枠外の系 {edge}個)")
        print(f"注釈付き画像: {out_path}  ← 枠が本物の高低気圧に付いているか確かめること")
    return out_path


def classify_and_show(image_path, threshold=None, annotate=False, *,
                      weights=DEFAULT_WEIGHTS, annot_weights=ANNOT_WEIGHTS,
                      templates=TEMPLATES_DIR, marks=MARKS_DIR,
                      letter_size=1.0, detect_threshold=0.65, annotated_path=None):
    """画像1枚を分類し、確信度がthresholdを超えたラベル分だけヒートマップを表示、
    それ以外はテキストのみで確信度一覧を出す。

    annotate=True にすると、先に高低気圧を検出して枠を描き込み、注釈付き画像で
    学習した重みを使う。**Grad-CAMは「モデルがどこを見たか」しか示さないが、
    枠は「検出が当たったか」を示す。**別のことを示すので、両方あると読み解ける。
    """
    import matplotlib.pyplot as plt

    from scripts.gradcam import explain_predictions_above_threshold
    from src.labels import LABEL_JA

    used_weights = weights
    if annotate:
        ok, missing = annotation_available(annot_weights, templates)
        if not ok:
            print("注釈方式は使えません(見つからないもの: "
                  + ", ".join(os.path.basename(m) for m in missing) + ")")
            print("素の天気図の方式で続けます。")
            annotate = False
        else:
            image_path = make_annotated(
                image_path, annotated_path, templates=templates, marks=marks,
                letter_size=letter_size, detect_threshold=detect_threshold)
            used_weights = annot_weights

    display_image, overlays, ranked = explain_predictions_above_threshold(
        image_path=str(image_path),
        weights_path=str(used_weights),
        threshold=threshold,
        # 描き込み済みの画像には前処理を二重にかけない
        apply_preprocess=not annotate,
    )

    n_panels = len(overlays) + 1
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]
    axes[0].imshow(display_image)
    axes[0].set_title("検出結果(枠つき)" if annotate else "入力画像(前処理後)")
    axes[0].axis("off")
    for ax, (label, prob, overlay) in zip(axes[1:], overlays):
        ax.imshow(overlay)
        ax.set_title(f"{LABEL_JA[label]}\n({prob * 100:.1f}%)")
        ax.axis("off")
    plt.tight_layout()
    plt.show()

    if not overlays:
        shown = "校正ファイルのしきい値" if threshold is None else f"確信度{threshold * 100:.0f}%"
        print(f"{shown}を超えるラベルはありませんでした。\n")

    print("--- 全ラベルの確信度 ---")
    for label, prob in ranked:
        print(f"{LABEL_JA[label]}: {prob * 100:.1f}%")
    return ranked
