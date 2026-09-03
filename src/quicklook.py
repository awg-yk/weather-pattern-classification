"""天気図1枚を分類して、根拠の絵と一緒に見るための一式。

Colabのノートブックと、手元(VS Code)のノートブックの**両方から同じものを
呼ぶ**ためにここに置いてある。以前はノートブックの中に関数を書き写していたが、
それだと片方だけ直して食い違う。実際、この計画では「学習に使った描き方と
推論の描き方が食い違うと成績が静かに落ちる」という失敗をしているので、
描き方を決める場所は1つにしておく。

手元での使い方(notebooks/predict_local.ipynb):

    from src.quicklook import classify_and_show
    classify_and_show("path/to/chart.png", threshold=0.5, annotate=True)

**どの時代の天気図でも同じ呼び方でよい。**前処理が切り取ったあとに
1453x1500へ揃えるので(`scripts/preprocess_jma.CANONICAL_SIZE`)、時代に
よらず記号の大きさが同じになる。以前は2023年を境に設定を打ち分けていたが、
揃えるようにしてから不要になった。実測(2000-01-01の天気図、しきい値0.65・
テンプレート原寸): 揃える前 H 0 / L 0 -> 揃えた後 H 3 / L 4。
2023年以降の平均(H 2.8 / L 3.9〜4.2)と同じ水準。
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

# 前処理済みの天気図の置き場所。2000〜2025年をここに揃えてある
PROCESSED_DIR = REPO_ROOT / "data" / "processed" / "all"

# 注釈付き画像の書き出し先。**入力の隣には置かない。**
# data/processed/all に書くと、学習に使うフォルダに派生画像が混ざる
ANNOTATED_DIR = REPO_ROOT / "reports" / "annotated"

# 日時10桁 -> パス の索引を、フォルダごとに覚えておく。
# 値は (フォルダの更新時刻, 索引)。更新時刻が変わったら作り直す
_INDEX: dict = {}


def chart_for(date, hour: int = 0, images_dir=PROCESSED_DIR):
    """日付から天気図のパスを引く。

    ファイル名の表記は取得元で違う(`Js_2023010100.png` と
    `Js_2023010100_page001.png`)ので、10桁の日時で照合する。

    索引は一度作ったら使い回す。17,898枚を毎回走査すると遅い。ただし
    **フォルダの更新時刻が変わったら作り直す。**そうしないと、ノートブックを
    開いたまま画像を足したときに「ありません」と言い続ける。
    """
    from src.split import index_images_by_stamp

    key = str(images_dir)
    stat = Path(images_dir).stat().st_mtime if Path(images_dir).is_dir() else None
    if _INDEX.get(key, (None, None))[0] != stat:
        _INDEX[key] = (stat, index_images_by_stamp(images_dir))
        if not _INDEX[key][1]:
            try:
                shown = Path(images_dir).relative_to(REPO_ROOT)
            except ValueError:
                shown = Path(images_dir)
            raise SystemExit(
                f"天気図が1枚もありません: {images_dir}\n"
                "  python -m scripts.preprocess_jma "
                f"--in-dir data/raw/new_png --out-dir {shown} で作れます")

    stamp = f"{str(date).replace('-', '').replace('/', '')[:8]}{hour:02d}"
    found = _INDEX[key][1].get(stamp)
    if found is None:
        have = sorted(_INDEX[key][1])
        raise SystemExit(
            f"{date} {hour:02d}Z の天気図がありません(探した名前: {stamp})\n"
            f"  {images_dir} にあるのは {len(have)}枚、"
            f"{have[0][:8]} 〜 {have[-1][:8]}\n"
            "  00Z と 12Z しかありません。hour は 0 か 12 を指定してください")
    return Path(found)


# 検出の設定。**時代で打ち分けない。**前処理がすべての天気図を
# 1453x1500 に揃えるので、記号の大きさは時代によらず同じになる。
# ここは runs/cv_annot_boxes を作ったときの値で、同梱の重みはこの設定で
# 描いた画像で学習してある。**変えると、モデルに学習時と違う絵を渡すことになる。**
DETECTION = {"letter_size": 1.0, "detect_threshold": 0.65}


def annotation_available(annot_weights=ANNOT_WEIGHTS, templates=TEMPLATES_DIR):
    """注釈方式が使える状態か(重みとテンプレートが揃っているか)を返す。"""
    missing = [str(p) for p in (annot_weights, templates) if not os.path.exists(p)]
    return (not missing), missing


def make_annotated(image_path, out_path=None, *, templates=TEMPLATES_DIR,
                   marks=MARKS_DIR, letter_size=None, detect_threshold=None,
                   quiet=False):
    """検出した枠を描き込んだ画像を作り、そのパスを返す。

    **描き方は学習に使ったものと揃える。**同梱の重みは枠のみ(前線の縁取り
    なし)で作った画像で学習してあるので、ここも枠のみにする。
    """
    from scripts.annotate_charts import annotate_one
    from scripts.preprocess_jma import (DEFAULT_STAMP_BOX, autocrop_to_content,
                                        mask_stamp_box)

    letter_size = DETECTION["letter_size"] if letter_size is None else letter_size
    detect_threshold = (DETECTION["detect_threshold"]
                        if detect_threshold is None else detect_threshold)

    image = Image.open(image_path).convert("RGB")
    image = mask_stamp_box(autocrop_to_content(image), DEFAULT_STAMP_BOX)
    marked, detections = annotate_one(
        np.array(image), templates,
        marks if marks and os.path.exists(marks) else None,
        letter_size=letter_size, threshold=detect_threshold,
        boxes=True, fronts=False,
    )
    # **入力の隣には置かない。**data/processed/all に書くと、学習に使う
    # フォルダに派生画像が混ざり、次の学習で拾われかねない
    out_path = Path(out_path) if out_path else (
        ANNOTATED_DIR / (Path(image_path).stem + "_annotated.png"))
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
                      letter_size=None, detect_threshold=None, annotated_path=None):
    """画像1枚を分類し、確信度がthresholdを超えたラベル分だけヒートマップを表示、
    それ以外はテキストのみで確信度一覧を出す。

    annotate=True にすると、先に高低気圧を検出して枠を描き込み、注釈付き画像で
    学習した重みを使う。**Grad-CAMは「モデルがどこを見たか」しか示さないが、
    枠は「検出が当たったか」を示す。**別のことを示すので、両方あると読み解ける。

    **どの時代の天気図でも同じ呼び方でよい。**前処理がすべての天気図を同じ
    大きさに揃えるので、検出の設定は1つで足りる。
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


def classify_date(date, hour: int = 0, threshold=None, annotate: bool = True,
                  images_dir=PROCESSED_DIR, **kwargs):
    """日付を指定して、その日の天気図を分類する。

    `classify_and_show` に日付から画像を引く手間を足しただけ。
    どの天気図を見たのかが分かるよう、パスを表示する。
    """
    path = chart_for(date, hour, images_dir)
    print(f"天気図: {path.name}  ({date} {hour:02d}Z)")
    return classify_and_show(path, threshold=threshold, annotate=annotate, **kwargs)
