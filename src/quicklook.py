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


def detect_on(image, *, templates=TEMPLATES_DIR, marks=MARKS_DIR,
              letter_size=None, detect_threshold=None):
    """前処理済みの画像1枚から検出結果だけを取り出す(描き込みはしない)。"""
    from scripts.annotate_charts import annotate_one

    letter_size = DETECTION["letter_size"] if letter_size is None else letter_size
    detect_threshold = (DETECTION["detect_threshold"]
                        if detect_threshold is None else detect_threshold)
    _marked, detections = annotate_one(
        np.array(image), str(templates),
        str(marks) if marks and os.path.exists(marks) else None,
        letter_size=letter_size, threshold=detect_threshold,
        boxes=False, fronts=False,
    )
    return detections


def show_site(date, hour: int = 0, site="komatsu", *, radius=None, grid=False,
              images_dir=PROCESSED_DIR, sites_path=None, figsize=(9, 9),
              letter_size=None, detect_threshold=None):
    """日付と地点を指定して、円と検出結果を描いた天気図をその場に表示する。

    円の内側の系は色付き(高=緑、低=赤)、外側は灰色で描く。
    `scripts/site_report.py` が数えているものを、そのまま目で確かめるためのもの。
    描き方は `src/sites.py` の1か所から呼んでいるので、コマンド側とずれない。

    grid=True にすると相対座標の目盛りを重ねる。地点の位置を直すときに使う。
    """
    import matplotlib.pyplot as plt

    from scripts.preprocess_jma import (CANONICAL_SIZE, DEFAULT_STAMP_BOX,
                                        autocrop_to_content, fit_to_canonical,
                                        mask_stamp_box)
    from src.sites import (draw_detections, draw_grid, draw_site, get_site,
                           summarize)

    found_site = get_site(site, sites_path) if isinstance(site, str) else site
    if radius is not None:
        found_site = type(found_site)(name=found_site.name, x=found_site.x,
                                      y=found_site.y, radius=radius,
                                      note=found_site.note)

    path = chart_for(date, hour, images_dir)
    image = Image.open(path).convert("RGB")
    # 前処理済みの画像にもう一度かけても結果は変わらない(実測で画素一致)
    image = mask_stamp_box(fit_to_canonical(autocrop_to_content(image), CANONICAL_SIZE),
                           DEFAULT_STAMP_BOX)

    detections = detect_on(image, letter_size=letter_size,
                           detect_threshold=detect_threshold)
    found = summarize(detections, found_site)

    if grid:
        image = draw_grid(image)
    image = draw_detections(image, detections, found_site)
    image = draw_site(image, found_site, text=found_site.name)

    plt.figure(figsize=figsize)
    plt.imshow(image)
    plt.title(f"{date} {hour:02d}Z  {found_site.name}"
              f"(半径 {found_site.radius})")
    plt.axis("off")
    plt.show()

    print(f"天気図: {path.name}")
    print(f"全体      : 高気圧 {found['n_high_all']}個 / 低気圧 {found['n_low_all']}個")
    print(f"{found_site.name} の周辺: 高気圧 {found['n_high']}個 / "
          f"低気圧 {found['n_low']}個")
    if found["nearest"]:
        near = found["nearest"]
        print(f"最も近い系: {near['kind']}  距離 {near['distance']:.3f}")
    elif found["n_high_all"] or found["n_low_all"]:
        # **「周辺に無い」=「誤り」ではない。**冬型のように広域の配置で決まる
        # 気圧配置では、地点の近くにあるのは混んだ等圧線であって中心ではない
        # (中心はシベリアと東海上)。前線通過・停滞前線も、前線は中心ではない
        # ので周辺は0になる。ラベルと併せて読むこと
        print("周辺に高気圧・低気圧の中心はありません。")
        print("  冬型・前線のように広域の配置で決まる気圧配置なら、これで正常です。")
        print("  日本海低気圧など『近くの系』が定義のラベルなら、要確認です。")
    else:
        print("そもそも1つも検出できていません(検出漏れを疑うこと)")
    return found


def site_scorer(*, annotate: bool = True, weights=DEFAULT_WEIGHTS,
                annot_weights=ANNOT_WEIGHTS, templates=TEMPLATES_DIR,
                marks=MARKS_DIR):
    """モデルとGrad-CAMを1回だけ用意する。

    **167日ぶんを回すときに効く。**1日ごとに重みを読み直すと、その時間だけで
    大半を使ってしまう。1枚だけ見るときも同じ入り口を通す(書き分けると
    片方だけ直して食い違う)。
    """
    import torch

    from scripts.gradcam import GradCAM, _load_model
    from src.train import get_transforms

    # **描き込みと重みは必ず組にする。**注釈付き画像で学習した重みに素の
    # 天気図を渡すと、モデルは見たことのない絵を受け取ることになる
    used_weights = weights
    if annotate:
        ok, missing = annotation_available(annot_weights, templates)
        if not ok:
            print("注釈方式は使えません(見つからないもの: "
                  + ", ".join(os.path.basename(m) for m in missing) + ")。"
                  "素の天気図の方式で続けます。")
            annotate = False
        else:
            used_weights = annot_weights

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = _load_model(str(used_weights), device)
    return {
        "model": model, "gradcam": GradCAM(model), "device": device,
        "transform": get_transforms(train=False, image_size=meta["image_size"]),
        "weights": str(used_weights), "annotate": annotate,
        "templates": templates, "marks": marks,
    }


def score_chart(path, site, scorer, *, mode: str = "proximity", scale=None,
                attention_weight: float = 1.0, keep_cam: bool = False):
    """天気図1枚を採点する。`(使った画像, ラベルごとの行)` を返す。

    行には 確信度(prob)・熱の割合(mass)・集中度(lift)・点数(score) が入る。
    """
    from scripts.gradcam import _probabilities
    from scripts.preprocess_jma import (CANONICAL_SIZE, DEFAULT_STAMP_BOX,
                                        autocrop_to_content, fit_to_canonical,
                                        mask_stamp_box)
    from src.labels import LABELS
    from src.sites import (attention_lift, attention_mass, attention_proximity,
                           proximity_lift)

    if mode not in ("proximity", "circle"):
        raise ValueError(f"mode は 'proximity' か 'circle' です: {mode!r}")

    image = Image.open(path).convert("RGB")
    image = mask_stamp_box(fit_to_canonical(autocrop_to_content(image), CANONICAL_SIZE),
                           DEFAULT_STAMP_BOX)
    if scorer["annotate"]:
        marked = make_annotated(path, templates=scorer["templates"],
                                marks=scorer["marks"], quiet=True)
        image = Image.open(marked).convert("RGB")

    tensor = scorer["transform"](image).unsqueeze(0).to(scorer["device"])
    probs = _probabilities(scorer["model"], tensor, scorer["weights"])

    rows = []
    for index, label in enumerate(LABELS):
        cam = scorer["gradcam"].generate(tensor, index)
        if mode == "proximity":
            mass = attention_proximity(cam, site, scale)
            lift = proximity_lift(cam, site, scale)
        else:
            mass = attention_mass(cam, site)
            lift = attention_lift(cam, site)
        prob = float(probs[index])
        row = {"label": label, "prob": prob, "mass": mass, "lift": lift,
               "score": prob * (mass ** attention_weight)}
        if keep_cam:
            row["cam"] = cam
        rows.append(row)
    return image, rows


def rerank_by_site(date, hour: int = 0, site="komatsu", *, attention_weight: float = 1.0,
                   mode: str = "proximity", scale=None,
                   radius=None, annotate: bool = True, top_k: int = 3,
                   images_dir=PROCESSED_DIR, sites_path=None,
                   weights=DEFAULT_WEIGHTS, annot_weights=ANNOT_WEIGHTS,
                   templates=TEMPLATES_DIR, marks=MARKS_DIR,
                   show: bool = True, figsize=(15, 5)):
    """Grad-CAMの熱が地点の円にどれだけ入っているかで、ラベルの順位を付け替える。

    考え方
    ------
    モデルは日本全体を見て判断するので、その地点に関係のない気圧配置を1位に
    出すことがある。一方 Grad-CAM は「モデルがどこを見て、そのラベルを出したか」
    を示す。**そのラベルの根拠が地点の近くに無いなら、その地点の現象の説明
    としては弱い。**そこで

        新しい点数 = 確信度 × (近さで重み付けした熱の割合 ** attention_weight)

    で並べ替える。`attention_weight=0` なら元の順位のまま、`1` で全面的に
    効かせる。

    mode
    ----
    "proximity"(既定) 地点からの距離で滑らかに重み付けする。
                      w = exp(-(距離 / scale) ** 2)、scale の既定は半径。
    "circle"          円の中だけを1、外を0にする(以前の方法)。

    **既定を距離にしてある。**円は境界のすぐ外を全部捨てる。小松のおろし風
    167日では、半径0.12の円に高低気圧が1つも入らない日が85.6%あり、その
    大半は「円が天気図の4.5%しかないから」で説明がついてしまった。
    距離0.13(円のすぐ外)の熱は、円では0.000、距離の重みでは0.300になる。
    両者の目盛りは揃えてあるので(src/sites.proximity_weight を参照)、
    集中度(lift)は同じ意味で読み比べられる。

    **注意: これは冬型のような広域の配置で決まるラベルを不利にする。**
    冬型の根拠は日本全体に広がる等圧線なので、円の中の割合は小さくなる。
    「その地点の近くにある系で説明したい」という用途に限って使うこと。

    戻り値は [(ラベル, 確信度, 熱の割合, 点数)] を点数の高い順に並べたもの。
    """
    import matplotlib.pyplot as plt

    from scripts.gradcam import _overlay_heatmap
    from src.labels import LABEL_JA
    from src.sites import draw_site, get_site

    found_site = get_site(site, sites_path) if isinstance(site, str) else site
    if radius is not None:
        found_site = type(found_site)(name=found_site.name, x=found_site.x,
                                      y=found_site.y, radius=radius,
                                      note=found_site.note)

    scorer = site_scorer(annotate=annotate, weights=weights,
                         annot_weights=annot_weights, templates=templates,
                         marks=marks)
    annotate = scorer["annotate"]
    path = chart_for(date, hour, images_dir)
    image, rows = score_chart(path, found_site, scorer,
                              mode=mode, scale=scale,
                              attention_weight=attention_weight, keep_cam=True)

    before = sorted(rows, key=lambda r: -r["prob"])
    after = sorted(rows, key=lambda r: -r["score"])
    rank_before = {r["label"]: i + 1 for i, r in enumerate(before)}

    if show:
        overlays = [r for r in before[:top_k]]
        panels = len(overlays) + 1
        fig, axes = plt.subplots(1, panels, figsize=figsize)
        axes = list(np.atleast_1d(axes))
        axes[0].imshow(draw_site(image, found_site, text=found_site.name))
        axes[0].set_title(f"{date} {hour:02d}Z\n{found_site.name}"
                          f"(半径 {found_site.radius})")
        axes[0].axis("off")
        for ax, row in zip(axes[1:], overlays):
            overlay = _overlay_heatmap(image, row["cam"])
            ax.imshow(draw_site(overlay, found_site))
            ax.set_title(f"{LABEL_JA[row['label']]}\n"
                         f"確信度 {row['prob'] * 100:.1f}% / "
                         f"集中度 {row['lift']:.2f}")
            ax.axis("off")
        plt.tight_layout()
        plt.show()

    used_scale = found_site.radius if scale is None else scale
    how = (f"地点からの近さ(exp(-(距離/{used_scale})^2))で重み付け"
           if mode == "proximity" else f"円(半径 {found_site.radius})の中だけ")
    print(f"天気図: {path.name}   重み: {'注釈付き' if annotate else '素'}")
    print(f"熱の測り方: {how}")
    print(f"点数 = 確信度 × (熱の割合 ** {attention_weight})\n")
    print(f"  {'順位':<4}{'ラベル':<24}{'確信度':>8}{'熱の割合':>9}"
          f"{'集中度':>8}{'点数':>9}  元の順位")
    print("  " + "-" * 70)
    for new_rank, row in enumerate(after, start=1):
        moved = rank_before[row["label"]] - new_rank
        arrow = f"  {rank_before[row['label']]}位"
        if moved > 0:
            arrow += f" (+{moved})"
        elif moved < 0:
            arrow += f" ({moved})"
        print(f"  {new_rank:<4}{LABEL_JA[row['label']]:<24}"
              f"{row['prob'] * 100:>7.1f}%{row['mass'] * 100:>7.0f}%"
              f"{row['lift']:>8.2f}{row['score']:>9.4f}{arrow}")

    top_before, top_after = before[0]["label"], after[0]["label"]
    print()
    if top_before == top_after:
        print(f"1位は変わりません: {LABEL_JA[top_after]}")
    else:
        print(f"1位が入れ替わりました: {LABEL_JA[top_before]} -> {LABEL_JA[top_after]}")
    print("  ※集中度(lift)は、1なら画像全体を一様に見ているのと同じ、"
          "1より大きいほど地点の近くに集中している。")
    print("  ※冬型・前線のように広域の配置で決まるラベルは、根拠が日本全体に"
          "広がるため不利に働きます。")

    return [(r["label"], r["prob"], r["mass"], r["score"]) for r in after]


def rerank_dates(dates, site="komatsu", hour: int = 0, *, attention_weight: float = 1.0,
                 mode: str = "proximity", scale=None, radius=None,
                 annotate: bool = True, date_column: str = "発生日",
                 images_dir=PROCESSED_DIR, sites_path=None,
                 weights=DEFAULT_WEIGHTS, annot_weights=ANNOT_WEIGHTS,
                 templates=TEMPLATES_DIR, marks=MARKS_DIR,
                 out=None, progress_every: int = 10):
    """日付の一覧をまとめて採点し、順位の入れ替わりを表にする。

    `dates` は日付の並びか、日付の入ったCSVのパス。
    戻り値は1日1行の DataFrame。`out` を渡すとCSVに書き出す。

    **モデルは1回だけ読み込む。**167日ぶんを1日ずつ読み直すと、その時間で
    大半を使ってしまう。
    """
    import time

    import pandas as pd

    from src.labels import LABEL_JA
    from src.sites import get_site

    found_site = get_site(site, sites_path) if isinstance(site, str) else site
    if radius is not None:
        found_site = type(found_site)(name=found_site.name, x=found_site.x,
                                      y=found_site.y, radius=radius,
                                      note=found_site.note)

    if isinstance(dates, (str, Path)):
        table = pd.read_csv(dates)
        if date_column not in table.columns:
            raise SystemExit(
                f"列 '{date_column}' がありません。列: {list(table.columns)}")
        parsed = pd.to_datetime(table[date_column], errors="coerce").dropna()
        wanted = [d.strftime("%Y-%m-%d") for d in parsed]
    else:
        wanted = [str(d) for d in dates]

    scorer = site_scorer(annotate=annotate, weights=weights,
                         annot_weights=annot_weights, templates=templates,
                         marks=marks)
    used_scale = found_site.radius if scale is None else scale
    print(f"地点: {found_site.name}(中心 {found_site.x}, {found_site.y})")
    how = (f"距離で重み付け exp(-(距離/{used_scale})^2)" if mode == "proximity"
           else f"円(半径 {found_site.radius})の中だけ")
    print(f"熱の測り方: {how}")
    print(f"点数 = 確信度 × (熱の割合 ** {attention_weight})")
    print(f"重み: {'注釈付き' if scorer['annotate'] else '素'}")
    print(f"対象: {len(wanted)}日\n")

    rows, missing = [], []
    started = time.time()
    for done, date in enumerate(wanted, start=1):
        try:
            path = chart_for(date, hour, images_dir)
        except SystemExit:
            missing.append(date)
            continue
        _image, scored = score_chart(path, found_site, scorer, mode=mode,
                                     scale=scale, attention_weight=attention_weight)
        before = max(scored, key=lambda r: r["prob"])
        after = max(scored, key=lambda r: r["score"])
        rows.append({
            date_column: date,
            "元の1位": LABEL_JA[before["label"]],
            "元の確信度": round(before["prob"], 4),
            "元の1位の集中度": round(before["lift"], 3),
            "新しい1位": LABEL_JA[after["label"]],
            "新しい1位の確信度": round(after["prob"], 4),
            "新しい1位の集中度": round(after["lift"], 3),
            "点数": round(after["score"], 5),
            "入れ替わった": before["label"] != after["label"],
            "filename": Path(path).name,
        })
        if done % progress_every == 0 or done == len(wanted):
            rate = (time.time() - started) / done
            print(f"  {done}/{len(wanted)}  {rate:.1f}秒/枚  "
                  f"残り{rate * (len(wanted) - done) / 60:.0f}分", flush=True)

    if missing:
        print(f"\n天気図が無い日: {len(missing)}日(集計から外しました)")
    if not rows:
        raise SystemExit("採点できた日がありません")

    result = pd.DataFrame(rows)
    changed = int(result["入れ替わった"].sum())
    print(f"\n{'=' * 56}\n1位が入れ替わった日: {changed}日 / {len(result)}日"
          f"({changed / len(result) * 100:.1f}%)\n{'=' * 56}")

    print("\n  1位のラベルの内訳")
    print(f"  {'ラベル':<24}{'元':>6}{'新':>6}{'差':>7}")
    print("  " + "-" * 44)
    before_counts = result["元の1位"].value_counts()
    after_counts = result["新しい1位"].value_counts()
    for label in sorted(set(before_counts.index) | set(after_counts.index),
                        key=lambda l: -int(before_counts.get(l, 0))):
        b = int(before_counts.get(label, 0))
        a = int(after_counts.get(label, 0))
        print(f"  {label:<24}{b:>6}{a:>6}{a - b:>+7}")

    moved = result[result["入れ替わった"]]
    if len(moved):
        print("\n  入れ替わりの内訳(多い順)")
        pairs = moved.groupby(["元の1位", "新しい1位"]).size().sort_values(ascending=False)
        for (old, new), count in pairs.head(10).items():
            print(f"    {old} -> {new}  {count}日")

    print("\n  ※集中度(lift)は、1なら画像全体を一様に見ているのと同じ、"
          "1より大きいほど地点の近くに集中している。")
    print("  ※冬型・前線のように広域の配置で決まるラベルは、根拠が日本全体に"
          "広がるため不利に働きます。")

    if out:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Excelでそのまま開けるようBOM付きUTF-8で書く
        result.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"\n書き出しました: {out_path}")
    return result
