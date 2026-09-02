r"""検出した高低気圧と前線を天気図に描き込み、注釈付きの画像を書き出す。

    天気図 -> 検出(方法③のSTEP1・STEP2) -> 描き込み -> 注釈付き画像 -> CNN(方法①)

狙いは2つある。

1. **CNNに、検出した位置を目立つ形で渡す。**H/L の文字も前線も元の天気図に
   既に描かれているので、これは情報の追加ではなく**強調**である。効くかどうかは
   実測で確かめる必要がある(下記の但し書き)。
2. **人が検出の当たり外れを目で確かめられる。**Grad-CAM は「モデルがどこを
   見たか」しか示さないが、この画像は「検出が正しかったか」を示す。運用で
   AIの判定を人が検証する場面ではこちらのほうが直接的である。

学習側の変更は要らない。`--data-dir` をこの出力に向けるだけで
`scripts/cross_validate.py` がそのまま動く。

**期待しすぎないための但し書き**
--------------------------------
枠を描いても、CNNが位置を絶対座標で読めるようにはならない。畳み込みは
平行移動に対して同等の応答を返すので、「枠がオホーツク海にある」ことは
依然として読み取りにくい。**効くとすれば「見つけやすくなる」ほうであって、
「どこにあるか分かる」ほうではない。**

検出は完璧ではない(1枚あたり約2.25個の誤検出が残る)。補助の答えとして
渡す場合は損失に混じるだけだが、**画像に描き込むと誤りが入力そのものに
焼き付く。**モデルはそれを本物として扱う。

**`--marks` を必ず渡すこと。**中心の×との裏書きが無いと、等圧線に横切られた
H/L が拾えない(`analyse_chart` の letter_threshold は印がある場合にだけ効く)。
実測で、印なしでは1枚あたり3個しか拾えず、印ありでは6個拾えた。

使い方:
    python -m scripts.annotate_charts --in-dir data\processed\jma `
        --templates data\templates --marks data\marks `
        --out-dir data\annotated --workers 8
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import time

import cv2
import numpy as np
from PIL import Image

from scripts.build_features import (
    IMAGE_SUFFIXES,
    LETTER_THRESHOLD,
    analyse_chart,
    load_templates_scaled,
    warn_about_misplaced_marks,
)
from src.chartfeatures import MARK_LETTER_RADIUS
from src.chartscale import (auto_letter_size, letter_size_arg,
                            sizes_around)

# 描き込みの色(RGB)。**天気図に既にある色帯と重ならないものを選ぶ。**
#
# 天気図側は 等圧線=黒、温暖前線=赤(252,4,4)、寒冷前線=青(4,4,252)、
# 閉塞前線=紫、海岸線と経緯線=赤茶(164,44,44)。`src/chartsymbols.DEFAULT_BANDS`
# の色相帯で言うと、空いているのは**色相21〜99(橙〜黄〜緑〜シアン)だけ**である。
#
# 当初 紫(200,0,200) を低気圧の枠に使おうとしたが、閉塞前線の帯に入っていた。
# 重なると、人が見て「元からある前線」か「描き込み」か区別できないうえ、
# この画像に対して方法③の色マスクをもう一度かけられなくなる。
# `tests/test_annotate.py` が全色を機械的に確かめている。
HIGH_COLOR = (0, 150, 0)            # 高気圧の枠(緑・色相60)
LOW_COLOR = (255, 130, 0)           # 低気圧の枠(橙・色相15)
FRONT_COLORS = {
    "warm_front": (240, 210, 0),        # 黄(色相26)
    "cold_front": (0, 210, 220),        # シアン(色相91)
    "occluded_front": (140, 220, 0),    # 黄緑(色相41)
    "stationary_front": (0, 170, 150),  # 青緑(色相86)
}

# 枠の大きさ(相対座標)。H/L の文字は約95x117、天気図は約1450x1500 なので
# 文字自体は 0.066 x 0.078 にあたる。**文字を囲めるよう、少し大きめに取る。**
# 文字の内側に描くと、人が見たときに「何を検出したのか」が読み取りにくい
BOX_W, BOX_H = 0.078, 0.092


def draw_annotations(rgb: np.ndarray, report: dict, detections,
                     boxes: bool = True, fronts: bool = True,
                     thickness: int = 3) -> np.ndarray:
    """検出を天気図に描き込んだ画像を返す。元の配列は変更しない。

    **輪郭だけを描き、塗りつぶさない。**下の天気図が隠れると、CNNが読める
    情報を減らすことになる。強調のつもりが情報の削除になっては本末転倒である。
    """
    out = rgb.copy()
    height, width = out.shape[:2]

    if fronts and report.get("masks"):
        # **前線そのものは塗り替えず、周りに縁取りだけを描く。**
        # 塗り替えると、天気図が元から持っている赤(温暖)・青(寒冷)の
        # 区別が消える。強調のつもりが情報の削除になる。縁取りなら、
        # 元の色を残したまま「ここを検出した」を示せる。
        #
        # 停滞前線は赤と青の交互という導出結果なので最後に描く(最も具体的)。
        order = ["warm_front", "cold_front", "occluded_front", "stationary_front"]
        found = {name: report["masks"].get(name) for name in order}
        found = {k: v for k, v in found.items() if v is not None and v.any()}
        # **全部の前線の元画素をまとめて守る。**自分のマスクだけ避けると、
        # あとから描く縁取りが別の前線の元画素を塗ってしまう(実測で1割)
        protected = np.zeros(out.shape[:2], dtype=bool)
        for mask in found.values():
            protected |= mask
        for name, mask in found.items():
            grown = cv2.dilate(mask.astype(np.uint8),
                               np.ones((thickness * 2 + 1,) * 2, np.uint8)).astype(bool)
            out[grown & ~protected] = FRONT_COLORS[name]

    if boxes:
        # 縁にある文字も描く。**位置としては使えないが、検出はできている。**
        # `split_by_edge` が highs/lows から外すのは「文字の位置は中心では
        # ないので矩形の内外判定に使うと嘘になる」ためであって、そこに系が
        # 無いという意味ではない。描かないと、拾えているのに拾えていないように
        # 見える(実測で、図の上端にある H が2つ漏れた)。
        # 細い線で描いて、中心として信用できるものと見分けられるようにする。
        for points, color, width_px in (
            (detections.highs, HIGH_COLOR, thickness),
            (detections.lows, LOW_COLOR, thickness),
            (detections.edge_highs, HIGH_COLOR, max(1, thickness - 2)),
            (detections.edge_lows, LOW_COLOR, max(1, thickness - 2)),
        ):
            for cx, cy in points:
                x0 = int((cx - BOX_W / 2) * width)
                y0 = int((cy - BOX_H / 2) * height)
                x1 = int((cx + BOX_W / 2) * width)
                y1 = int((cy + BOX_H / 2) * height)
                cv2.rectangle(out, (x0, y0), (x1, y1), color, width_px)
    return out


def annotate_one(rgb: np.ndarray, templates_dir, marks_dir=None, *,
                 scale: float = 0.7, letter_size=1.0,
                 mark_scale: float = 1.0,
                 mark_radius: float = MARK_LETTER_RADIUS,
                 threshold: float = 0.65,
                 letter_threshold: float = LETTER_THRESHOLD,
                 angle_range: float = 60.0, angle_step: float = 5.0,
                 boxes: bool = True, fronts: bool = False,
                 thickness: int = 3) -> tuple:
    """1枚を注釈付きにして返す。`(注釈付き画像, 検出結果)`。

    `scripts/predict.py` から使う。学習に使った画像と**同じ描き方**にする
    必要があるので、既定は `--no-fronts` で回した `cv_annot_boxes` に合わせて
    `fronts=False` にしてある。**ここが学習時と食い違うと、モデルは見たことの
    ない絵を渡されることになり、成績が静かに落ちる。**
    """
    # letter_size は解像度の違う天気図を救うための倍率。テンプレートは特定の
    # 天気図から切り出したものなので、解像度が違うと同じ H でも画素数が違い、
    # まったく当たらない(scripts/diagnose_detection.py で測れる)。
    # scale は画像とテンプレートの両方にかかる速度の調整なので、大きさの
    # 食い違いは直せない。**両方を掛けたものがテンプレートの最終的な倍率**。
    # 自動のときは、推定した倍率がぴったりとは限らないので、まわりを少しだけ
    # 振って当てる(src/chartscale.SPREAD)。手で倍率を指定したときは振らない
    sizes = (1.0,)
    if letter_size == "auto":
        letter_size, _note = auto_letter_size(rgb.shape[1], templates_dir)
        sizes = sizes_around(letter_size)
        if letter_size != 1.0:
            # テンプレートは推定倍率まで縮めてあるので、振る幅は1.0のまわり
            sizes = tuple(s / letter_size for s in sizes)
    letters = load_templates_scaled(Path(templates_dir), scale * letter_size,
                                    quiet=True)
    if not letters:
        raise SystemExit(
            f"{templates_dir} に H/L のテンプレートがありません。\n"
            "検出には人が切り出したテンプレートが要ります"
            "(docs/2026-08-26-detection-prescreen.md)。"
        )
    marks = (load_templates_scaled(Path(marks_dir), mark_scale, quiet=True)
             if marks_dir and Path(marks_dir).exists() else {})
    detections, report = analyse_chart(
        rgb, letters, marks, scale, threshold, angle_range, angle_step,
        mark_scale, mark_radius, letter_threshold,
        overlay_dir=None, want_masks=fronts, letter_sizes=sizes,
    )
    marked = draw_annotations(rgb, report, detections,
                              boxes=boxes, fronts=fronts, thickness=thickness)
    return marked, detections


_WORKER = {}


def _init_worker(template_dir, mark_dir, scale, letter_size, mark_scale,
                 mark_radius, letter_threshold, options):
    # **文字のテンプレートだけ letter_size を掛ける。**画像側の倍率 (scale) は
    # そのまま analyse_chart に渡す。ここで両方を掛けてしまうと、画像も一緒に
    # 縮んで相対的な大きさが元に戻り、直したつもりが何も変わらない。
    _WORKER["letters"] = load_templates_scaled(Path(template_dir),
                                               scale * letter_size, quiet=True)
    _WORKER["marks"] = (load_templates_scaled(Path(mark_dir), mark_scale, quiet=True)
                        if mark_dir else {})
    _WORKER.update(scale=scale, mark_scale=mark_scale, mark_radius=mark_radius,
                   letter_threshold=letter_threshold, **options)


def _run_one(job) -> tuple:
    path, out_path, threshold, angle_range, angle_step = job
    detections, report = analyse_chart(
        Path(path), _WORKER["letters"], _WORKER["marks"], _WORKER["scale"],
        threshold, angle_range, angle_step, _WORKER["mark_scale"],
        _WORKER["mark_radius"], _WORKER["letter_threshold"],
        overlay_dir=None, want_masks=True,
    )
    rgb = np.array(Image.open(path).convert("RGB"))
    marked = draw_annotations(rgb, report, detections,
                              boxes=_WORKER["boxes"], fronts=_WORKER["fronts"],
                              thickness=_WORKER["thickness"])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(marked).save(out_path)
    return (Path(path).name, len(detections.highs), len(detections.lows),
            len(detections.edge_highs) + len(detections.edge_lows))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--in-dir", required=True)
    parser.add_argument("--out-dir", required=True,
                        help="注釈付き画像の書き出し先。ファイル名は元のまま")
    parser.add_argument("--templates", default="data/templates")
    parser.add_argument("--marks", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--years", type=int, nargs="+", default=None,
                        help="この年の天気図だけを処理する。**学習に使うのは"
                             "ラベルのある年だけ**なので、全期間を描き込むのは"
                             "時間の無駄になる(17,898枚と2,168枚では数時間違う)")
    parser.add_argument("--scale", type=float, default=0.7)
    parser.add_argument("--letter-size", type=letter_size_arg, default=1.0,
                        help="H/Lのテンプレートだけを縮める倍率。解像度の違う"
                             "天気図で検出できないときに使う"
                             "(scripts/diagnose_detection.py が値を教える)")
    parser.add_argument("--mark-scale", type=float, default=1.0)
    parser.add_argument("--mark-radius", type=float, default=MARK_LETTER_RADIUS)
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--letter-threshold", type=float, default=LETTER_THRESHOLD)
    parser.add_argument("--angle-range", type=float, default=60.0)
    parser.add_argument("--angle-step", type=float, default=5.0)
    parser.add_argument("--thickness", type=int, default=3,
                        help="線の太さ。224x224に縮めても残る太さが要る")
    parser.add_argument("--no-boxes", action="store_true", help="高低気圧の枠を描かない")
    parser.add_argument("--no-fronts", action="store_true", help="前線を塗らない")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args()

    paths = sorted(p for p in Path(args.in_dir).iterdir()
                   if p.suffix.lower() in IMAGE_SUFFIXES)
    if args.years:
        import pandas as pd

        from src.split import parse_datetime

        wanted = set(args.years)
        before = len(paths)

        def in_wanted(path) -> bool:
            stamp = parse_datetime(path.name)
            return not pd.isna(stamp) and stamp.year in wanted

        paths = [p for p in paths if in_wanted(p)]
        print(f"{sorted(wanted)} に絞り込み: {before}枚 -> {len(paths)}枚")
        if not paths:
            raise SystemExit(
                f"その年の天気図がありません。ファイル名に日付(YYYYMMDDHH)が"
                f"入っているか確かめてください: {args.in_dir}")
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        raise SystemExit(f"画像が見つかりません: {args.in_dir}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # 途中で止めても続きから。**サイズ0のファイルは書きかけとみなして作り直す**
    todo = [p for p in paths
            if not (out_dir / p.name).exists() or (out_dir / p.name).stat().st_size == 0]
    if len(todo) < len(paths):
        print(f"{len(paths) - len(todo)}件は書き出し済み。残り{len(todo)}件から再開する。")
    if not todo:
        print("すべて済んでいます。作り直すには --out-dir の中身を消してください。")
        return

    if args.letter_size == "auto":
        # **並列の子には数値で配る。**フォルダ内の天気図は同じ大きさなので、
        # 1枚目で決めれば足りる。子で毎回決めると、同じ知らせが人数ぶん出る
        first = Image.open(todo[0]).convert("RGB")
        args.letter_size, note = auto_letter_size(first.size[0], args.templates)
        print(f"大きさの自動調整: {note}")
    letters = load_templates_scaled(Path(args.templates),
                                    args.scale * args.letter_size)
    warn_about_misplaced_marks(letters)
    size_note = f"、文字の倍率 {args.letter_size:g}" if args.letter_size != 1.0 else ""
    print(f"文字 {len(letters)}枚{size_note}、{len(todo)}枚を{args.workers}並列で処理する。")

    jobs = [(str(p), str(out_dir / p.name), args.threshold,
             args.angle_range, args.angle_step) for p in todo]
    options = {"boxes": not args.no_boxes, "fronts": not args.no_fronts,
               "thickness": args.thickness}

    started, done, highs, lows, edges = time(), 0, 0, 0, 0
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=_init_worker,
        initargs=(args.templates, args.marks, args.scale, args.letter_size,
                  args.mark_scale, args.mark_radius, args.letter_threshold,
                  options),
    ) as pool:
        for _name, n_high, n_low, n_edge in pool.map(_run_one, jobs, chunksize=4):
            done += 1
            highs += n_high
            lows += n_low
            edges += n_edge
            if done % 20 == 0 or done == len(todo):
                rate = (time() - started) / done
                print(f"  {done}/{len(todo)}  {rate:.1f}秒/枚  "
                      f"残り{rate * (len(todo) - done) / 60:.0f}分", flush=True)

    print(f"\n書き出しました: {out_dir.resolve()}")
    per = max(1, done)
    print(f"1枚あたり 高気圧の枠 {highs / per:.2f} / 低気圧の枠 {lows / per:.2f} / "
          f"縁の枠(細線) {edges / per:.2f}")
    if not args.marks and (highs + lows) / per < 5.0:
        print("★--marks を指定していません。等圧線に横切られた H/L は、"
              "中心の×との裏書きが無いと拾えません。")
        print("  data\\marks を渡すと拾える数が増えます(実測で 高2.65 / 低3.40)。")
    print("\n**まず数枚を目で見ること。**枠が本物の高低気圧に付いているか、")
    print("前線の塗りが実際の前線と合っているかを確かめてから学習に回す。")
    print("\n学習側の変更は要らない。--data-dir をこの出力に向けるだけ:")
    print(f"  python -m scripts.cross_validate --data-dir {args.out_dir} "
          "--labels data\\labels_v2.csv --years 2023 2024 2025 --out-dir runs\\cv_annot")


if __name__ == "__main__":
    main()
