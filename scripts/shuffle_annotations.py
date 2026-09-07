r"""**対照実験用**: 枠を、その天気図とは無関係な位置に描く。

何を確かめるのか
----------------
検出した高低気圧を枠として描き込むと macro F1 が 0.640 -> 0.669 に上がった。
しかし**なぜ上がったのかは、2通りの説明がつく。**

  (a) 枠が**正しい位置**にあることが効いた
  (b) 単に**目立つ印**が付いたことが効いた(位置は関係ない)

この2つは、これまでの実験では切り分けられていなかった。

そこで、**個数も見た目もそのままに、位置だけを嘘にした画像**を作る。
各天気図に、別の天気図から取ってきた枠を描く。

  * 結果が 0.669 のまま     -> (b)。位置は効いておらず、印が付いただけ
  * 結果が 0.640 まで落ちる -> (a)。位置が効いていた
  * 0.640 より下がる        -> 嘘の位置が積極的に邪魔をしている

**別の天気図の枠を使い回すのが要点。**でたらめな座標を振ると、海の真ん中や
図の隅など**ありえない場所**に枠が出て、「位置が嘘」ではなく「絵が不自然」を
測ってしまう。他の天気図の枠なら、枠の個数も置かれる場所ももっともらしいまま、
**その天気図との対応だけ**が壊れる。

検出はやり直さない
------------------
検出はこの一連の処理で一番重い。先に

    python -m scripts.annotate_charts --in-dir data\processed\all ^
        --out-dir data\processed\all_annot --years 2023 2024 2025 --no-fronts ^
        --marks data\marks --dump-detections data\detections.json --workers 4

で座標を書き出しておけば、ここでは描くだけなので数分で終わる。

使い方:
    python -m scripts.shuffle_annotations ^
        --in-dir data\processed\all --detections data\detections.json ^
        --out-dir data\processed\all_shuffled --years 2023 2024 2025

そのあと、学習は --data-dir を差し替えるだけ:
    python -m scripts.cross_validate --data-dir data\processed\all_shuffled ^
        --labels data\labels_v2.csv --years 2023 2024 2025 ^
        --out-dir runs\new_shuffled
"""

import argparse
import json
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from PIL import Image

from scripts.annotate_charts import draw_annotations
from src.chartfeatures import ChartDetections


def _as_detections(found: dict) -> ChartDetections:
    """JSONの辞書を、描画に渡せる形に戻す。"""
    return ChartDetections(
        highs=[tuple(p) for p in found.get("highs", [])],
        lows=[tuple(p) for p in found.get("lows", [])],
        edge_highs=[tuple(p) for p in found.get("edge_highs", [])],
        edge_lows=[tuple(p) for p in found.get("edge_lows", [])],
    )


def derange(names: list, seed: int) -> dict:
    """各天気図に、**自分以外の**天気図を1つ割り当てる。

    自分に当たったままだと、その1枚だけ「正しい位置の枠」になってしまう。
    実験の対照として成り立たないので、必ずずらす。
    """
    shuffled = list(names)
    rng = random.Random(seed)
    for _ in range(100):
        rng.shuffle(shuffled)
        if all(a != b for a, b in zip(names, shuffled)):
            return dict(zip(names, shuffled))
    # 100回引き直しても当たりが残るのは、名前が1つしかない場合くらい
    raise SystemExit(
        f"自分以外を割り当てられませんでした(天気図が{len(names)}枚しかない?)")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--in-dir", required=True, help="前処理後の天気図")
    parser.add_argument("--detections", required=True,
                        help="annotate_charts --dump-detections が書いたJSON")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--years", type=int, nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=0, help="割り当ての乱数の種")
    parser.add_argument("--thickness", type=int, default=3,
                        help="枠の太さ。**本番と揃えること**")
    args = parser.parse_args()

    found_all = json.loads(Path(args.detections).read_text(encoding="utf-8"))
    print(f"検出の記録: {len(found_all)}件")

    paths = sorted(p for p in Path(args.in_dir).iterdir()
                   if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if args.years:
        import pandas as pd

        from src.split import parse_datetime

        wanted = set(args.years)
        before = len(paths)
        paths = [p for p in paths
                 if not pd.isna(parse_datetime(p.name))
                 and parse_datetime(p.name).year in wanted]
        print(f"{sorted(wanted)} に絞り込み: {before}枚 -> {len(paths)}枚")

    usable = [p for p in paths if p.name in found_all]
    if len(usable) < len(paths):
        print(f"★検出の記録に無い天気図が {len(paths) - len(usable)}枚あります。"
              "その分は飛ばします")
        print("  annotate_charts を同じ --in-dir / --years で回したか確かめてください")
    if not usable:
        raise SystemExit("描ける天気図がありません")

    names = [p.name for p in usable]
    assignment = derange(names, args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(usable)}枚に、**別の天気図の枠**を描きます(種 {args.seed})\n")

    moved = 0
    for done, path in enumerate(usable, start=1):
        borrowed = found_all[assignment[path.name]]
        detections = _as_detections(borrowed)
        rgb = np.array(Image.open(path).convert("RGB"))
        marked = draw_annotations(rgb, {}, detections, boxes=True, fronts=False,
                                  thickness=args.thickness)
        Image.fromarray(marked).save(out_dir / path.name)
        moved += len(detections.highs) + len(detections.lows)
        if done % 200 == 0 or done == len(usable):
            print(f"  {done}/{len(usable)}", flush=True)

    print(f"\n書き出しました: {out_dir.resolve()}")
    print(f"1枚あたりの枠(中心が図内): {moved / len(usable):.2f}個")
    print("\n**まず数枚を目で見ること。**枠が高低気圧から外れていれば正しく"
          "作れています(外れているのが狙いです)。")
    print("\n学習:")
    print(f"  python -m scripts.cross_validate --data-dir {args.out_dir} "
          "--labels data\\labels_v2.csv --years 2023 2024 2025 "
          "--out-dir runs\\new_shuffled")


if __name__ == "__main__":
    main()
