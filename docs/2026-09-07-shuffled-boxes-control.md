# 枠が効いたのは「位置」か「印」か ― 切り分ける対照実験

2026-09-07

## 何が分かっていないのか

検出した高低気圧を枠として描き込むと macro F1 が 0.640 -> 0.669 に上がった。
これは3foldとも同じ向きで、確かな効果である。

**しかし「なぜ上がったか」には2通りの説明がつき、まだ切り分けていない。**

  (a) 枠が**正しい位置**にあることが効いた
  (b) 単に**目立つ印**が付いたことが効いた(位置は関係ない)

これまでの実験(CoordConv・補助学習)が示したのは「座標を数値で渡しても
効かない」までで、(a)と(b)の区別はしていない。`docs/2026-08-27-features-vs-cnn.md`
の「詰まっていたのは『そこに何かがあると気づくこと』だった」という説明も、
**4つの結果を後から説明する筋書きであって、直接確かめたものではない。**

利用者から「位置というより、検出して印を付けたから分かりやすくなっただけでは
ないか」という指摘があった。**その可能性は否定できていない。**

## 実験

**個数も見た目もそのままに、位置だけを嘘にした画像で学習する。**
各天気図に、**別の天気図から取ってきた枠**を描く。

読み方:

| 結果 | 意味 |
|---|---|
| 0.669 のまま | (b)。位置は効いておらず、印が付いただけ |
| 0.640 まで落ちる | (a)。**位置が効いていた** |
| 0.640 より下 | 嘘の位置が積極的に邪魔をしている |

### なぜ「別の天気図の枠」なのか

でたらめな座標を振ると、図の隅など**ありえない場所**に枠が出る。それでは
「位置が嘘」ではなく「絵が不自然」を測ってしまう。他の天気図の枠なら、
枠の個数も置かれる場所ももっともらしいまま、**その天気図との対応だけ**が壊れる。

割り当ては並べ替え(derangement)なので、**データ全体で見た「1枚あたりの
枠の数」の分布は元と完全に同じ**になる。自分の枠が残る天気図は1枚も無い。
どちらもテストで固定した(`tests/test_shuffle_annotations.py`)。

## 走らせ方

検出はこの一連の処理で一番重いので、**座標を書き出して使い回す。**

    :: 1. 検出して座標を残す(30分ほど。--out-dir は既存のものでよい)
    python -m scripts.annotate_charts --in-dir data\processed\all ^
        --out-dir data\processed\all_annot --years 2023 2024 2025 ^
        --no-fronts --marks data\marks ^
        --dump-detections data\detections.json --workers 4

    :: 2. 位置だけ嘘の枠を描く(数分)
    python -m scripts.shuffle_annotations --in-dir data\processed\all ^
        --detections data\detections.json ^
        --out-dir data\processed\all_shuffled --years 2023 2024 2025

    :: 3. 数枚を目で見る。**枠が高低気圧から外れていれば正しく作れている**

    :: 4. 学習(1foldあたり約30分、3foldで1.5時間)
    python -m scripts.cross_validate --data-dir data\processed\all_shuffled ^
        --labels data\labels_v2.csv --years 2023 2024 2025 ^
        --out-dir runs\new_shuffled

    :: 5. 3つ並べる
    python -m scripts.compare_runs runs\new_baseline runs\new_annot runs\new_shuffled
    python -m scripts.report_metrics --run runs\new_shuffled ^
        --compare runs\new_annot --name "嘘の枠" --compare-name "正しい枠"

**手順1の --out-dir は既にある `all_annot` を指してよい。**書き出し済みの
ファイルは飛ばすので、実質「検出して座標を残すだけ」になる。

## 揃えること

* `--marks data\marks` を付ける。本番の `all_annot` がそうなっているなら揃える
  (付けないと拾える数が減り、枠の個数が変わってしまう)
* `--thickness` は既定の3のまま。枠の見た目を変えると、位置以外も変わる
* `--no-fronts`。本番が枠のみで学習しているため

**ここがずれると、位置以外の違いも一緒に測ることになる。**

## 結果

(実行後に記入する)

| 実行 | macro F1 | 1位正解率 |
|---|---|---|
| `new_baseline`(枠なし) | 0.640 ± 0.013 | 81.6% |
| `new_annot`(正しい枠) | 0.669 ± 0.004 | 83.4% |
| `new_shuffled`(嘘の枠) | | |
