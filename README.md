# 天気図パターン分類AI (weather-pattern-classification)

機械学習（CNN）を用いて天気図を気圧配置パターンごとに自動分類するプロジェクトです。
最終的にはWebサイト上でアップロードした天気図画像（またはERA5由来の気圧場）を分類できるようにします。

## すぐに使う

学習済みモデル(`weights/model.pt`)をリポジトリに同梱しているので、学習不要ですぐに使える。

<a href="https://colab.research.google.com/github/awg-yk/weather-pattern-classification/blob/main/notebooks/predict.ipynb" target="_blank" rel="noopener noreferrer"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

上のバッジからColabノートブックを開けば、①画像をアップロードする、②日付を指定して
気象庁アーカイブから直接取得する、のどちらかの方法で分類結果と判断根拠(Grad-CAMの
ヒートマップ)が表示される。

手元の環境で使いたい場合はコマンドラインでも同じことができる。

```bash
git clone https://github.com/awg-yk/weather-pattern-classification.git
cd weather-pattern-classification
pip install -r requirements.txt

# 画像を渡す場合
python scripts/predict.py path/to/天気図画像.png

# 日付を指定して気象庁アーカイブから直接取得する場合(画像不要)
python scripts/predict.py --date 2025-01-01 --hour 0
```

気象庁の生のPDF変換画像(枠・座標グリッド・日時スタンプ付き)であれば自動で前処理される。
既に前処理済みの画像を渡す場合は `--no-preprocess` を付ける。
`--date`で指定できるのは気象庁JSMAPアーカイブの範囲(2022年10月1日以降)のみ。

**2000年〜2022年9月分について**: 国立国会図書館デジタルコレクション(NDL)にも
同期間の天気図があり、`scripts/collect_ndl.py`でpidの検索・日別ファイル一覧の取得までは
動作するが、実際のPDF本体のダウンロードが401 Unauthorizedになる問題が未解決のため、
現状は無効化している(`scripts/fetch_and_predict.py`の`ENABLE_NDL = False`)。
ブラウザで直リンクURLを直接開いても401になることを確認済みで、単純なCookie/Referer
の問題ではなく、NDL側の何らかの認証フローが必要とみられる。

そのため、この期間分はNDLのサイト上で手動でダウンロードし、PDF→PNG→JPEG変換は
別リポジトリで行う運用にしている。変換済みのJPEGは
[weather-pattern-classification-data](https://github.com/awg-yk/weather-pattern-classification-data)
にあり、`scripts/fetch_manual_chart.py`が日付指定で個別に取得する。

## データ収集〜学習を一から試す場合

手順は`notebooks/weather_pattern_classification.ipynb`にまとめてある
(データ収集・前処理・ラベリング・学習・評価・Grad-CAM・Web推論デモ)。

## 全体方針

1. **データ収集**: 気象庁の地上天気図画像、またはERA5再解析データ（海面気圧など）を収集
2. **ラベリング**: 気圧配置パターン（西高東低・南岸低気圧・台風など）でラベル付け
   - JMA画像 → 人手 or 半自動（規則ベース補助）でラベリング
   - ERA5 → 気圧場の統計量から規則ベースで自動ラベリング（後で人手検証）
3. **モデル学習**: 転移学習ベースのCNN（EfficientNet / ResNet）で画像分類
4. **推論API**: FastAPIでモデルをサーブ
5. **Webフロントエンド**: 画像をアップロードすると分類結果を表示するシンプルなUI

## 分類ラベル（初期案・気象庁的な代表パターン）

| ラベル | 説明 |
|---|---|
| `winter_pressure_pattern` | 西高東低（冬型の気圧配置） |
| `nankigan_low` | 南岸低気圧 |
| `japan_sea_low` | 日本海低気圧 |
| `futatsudama_low` | 二つ玉低気圧 |
| `typhoon` | 台風 |
| `migratory_high` | 移動性高気圧（帯状高気圧`zonal_high`を統合） |
| `pacific_high` | 太平洋高気圧型（旧`summer_pressure_pattern`。移動しない・北に低気圧がないケースでも判断しやすいよう気圧系そのもので命名） |
| `front_passage` | 前線通過（寒冷前線・温暖前線） |
| `stationary_front` | 停滞前線 |
| `okhotsk_high` | オホーツク海高気圧 |

季節を区別しないため梅雨前線は対象外とし、代わりに日本海側から接近する
低気圧のパターンを独立したラベルとして追加している。1枚の天気図に複数の
パターンが同時に当てはまることがあるため、マルチラベル分類として扱う
（詳細は「ラベリング」の節を参照）。

`src/labels.py` で管理し、精度・データ量を見ながら統廃合します。

**ラベルを統合したいとき**: `src/labels.py`から統合元のラベルを削除した上で、
既存の`labels.csv`を`scripts/merge_labels.py`で書き換える。

```bash
python scripts/merge_labels.py --labels data/labels_v2.csv --from zonal_high --to migratory_high
```

自動でバックアップ(`.bak`)が作られる。**ラベル数(クラス数)が変わるとモデルの
出力層のサイズも変わるため、`labels.py`の変更と再学習は必ずセットで行うこと。**
`labels.py`だけ先に変更すると、まだ再学習していない既存の`weights/model.pt`
(元のクラス数のまま)と食い違い、`predict.py`やWeb UIがサイズ不一致エラーで
動かなくなる。

## ディレクトリ構成

```
data/
  raw/            # ダウンロードした元画像・元データ（gitignore対象）
  processed/      # 学習用に前処理した画像（gitignore対象）
  labels_v2.csv   # ファイル名とラベルの対応表。**学習・評価にはこちらを使う**
  labels.csv      # 旧版。okhotsk_highの陽性が85件しかない(v2は199件)
  review_okhotsk*.csv  # okhotsk_highを見直したときの判定記録
scripts/
  preflight.py        # **学習を始める前の設定チェック(数秒)**
  cross_validate.py   # leave-one-year-out交差検証。報告用の数値はこれ
  compare_runs.py     # 実行どうしの比較。ラベルを絞って平均を取り直せる
  ensemble_chart_grid.py  # 天気図モデルと格子モデルの確率を混ぜる
  collect_jma.py      # 気象庁天気図画像の収集スクリプト
  download_era5.py    # ERA5データ取得スクリプト（CDS API）
  label_tool.py       # ラベル付け・見直し(盲検レビュー)
  auto_label_era5.py  # ERA5気圧場からの規則ベース自動ラベリング
src/
  dataset.py      # 天気図画像のDataset
  era5_grid.py    # ERA5格子を直接入力にするDataset
  model.py        # EfficientNet / SmallCNN / CoordConv / FeatureFusion
  train.py        # 学習スクリプト
  evaluate.py     # 評価スクリプト
  split.py        # train/val/testの分け方
  labels.py       # ラベル定義
docs/
  2026-08-21-chart-vs-era5-grid.md  # 天気図とERA5格子の比較。結論と未解決の課題
runs/             # 交差検証の出力。summary.jsonだけ追跡する
webapp/
  backend/        # FastAPI推論API
  frontend/       # シンプルなアップロードUI
```

**天気図画像はこのリポジトリには入っていない。** 気象庁から取得する場合は
`scripts/collect_jma.py`、2000〜2022年分は
[weather-pattern-classification-data](https://github.com/awg-yk/weather-pattern-classification-data)
にある。`--data-dir` には画像を置いた実際のディレクトリを渡すこと。

## セットアップ

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## データ収集について

### 気象庁の天気図を使う場合（メインデータソース）
気象庁「保存用天気図」(日本域天気図 JSMAP) をPDFでダウンロードし、PNGに変換します。

```
https://www.data.jma.go.jp/yoho/data/wxchart/archive/{yyyy}_{mm}/PDFDATA/JSMAP/Js_{yyyymmddHH}.pdf
```

このシリーズは配色が統一されています（海岸線・経緯度線=赤茶色、等圧線=黒、
温暖前線=赤、寒冷前線=青、閉塞前線=ピンク）。前線種別の識別に色情報がそのまま
使えるため、グレースケール化せずカラーのまま学習データとして利用します。

PDF→PNG変換に `poppler-utils` が必要です。

```bash
# Debian/Ubuntu
sudo apt-get install poppler-utils
# Mac
brew install poppler
```

```bash
python scripts/collect_jma.py --start 2026-01-01 --end 2026-04-30 --out data/raw/jma
```

ダウンロード後、`scripts/preprocess_jma.py` で余白の自動クロップと右下の日時スタンプの
マスク処理を行う（スタンプの文字自体をモデルが学習してしまうのを防ぐため）。

```bash
python scripts/preprocess_jma.py --in-dir data/raw/jma/png --out-dir data/processed/jma
```

利用規約を確認の上、許可された範囲・頻度で利用してください。

### ラベリング（マルチラベル対応）

> **どのラベルファイルを使うかを毎回確かめること。** いま学習・評価に使うのは
> `data/labels_v2.csv`。`data/labels.csv` は okhotsk_high の見直し前の版で、
> 陽性が85件しかない(v2は199件)。両方が手元にある状態で古いほうを使ってしまい、
> 丸一日ぶんの実験をやり直したことがある。`python -m scripts.preflight --labels <ファイル>
> --years ...` を流せば、そのファイルのラベルごとの件数が数秒で出る。

パターンラベルは付いていないため、`scripts/label_tool.py` を使ってColab上で
チェックボックスでラベル付けする。1枚の天気図に複数のパターンが同時に
当てはまることがある（例: 西高東低かつ日本海低気圧）ため、複数選択に対応している。
`data/labels_v2.csv` の label列にはパイプ区切りで保存される
(例: `winter_pressure_pattern|japan_sea_low`)。

```python
import sys
sys.path.append("/content/weather-pattern-classification")
from scripts.label_tool import run_labeling_session, run_review_session

# 1) 未ラベルの新しい画像にラベルを付ける
run_labeling_session(
    images_dir="<画像のディレクトリ>",
    labels_csv="data/labels_v2.csv",
)
```

- ラベルCSVに追記していく形式なので、途中で中断しても再開時にラベル済みの
  画像は自動でスキップされる
- チェックボックスで複数選択後「決定」ボタンで次の画像に進む
- 「戻る」ボタンで直前の1件を取り消せる
- 判断に迷う画像は「わからない/該当なし」で `unclassified` として記録し、後でまとめて見直す
- ラベル付け作業もColabのランタイムが切れるとラベルCSVが消えるため、
  こまめにGoogle Driveへコピーするか、`labels_csv`引数を直接Drive上のパスにする

少数派のラベル（例: `futatsudama_low`）を増やしたい場合は、新規収集の代わりに
既にラベル済みの近いパターンの画像を見直して追加タグを付けることもできる。

```python
# 2) 既にラベル済みの画像を見直して、追加のタグを付け足す
run_review_session(
    images_dir="<画像のディレクトリ>",
    labels_csv="data/labels_v2.csv",
    filter_labels=["japan_sea_low", "nankigan_low"],  # この中のどれかが付いている画像だけ対象
)
```

`filter_labels`を省略すると全件が見直し対象になる。チェックボックスには現在のラベルが
反映された状態で表示され、「保存して次へ」で上書き、「変更せず次へ」でスキップできる。

### ERA5を使う場合
[Copernicus Climate Data Store](https://cds.climate.copernicus.eu/) のアカウントを作成し、
APIキーを `~/.cdsapirc` に設定した上で `scripts/download_era5.py` を実行します。
海面更正気圧（mslp）の格子データを取得し、`era5_to_image.py` で天気図風の等圧線画像に変換、
`auto_label_era5.py` で気圧配置の特徴量（気圧傾度の向き、等圧線の混み具合など）から
規則ベースで初期ラベルを付与します。人手で一部検証してから学習に使うことを推奨します。

## 学習

```bash
python -m src.train --data-dir <画像のディレクトリ> --labels data/labels_v2.csv --epochs 30
```

主なオプション:

| オプション | 既定 | 説明 |
|---|---|---|
| `--input-mode` | `chart` | `chart`=天気図の画像。`era5-grid`=ERA5の海面更正気圧・850hPa気温の格子をそのまま入力する |
| `--arch` | `efficientnet_b0` | `small_cnn`は浅い自作ネット。**ERA5格子ではこちらを使う**(EfficientNetは同じパラメータ数でも自明な予測を下回る。`src/model.py`の`SmallCNN`を参照) |
| `--coordconv` | なし | 入力に座標チャンネルを足す。位置で決まる気圧配置(オホーツク海高気圧など)の判別を助ける |
| `--grid-size` | 128 | `--input-mode era5-grid`のときの格子の一辺。ERA5の元データは181×221点なので、128は粗い |
| `--cnn-widths` | 128 256 512 512 | `--arch small_cnn`の各段の幅。実測で最良かつfold間のばらつきも最小だった値 |
| `--image-size` | 224 | 天気図の入力解像度。VRAMを解像度の2乗で消費するので`--batch-size`を併せて下げること |
| `--select-metric` | `macro_ap` | ベストモデルの保存・早期終了の判定指標。**閾値0.5固定のmacro F1で選ぶと、pos_weightで陽性寄りに学習させる都合で初期エポックが最高値を取り、実質未学習の重みが保存される**(実際に起きた)。既定のmacro APは閾値を決めずに順位付けの良さを測るので、閾値を最適化する最終評価と対応する |
| `--val-mode` | `tail` | 検証データの取り方。`tail`=学習期間の末尾、`spread`=1年を通して等間隔に週を抜き取る。`tail`だと検証が秋冬に偏り、夏の型(オホーツク海高気圧・太平洋高気圧)の閾値やモデル選択がその季節を見ないまま決まる |
| `--split-mode` | `temporal` | train/val/testの分け方。`temporal`=日付順のブロック、`by_year`=年ごと、`random`=従来のランダム分割 |
| `--test-ratio` | 0.0 | 最終報告用に取り分けるテストデータの割合。学習にも閾値探索にも使わない |
| `--seed` | 42 | 分割と乱数の固定。件数を変えて比較する際は必ず揃えること |
| `--train-limit` | なし | 学習に使う件数を制限する(検証セットは変えない)。学習件数と性能の関係を調べる実験用 |

### 分割方式と時間的リーク

天気図は**隣り合う日どうしが極めて似ている**ため、日付を無視してランダムに分割すると
「7月3日で学習し、7月4日で評価する」状態になり、実質的に見たことのある画像を当てている
だけのスコアが出る。日次データでランダム分割した場合、検証画像の8割以上が学習画像と
1日差になる。

このため既定は `temporal`(日付順のブロック分割)とし、学習・検証・テストが時間軸上で
重ならないようにしている。手元のデータでリーク量を確認するには:

```bash
python -m scripts.check_leakage --labels data/labels_v2.csv
```

季節の偏りが気になる場合(テスト期間が冬だけになる等)は `--split-mode by_year` を使うと、
各分割が全季節を含むようになる。`random` は過去の実験を再現する用途にのみ使うこと。

### train / val / test の役割

| 分割 | 使いみち |
|---|---|
| train | モデルのパラメータを学習する |
| val | 早期終了・ベストモデルの選択・判定閾値の探索 |
| test | 最終報告の数値を1回だけ測る。上のどれにも使ってはいけない |

`--test-ratio 0.2` のように指定すると `src/evaluate.py --split test` で最終評価できる。
`--optimize-thresholds` と併用した場合、**閾値はvalで決めてtestに適用**されるため、
閾値をテストデータに合わせ込むことによる水増しが起きない。

## 交差検証(報告用の数値はこれで出す)

時系列で後ろを切り出す分割には弱点がある。テスト期間が特定の季節に偏るため、
台風のように夏にしか現れないラベルがテストセットに1件も入らず**評価できない**。

`leave-one-year-out`(1年ずつテストに回す)ならテストが必ず通年になるので、
全ラベルが評価対象に入る。さらにfoldごとに推定値が得られるので、
**平均±標準偏差**で報告でき、差が誤差かどうかを判断できる。

**始める前に必ず `preflight` を流すこと。** 1foldに数十分かかるので、設定の不備に
学習を始めてから気づくと損失が大きい。数秒で終わる。

```bash
python -m scripts.preflight \
    --data-dir <画像のディレクトリ> --labels data/labels_v2.csv \
    --years 2023 2024 2025
```

どのラベルファイルに何件入っているか、画像やERA5が全行そろっているか、foldごとの
分割、そして**検証データに1件も現れないラベル**が出る。最後は特に効く——そのラベルに
ついては閾値もモデル選択も決めようがなく、決めればでたらめになる。

```bash
python -m scripts.cross_validate \
    --data-dir <画像のディレクトリ> --labels data/labels_v2.csv \
    --years 2023 2024 2025 --out-dir runs/v2_chart

# ERA5の格子を入力にする場合
python -m scripts.cross_validate --input-mode era5-grid --arch small_cnn --coordconv \
    --labels data/labels_v2.csv --years 2023 2024 2025 --out-dir runs/v2_grid
```

| fold | 学習 | テスト |
|---|---|---|
| 1 | 2024, 2025 | 2023(通年) |
| 2 | 2023, 2025 | 2024(通年) |
| 3 | 2023, 2024 | 2025(通年) |

学習に使う年のうち、日付順で後ろ20%が検証用(早期終了・閾値決定)に回る。
テスト年から`--gap-days`(既定3日)以内の学習データは除外される。年をまたぐ
境界では12月末と1月初が数日差になり、天気図がほとんど同じになるため。

途中で止まった場合は `--skip-existing` を付けて再実行すると、結果JSONが
残っているfoldは飛ばして続きから進む。

**評価セットに1件も出現しなかったラベルについて**: sklearnはF1を0として扱うが、
これは「性能が0」ではなく「測れていない」という意味なので、macro平均に含めると
実力を過小評価する。`evaluate.py` は出現したラベルだけに絞ったmacro F1も併記する。
報告にはそちらを使い、評価できなかったラベルは明示すること。

### macro F1 の絶対値を、そのまま読まないこと

出現率 p のラベルは、**全部を陽性と答えるだけで** F1 = 2p/(1+p) を取る。移動性高気圧
(出現率41%)なら0.59。この10ラベルの平均で0.26〜0.28になるため、macro F1の絶対値は
学習の成果を表さない。

`evaluate.py` は毎回この基準と、そこからの上積みを出力する。下回っていれば警告する。

```
全部を陽性と答えるだけで得られる macro F1: 0.281  → このモデルの上積み: +0.342
```

**この基準を出していなかったために、自明な予測を下回る結果(0.250 対 0.258)を
「低いが学習はできている」と読み違えたことがある。** 報告には上積みを併記すること。

### 実行どうしを比べる

```bash
python -m scripts.compare_runs runs/v2_chart runs/v2_grid \
    --exclude front_passage stationary_front
```

`--exclude` で指定したラベルを外して平均を取り直す(基準も同じラベルで取り直される)。
ERA5格子には前線の記号が含まれないので、その不利を除いて比べたいときに使う。

foldは実行どうしで共通なので、平均どうしではなく**fold単位で対応させた差**も出る。
年ごとの差が揃って同符号なら、平均の差が標準偏差より小さくても実質的な改善と読める。

### 天気図とERA5格子を組み合わせる

```bash
python -m scripts.ensemble_chart_grid --data-dir <画像> --labels data/labels_v2.csv \
    --chart-weights runs/v2_chart --grid-weights runs/v2_grid \
    --years 2023 2024 2025 --out runs/v2_ensemble.json
```

学習済みの重みを使うので再学習は不要。両者の確率を `p = (1-w)*天気図 + w*格子` で
混ぜる。**wはラベルごとに決める**——天気図が強い台風と格子が強いオホーツク海高気圧を
ひとつのwで妥協させると、両方が損をする。wと閾値はどちらも検証データで決め、
テストには一度も触れずに適用する。

学習した重みには入力解像度とラベル一覧が同梱される。`predict.py`・`evaluate.py`・Grad-CAM・
Web UIはこれを読んで前処理を自動的に合わせるため、`--image-size`を変えても推論側の指定は不要。
(EfficientNetは適応的プーリングを使うため、サイズが食い違ってもエラーにならず「黙って精度が
落ちる」。それを防ぐための仕組み。)

**注意**: `--out`を省略すると毎回`weights/model.pt`に上書き保存される。過去のチェックポイントを
残したい場合は、再学習のたびに `--out weights/model_YYYYMMDD.pt` のように日付やバージョンを
含めたファイル名を指定すること。上書きしてしまうと、その時点の重みは復元できない。

## いまの結果

leave-one-year-out 交差検証(2023/2024/2025)、`data/labels_v2.csv`、macro F1。
詳細と、そこに至るまでに直した測定上の欠陥は
[`docs/2026-08-21-chart-vs-era5-grid.md`](docs/2026-08-21-chart-vs-era5-grid.md)。

| 入力 | macro F1 | 自明な予測との差 | fold間の標準偏差 |
|---|---|---|---|
| 天気図画像 | 0.626 | +0.359 | 0.006 |
| ERA5格子 | 0.512 | +0.245 | 0.023 |
| 両方(確率を混ぜる) | **0.640** | **+0.374** | 0.022 |

**ERA5の気圧場は天気図画像の代わりにならない。** 出発点の「格子は天気図の元データ
なのだから上位互換のはず」という仮説は否定された。前線2ラベルを除いても差は0.089残り、
全foldで同じ向き。前線記号だけでは説明できない。

**ただし、位置で定義される型では逆転する。** オホーツク海高気圧は格子0.470 対
天気図0.319で、3つのfoldすべてで格子が上。混合時にこのラベルへ選ばれる重みは0.93
(0=天気図のみ / 1=格子のみ)で、台風の0.02と対照的。**得意分野が違うので、
両方使うのが最良。**

### 論文の図として出す(ラベルを指定して成功例・失敗例を並べる)

`scripts/gradcam_report.py` は、ラベルを1つ指定して、そのラベルが正解となっている
テスト画像から**当てられた例と見逃した例**を並べた図を作る。得意なラベルと苦手な
ラベルを対比させると、何を手がかりにしているのかが読み取りやすい。

```bash
# 得意な例(台風。天気図0.766 対 ERA5格子0.478)
python -m scripts.gradcam_report \
    --data-dir <画像のディレクトリ> --labels data/labels_v2.csv \
    --weights runs/v2_chart/model_test2025.pt \
    --years 2023 2024 2025 --split-mode loyo --test-year 2025 \
    --label typhoon --out runs/v2_chart/gradcam_typhoon.png

# 苦手な例(オホーツク海高気圧。天気図0.319 対 ERA5格子0.470と、唯一格子が上回る)
python -m scripts.gradcam_report ... --label okhotsk_high --out runs/v2_chart/gradcam_okhotsk.png
```

分割の指定(`--years` / `--split-mode` / `--test-year` など)は学習時と同じ値にすること。
学習に使った画像が図に混ざらないよう、テストセットからのみ選ばれる。

## モデルの判断根拠を可視化する(Grad-CAM)

CNNは「H/Lの文字」「前線の色」「等圧線の形」を人間のように記号として理解しているわけではなく、
ラベルと統計的に相関する画素パターンを学習しているだけである。実際に画像のどこに注目して
予測しているかを確認するため、Grad-CAMで可視化できる。

```python
import sys
sys.path.append("/content/weather-pattern-classification")
from scripts.gradcam import show_gradcam

show_gradcam(
    image_path="/path/to/some_chart.png",
    weights_path="/content/drive/MyDrive/weather-pattern-classification-data/weights/model.pt",
    top_k=3,                 # 確信度が高い上位k個のラベルを可視化
    apply_preprocess=True,   # 生のJMA画像(枠・スタンプ付き)ならTrue、前処理済みならFalse
)
```

確信度が高い上位ラベルごとに、モデルが注目した領域がヒートマップで重ねて表示される。
H/Lの記号や前線・等圧線のあたりに反応が集中していれば信頼できる判断をしている可能性が高く、
逆に無関係な枠線や余白に反応している場合は、前処理やデータの見直しが必要というサインになる。

## 進捗

- [x] プロジェクト雛形作成
- [x] データ収集スクリプトの実装・実行（2023〜2026年、2432枚）
- [x] ラベリング（2401枚。okhotsk_highは盲検レビューで見直し済み）
- [x] モデル学習・評価（leave-one-year-out交差検証。`docs/2026-08-21-chart-vs-era5-grid.md`）
- [x] 天気図画像とERA5格子の比較
- [ ] futatsudama_lowのラベル見直し（10ラベル中もっとも不安定。0.433 ± 0.135）
- [ ] ERA5のチャンネル追加（いまは海面更正気圧と850hPa気温だけ。500hPa高度・風を足せば前線の判別が変わりうる）
- [ ] 2000〜2022年のラベリング（天気図15,522枚は取得済み、ERA5も取得可能。必要なのはラベルだけ）
- [ ] 推論APIとWeb UI
