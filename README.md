# 天気図パターン分類AI (weather-pattern-classification)

機械学習（CNN）を用いて天気図を気圧配置パターンごとに自動分類するプロジェクトです。
最終的にはWebサイト上でアップロードした天気図画像（またはERA5由来の気圧場）を分類できるようにします。

## すぐに使う

学習済みモデル(`weights/model.pt`)をリポジトリに同梱しているので、学習不要ですぐに使える。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/awg-yk/weather-pattern-classification/blob/claude/weather-chart-classification-4b6in1/notebooks/predict.ipynb)

上のバッジからColabノートブックを開けば、画像をアップロードするだけで分類結果と
判断根拠(Grad-CAMのヒートマップ)が表示される。

手元の環境で使いたい場合はコマンドラインでも同じことができる。

```bash
git clone -b claude/weather-chart-classification-4b6in1 https://github.com/awg-yk/weather-pattern-classification.git
cd weather-pattern-classification
pip install -r requirements.txt

python scripts/predict.py path/to/天気図画像.png
```

気象庁の生のPDF変換画像(枠・座標グリッド・日時スタンプ付き)であれば自動で前処理される。
既に前処理済みの画像を渡す場合は `--no-preprocess` を付ける。

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
| `migratory_high` | 移動性高気圧 |
| `zonal_high` | 帯状高気圧（春・秋の高気圧） |
| `summer_pressure_pattern` | 南高北低（夏型の気圧配置） |
| `cold_front_passage` | 寒冷前線通過 |
| `stationary_front` | 停滞前線 |

季節を区別しないため梅雨前線は対象外とし、代わりに日本海側から接近する
低気圧のパターンを独立したラベルとして追加している。1枚の天気図に複数の
パターンが同時に当てはまることがあるため、マルチラベル分類として扱う
（詳細は「ラベリング」の節を参照）。

`src/labels.py` で管理し、精度・データ量を見ながら統廃合します。

## ディレクトリ構成

```
data/
  raw/            # ダウンロードした元画像・元データ（gitignore対象）
  processed/      # 学習用に前処理した画像
  labels.csv      # ファイル名とラベルの対応表
scripts/
  collect_jma.py      # 気象庁天気図画像の収集スクリプト
  download_era5.py    # ERA5データ取得スクリプト（CDS API）
  era5_to_image.py    # ERA5気圧場を天気図風画像に変換
  auto_label_era5.py  # ERA5気圧場からの規則ベース自動ラベリング
src/
  dataset.py      # PyTorch Dataset/DataLoader定義
  model.py         # CNNモデル定義（転移学習）
  train.py         # 学習スクリプト
  evaluate.py       # 評価スクリプト
  labels.py         # ラベル定義
webapp/
  backend/          # FastAPI推論API
  frontend/          # シンプルなアップロードUI
```

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

パターンラベルは付いていないため、`scripts/label_tool.py` を使ってColab上で
チェックボックスでラベル付けする。1枚の天気図に複数のパターンが同時に
当てはまることがある（例: 西高東低かつ日本海低気圧）ため、複数選択に対応している。
`data/labels.csv` の label列にはパイプ区切りで保存される
(例: `winter_pressure_pattern|japan_sea_low`)。

```python
import sys
sys.path.append("/content/weather-pattern-classification")
from scripts.label_tool import run_labeling_session, run_review_session

# 1) 未ラベルの新しい画像にラベルを付ける
run_labeling_session(
    images_dir="data/processed/jma",
    labels_csv="data/labels.csv",
)
```

- `data/labels.csv` に追記していく形式なので、途中で中断しても再開時にラベル済みの
  画像は自動でスキップされる
- チェックボックスで複数選択後「決定」ボタンで次の画像に進む
- 「戻る」ボタンで直前の1件を取り消せる
- 判断に迷う画像は「わからない/該当なし」で `unclassified` として記録し、後でまとめて見直す
- ラベル付け作業もColabのランタイムが切れると`data/labels.csv`が消えるため、
  こまめにGoogle Driveへコピーするか、`labels_csv`引数を直接Drive上のパスにする

少数派のラベル（例: `futatsudama_low`）を増やしたい場合は、新規収集の代わりに
既にラベル済みの近いパターンの画像を見直して追加タグを付けることもできる。

```python
# 2) 既にラベル済みの画像を見直して、追加のタグを付け足す
run_review_session(
    images_dir="data/processed/jma",
    labels_csv="data/labels.csv",
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
python src/train.py --data-dir data/processed --labels data/labels.csv --epochs 30
```

**注意**: `--out`を省略すると毎回`weights/model.pt`に上書き保存される。過去のチェックポイントを
残したい場合は、再学習のたびに `--out weights/model_YYYYMMDD.pt` のように日付やバージョンを
含めたファイル名を指定すること。上書きしてしまうと、その時点の重みは復元できない。

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
- [ ] データ収集スクリプトの実装・実行
- [ ] ラベリング（初期セット数百枚）
- [ ] モデル学習・評価
- [ ] 推論APIとWeb UI
