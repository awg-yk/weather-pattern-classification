# 天気図パターン分類AI (weather-pattern-classification)

機械学習（CNN）を用いて天気図を気圧配置パターンごとに自動分類するプロジェクトです。
最終的にはWebサイト上でアップロードした天気図画像（またはERA5由来の気圧場）を分類できるようにします。

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
| `futatsudama_low` | 二つ玉低気圧 |
| `baiu_front` | 梅雨前線 |
| `typhoon` | 台風 |
| `migratory_high` | 移動性高気圧 |
| `zonal_high` | 帯状高気圧（春・秋の高気圧） |
| `summer_pressure_pattern` | 南高北低（夏型の気圧配置） |
| `cold_front_passage` | 寒冷前線通過 |
| `stationary_front` | 停滞前線 |

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

### 気象庁の天気図を使う場合
気象庁の実況天気図・過去の天気図は利用規約を確認の上、`scripts/collect_jma.py` で
日付を指定してダウンロードします。パターンラベルは付いていないため、
`data/labels.csv` に手作業でラベルを追記していく運用になります
（過去の顕著な事例をまとめた気象庁・気象庁ライブラリの資料を参考にすると効率的です）。

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

## 進捗

- [x] プロジェクト雛形作成
- [ ] データ収集スクリプトの実装・実行
- [ ] ラベリング（初期セット数百枚）
- [ ] モデル学習・評価
- [ ] 推論APIとWeb UI
