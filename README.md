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

表示される確信度については[確信度(表示する%)の校正](#確信度表示するの校正)も参照。
重みの隣に校正ファイルが無い場合、確信度は学習時の`pos_weight`のぶん実際より高く出る。
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
python scripts/merge_labels.py --labels data/labels.csv --from zonal_high --to migratory_high
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
  processed/      # 学習用に前処理した画像
  labels.csv      # ファイル名とラベルの対応表
scripts/
  collect_jma.py      # 気象庁天気図画像の収集スクリプト
  download_era5.py    # ERA5データ取得スクリプト（CDS API）
  era5_to_image.py    # ERA5気圧場を天気図風画像に変換
  auto_label_era5.py  # ERA5気圧場からの規則ベース自動ラベリング
  calibrate.py        # 確信度の校正を作る（<重み名>.calib.json）
src/
  dataset.py      # PyTorch Dataset/DataLoader定義
  model.py         # CNNモデル定義（転移学習）
  train.py         # 学習スクリプト
  evaluate.py       # 評価スクリプト
  calibration.py    # 確信度の校正（表示%を実際の的中率に合わせる）
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
python -m src.train --data-dir data/processed --labels data/labels.csv --epochs 30
```

主なオプション:

| オプション | 既定 | 説明 |
|---|---|---|
| `--image-size` | 224 | 入力解像度。天気図は等圧線・前線記号が細かく224では潰れやすい。384程度まで上げると精度が改善しうるが、VRAMを解像度の2乗で消費するので`--batch-size`を併せて下げること |
| `--select-metric` | `macro_f1` | ベストモデルの保存・早期終了の判定指標。`val_loss`は改善が早く止まりF1のピークを取り逃すことが実測で確認されているため既定はmacro F1。過去の実験を再現する場合のみ`val_loss`を指定する |
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
python -m scripts.check_leakage --labels data/labels.csv
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

## 確信度(表示する%)の校正

### 何が問題だったか

学習済みモデルの生の出力をそのまま確信度として表示すると、**人が見れば明らかに違う
判定に60%前後の数字が付く**ことがあった。これはモデルの実力の問題ではなく、
確率の読み方の問題である。

原因は学習時の`pos_weight`。`src/train.py`は少数ラベルの取りこぼしを防ぐため
`BCEWithLogitsLoss(pos_weight=w)`で学習している。この損失を最小にする出力は、
真の確率 p に対して

```
sigmoid(z) = w·p / (w·p + (1 - p))        逆に解くと    p = sigmoid(z - log w)
```

となり、**構造的に p より大きい**。既定の`--pos-weight-cap 8`が効いているラベルなら、

```
表示 60%  →  実体は sigmoid(logit(0.6) - log 8) ≒ 16%
```

でしかない。さらに`w`はラベルごとに違うので、かさ上げの量もラベルごとに違い、
ラベル間で確信度を比較すること自体が成り立っていなかった
(10ラベルの最大値を取る`scripts/classify_dates.py`は、この歪みを直接受ける)。

これに、データが少ないCNNにありがちな自信過剰が上乗せされる。

### どう直したか

検証データを使い、ラベルごとに `p = sigmoid(a·z + b)` の a, b を当てはめ直す
(Platt scaling)。`a=1, b=-log w`が上の`pos_weight`の補正そのものなので、この形は
解析的な補正を含んでいる。検証データに陽性がほとんど無いラベルは当てはめが
過学習するため、`-log w`の解析的な補正だけを使う。

**学習済みの重みは一切変えない。** 確率への直し方(と判定しきい値)だけを調整する。

```bash
# 学習すると自動で <重み名>.calib.json が作られる（--no-calibration で無効化できる）
python -m src.train --data-dir data/processed --labels data/labels.csv --test-ratio 0.1

# 既存の重みに後から校正を付ける / 効き目を確認する
python -m scripts.calibrate \
    --data-dir data/processed --labels data/labels.csv \
    --weights weights/model.pt --test-ratio 0.1 --report-split test
```

`scripts/predict.py`・`webapp`・`scripts/classify_dates.py`・Grad-CAMは、重みの隣に
`<重み名>.calib.json`があれば自動的に読み込む。無ければ従来どおり生の値を表示し、
未校正である旨を出力する(既存の重みがそのまま動く)。

### この修正で変わるもの・変わらないもの

Platt scaling は `a>0` である限り**単調変換**なので、ラベルごとに閾値を選び直せば
予測されるラベル集合は1件も変わらない。`scripts/cross_validate.py` は
`--optimize-thresholds` で走っており、`find_best_thresholds` が検証データで
閾値を選び直しているため、**過去の交差検証の数値は校正の影響を受けない**。

| 数値 | 校正の影響 |
|---|---|
| 交差検証の macro F1 | 変わらない(閾値を選び直せば予測が同一のため) |
| macro AP | 変わらない(順位だけで決まるため) |
| ラベル別 F1 / precision / recall | 変わらない |
| `predict` / `webapp` が表示する確信度 | **大きく変わる**(本題) |
| `classify_dates.py` の「その日の気圧配置」 | **変わる**(10ラベルの最大値=ラベル間比較のため) |

つまりこれは「今までの成績が間違っていた」という修正ではない。**表示と、
ラベルをまたいで比べる処理**を直すものである。`classify_dates.py` は後者に
あたるため、これまで誤った気圧配置を出していた可能性がある。

### 効き目の見かた

`scripts/calibrate.py`と`src/evaluate.py`が、**1位に出したラベルの確信度**と
**それが実際に正解だった割合**を突き合わせた表(信頼度図)を出す。

```
[校正前] 平均確信度 75.9% / ECE 0.759
  確信度の範囲       件数    平均確信度   実際の正解率      ズレ
   70%〜 80%        39      75.9%       0.0%    +75.9pt  ←表示が高すぎる
```

ECEは「表示した%」と「実際に当たった割合」のズレの平均で、0に近いほど画面の数字が
そのまま当たる確率として読める。校正後にこの値が下がっていれば、60%と書いてある
判定はおよそ10回に6回当たる、という意味になっている。

### 確信度が低いときは判定を出さない

校正すると、自信の無い判定が正直に低い数字で出るようになる。それを使って、
しきい値に届かない事例は答えを断定しない。

- `scripts/predict.py` / Web UI: どのラベルもしきい値を超えなければ「判定保留」と表示する
- `scripts/classify_dates.py`: `判定`列が`要確認`になり、集計でも確定分と分けて数える
  (`--min-confidence`でしきい値を上書き、`--drop-uncertain`で気圧配置列を空にできる)

しきい値はラベルごとに、校正後の確率でF1が最大になる値を検証データから選んでいる。
校正後の確率は少数ラベルほど小さい値に収まるため、一律0.5では拾えなくなるため。

### 注意

- 校正は**確信度の意味を直すもので、当たる・当たらないを改善するものではない**。
  モデルが自信を持って間違える事例は残る。ただし校正後はそれが件数として見えるので、
  信頼度図の「高い確信度の帯で正解率が低い」行を見れば、どのくらい残っているかが分かる。
- 重みごとに校正が必要(foldごとの重みは、それぞれのvalで当てはめる)。
  `scripts/classify_dates.py`に複数の重みを渡す場合、一部だけ校正済みだと確率の意味が
  揃わないためエラーになる。
- 同梱の`weights/model.pt`にはまだ校正ファイルが付いていない。上のコマンドで
  `weights/model.calib.json`を作ってコミットすると、Colabノートブックでも効くようになる。
  ただしこの重みは`data/labels.csv`(2432件)より前にコミットされたもので、どの
  ラベル付けで学習されたかがリポジトリからは追えない。**学習し直してから校正するのが確実。**

### 古い校正ファイルの取り違えを防ぐ

校正は重みごとに違うので、**学習し直したら作り直さなければならない**。古い校正
ファイルが新しい重みの隣に残っていると、確率の直し方だけが古いまま「静かに間違う」。
これを機械的に防ぐため、校正ファイルは自分が何から作られたかを記録する。

```json
"source": {
  "weights_sha256": "881b168d7dff378a",   // 重みの中身の指紋
  "labels_csv": "data/labels.csv",
  "labels_sha256": "0302229244f858cc",    // ラベルCSVの中身の指紋
  "image_size": 224,
  "pos_weight": [8.0, 8.0, 3.5, ...],
  "pos_weight_source": "checkpoint",      // または "recomputed"(学習データから再計算)
  "pos_weight_cap": 8.0,
  "split": {"mode": "temporal", "val_ratio": 0.2, "test_ratio": 0.1, "seed": 42, ...},
  "created_at": "2026-08-22T08:30:29+00:00"
}
```

読み込み時に重みの指紋を計算して突き合わせ、食い違えば推論を**止める**
(`StaleCalibrationError`)。黙って古い校正を使うことはない。

```
エラー: 校正ファイルが、隣にある重みとは別の重みから作られています。
  重み        : weights/model.pt(指紋 881b168d7dff378a)
  校正ファイル: weights/model.calib.json(指紋 9e00a4a93d21cc65 の重み用)
  ...
  1) python -m scripts.calibrate で校正を作り直す(推奨)
  2) weights/model.calib.json を削除する(未校正の生の値に戻る)
```

Webサーバーは起動自体は成功するが、モデルを読み込まず`/predict`が503でこの理由を返す
(未校正の値を黙って配信するより安全なため)。

### 時間的リークについて

`temporal`分割は日付順に train → val → test と切るだけで、**境界の2か所には間隔が無い**
(`--gap-days`が効くのは`loyo`のときだけ)。学習の最終日と検証の初日は1日しか離れておらず、
天気図はほぼ同じになる。校正パラメータもECEも、その数件のぶんだけ実力より良く出る。

`scripts/calibrate.py`は勝手に除外せず(除外すると過去の実験と分割が変わって比較できなくなる)、
件数を出す。

```
  時間的リークの目安: val 486件のうち6件(1.2%)が学習データと3日以内。最短1日
  時間的リークの目安: testは学習データから3日以上離れています
```

報告するECEは`--report-split test`で測ること。valは校正を当てはめた側なので、
そこで測った値は必ず良く出る。

### テスト

```bash
python tests/test_calibration.py        # pytest不要
python -m pytest tests/ -q              # pytestがあれば
```

## 交差検証(報告用の数値はこれで出す)

時系列で後ろを切り出す分割には弱点がある。テスト期間が特定の季節に偏るため、
台風のように夏にしか現れないラベルがテストセットに1件も入らず**評価できない**。

`leave-one-year-out`(1年ずつテストに回す)ならテストが必ず通年になるので、
全ラベルが評価対象に入る。さらにfoldごとに推定値が得られるので、
**平均±標準偏差**で報告でき、差が誤差かどうかを判断できる。

```bash
python -m scripts.cross_validate \
    --data-dir data/processed --labels data/labels.csv \
    --years 2023 2024 2025 --out-dir runs/loyo
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

学習した重みには入力解像度とラベル一覧が同梱される。`predict.py`・`evaluate.py`・Grad-CAM・
Web UIはこれを読んで前処理を自動的に合わせるため、`--image-size`を変えても推論側の指定は不要。
(EfficientNetは適応的プーリングを使うため、サイズが食い違ってもエラーにならず「黙って精度が
落ちる」。それを防ぐための仕組み。)

**注意**: `--out`を省略すると毎回`weights/model.pt`に上書き保存される。過去のチェックポイントを
残したい場合は、再学習のたびに `--out weights/model_YYYYMMDD.pt` のように日付やバージョンを
含めたファイル名を指定すること。上書きしてしまうと、その時点の重みは復元できない。

### 論文の図として出す(ラベルを指定して成功例・失敗例を並べる)

`scripts/gradcam_report.py` は、ラベルを1つ指定して、そのラベルが正解となっている
テスト画像から**当てられた例と見逃した例**を並べた図を作る。得意なラベルと苦手な
ラベルを対比させると、何を手がかりにしているのかが読み取りやすい。

```bash
# 得意な例(台風)
python -m scripts.gradcam_report \
    --data-dir data/processed --labels data/labels.csv \
    --weights runs/loyo/model_test2025.pt \
    --years 2023 2024 2025 --split-mode loyo --test-year 2025 \
    --label typhoon --out runs/loyo/gradcam_typhoon.png

# 苦手な例(オホーツク海高気圧)
python -m scripts.gradcam_report ... --label okhotsk_high --out runs/loyo/gradcam_okhotsk.png
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
- [ ] データ収集スクリプトの実装・実行
- [ ] ラベリング（初期セット数百枚）
- [ ] モデル学習・評価
- [ ] 推論APIとWeb UI
