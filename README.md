# 天気図パターン分類AI (weather-pattern-classification)

機械学習（CNN）を用いて天気図を気圧配置パターンごとに自動分類するプロジェクトです。
最終的にはWebサイト上でアップロードした天気図画像（またはERA5由来の気圧場）を分類できるようにします。

## すぐに使う

### 検出した枠を描き込んでから分類する(注釈方式)

高低気圧を先に検出して枠を描き、その画像で分類する方式が使える。素の天気図で
学習したモデルより macro F1 が +0.021 高い(3年とも同じ向き。
`docs/2026-08-27-features-vs-cnn.md`)。

**成績とは別に、Grad-CAM では示せないものが示せる。**Grad-CAM は「モデルが
どこを見たか」だが、枠は「**検出が当たったか**」を示す。運用でAIの判定を人が
検証する場面ではこちらのほうが直接的である。

```bash
python scripts/predict.py path/to/天気図.png --annotate \
    --weights weights/model_annot.pt --save-annotated annotated.png
```

`--annotate` と `--weights weights/model_annot.pt` は**必ず組にすること。**
入力の見た目が学習時と違うと、モデルは見たことのない絵を渡されて成績が静かに
落ちる。Colabノートブックでは `USE_ANNOTATION` にチェックを入れると同じことができる。

必要なもの:

| | 置き場所 | 用意する人 |
|---|---|---|
| H/L の文字テンプレート | `data/templates/` | **人が手で切り出す**(同ディレクトリのREADME) |
| 中心の印のテンプレート | `data/marks/` | 同上。無いと検出が半減する |
| 注釈方式の重み | `weights/model_annot.pt` | `runs/cv_annot_boxes/model_test<年>.pt` を複製 |

**検出が0個になるとき**の原因は2つある。

1つ目は**線が真っ黒でない天気図**(紙のスキャンなど)。等圧線の色帯は
気象庁PDF版に合わせて V<=90 と決め打ちしてあり、線の濃さが V=110 を超えると
マスクが空になって1個も取れない。合成天気図では 10.82% -> 0.00%、検出 3個 -> 0個。
**これは自動で救うようにしてある**(`src.chartsymbols.ink_mask`)。色帯が
ほとんど空のときだけ濃さのしきい値(大津の方法)に切り替わる。色帯が読める
天気図では1画素も変わらないので、記録済みの結果は再現できる。切り替わったときは
`predict.py` がその旨を表示する。**色を見ないぶん海岸線も線として拾うので、
枠の位置は目で確かめること。**

2つ目は解像度の違い。テンプレートは
特定の天気図から切り出したもので、`cv2.matchTemplate` は大きさの違いに対応
しない。**解像度が違うと同じ H でも画素数が違い、1個も当たらない。**
`--scale` では直らない(画像とテンプレートの両方に同じ倍率がかかるので、
相対的な大きさが変わらない)。原因の切り分けはこれで行う:

```bash
python -m scripts.diagnose_detection --images <うまくいく天気図> --images <いかない天気図>
```

倍率ごとの検出数を出し、解像度の違いなのか、白黒スキャンで色帯が空なのか、
記号の書体が違うのかを見分ける。解像度の違いなら `--letter-size <倍率>` で
直る(`predict.py`・`annotate_charts.py`・`build_features.py` のすべてにある)。
`--letter-size auto` にすると `data/templates/reference.json` の基準幅
(テンプレートを切り出した天気図の幅)との比から自動で決める。

**2000〜2022年の天気図について(解決済み)**: 国立国会図書館由来の天気図は、
低解像度で変換すると前処理後が約1052pxになり、記号の高さが96px、2023年以降の
テンプレート139pxに対して30%小さくなって**1個も当たらなかった**。同じPDFを
2023年以降と同じ設定(200DPI、生2339x1653)で変換し直すと前処理後1499pxになり、
差は3.2%まで縮まる。この差は `--letter-size auto` が吸収する。実測:

    素のまま        H 2個 / L 4個
    auto(1.032倍)  H 3個 / L 4個   <- 2023年以降の平均 H 2.8 / L 3.9〜4.2 と同水準

しきい値だけは下げる必要がある。テンプレートは2023年以降の天気図から切り出した
ものなので、同じ2023年の記号に一番よく合い、時代が違うとスコアが少し下がる。
同じ天気図で測った結果:

    --threshold 0.65(既定)  10個中 7個(取りこぼし3、誤検出0)
    --threshold 0.60         10個中 9個(取りこぼし1、誤検出0)
    --threshold 0.55         10個中 10個(取りこぼし0、**誤検出0**)

**既定の 0.65 は変えないこと。**記録に残っている成績(`cv_annot_boxes` の
macro F1 0.667、基準 0.647 など)はすべて 0.65 で出したものなので、既定を
変えると2023年以降の検出も変わり、それらの数字がやり直しになる。古い天気図を
扱うときだけ `--threshold 0.55` を明示的に渡す。

つまり**テンプレートの切り直しは不要**で、生画像を同じ設定で変換し直し、
しきい値を 0.55 にすればよい。
なお、前処理後の大きさが揃わないのは異常ではない ― 切っているのは紙ではなく
枠なので、枠が3.2%大きく描かれていれば切り取り後も3.2%大きくなる
(`python -m scripts.preprocess_jma --report` で切り取り位置を確かめられる)。



学習済みモデル(`weights/model.pt`)をリポジトリに同梱しているので、学習不要ですぐに使える。
確信度の校正(`weights/model.calib.json`)も同梱してあるので、表示される%は実際に
当たる割合の目安として読める([確信度の校正](#確信度表示するの校正)を参照)。

同梱している重みは leave-one-year-out の 2023〜2024年で学習したもの
(`runs/v2_chart_spread/model_test2025.pt`)。**2025年は学習に使っていない**ので、
2025年の天気図で試せば実力どおりの判定になる。3年分すべてを使った最終モデルでは
ない点は承知しておくこと。

<a href="https://colab.research.google.com/github/awg-yk/weather-pattern-classification/blob/main/notebooks/predict.ipynb" target="_blank" rel="noopener noreferrer"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

**手元(VS Code)で同じことをする場合**は `notebooks/predict_local.ipynb` を開く。
Colab版と**同じ関数**(`src/quicklook.py`)を呼んでいるので、結果は一致する。
セットアップのセル(clone・pip install)が無いぶん短い。

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
  regions.csv     # ラベルごとの「見るべき領域」。Grad-CAMの当たり判定に使う
  review_okhotsk*.csv  # okhotsk_highを見直したときの判定記録
scripts/
  preflight.py        # **学習を始める前の設定チェック(数秒)**
  cross_validate.py   # leave-one-year-out交差検証。報告用の数値はこれ
  compare_runs.py     # 実行どうしの比較。ラベルを絞って平均を取り直せる
  ensemble_chart_grid.py  # 天気図モデルと格子モデルの確率を混ぜる
  ensemble_chart_features.py # 天気図モデルと検出した特徴量の確率を混ぜる
  annotate_charts.py      # 検出した高低気圧と前線を天気図に描き込む(注釈付き画像)
  collect_jma.py      # 気象庁天気図画像の収集スクリプト
  download_era5.py    # ERA5データ取得スクリプト（CDS API）
  label_tool.py       # ラベル付け・見直し(盲検レビュー)
  regions_preview.py  # data/regions.csv の矩形を天気図に重ねて確認する
  attention_check.py  # Grad-CAMが「見るべき領域」を見ているかをラベル別に測る
  chart_palette.py    # 天気図に実際に使われている色を測る(色マスクの閾値決め)
  extract_fronts.py   # 前線を色マスクだけで取れるか見る(物体検出の下見)
  extract_symbols.py  # H/L/T/TD をテンプレートマッチングで取れるか見る(scan/cluster/grid/cut/match)
  check_alignment.py  # 天気図が日付をまたいで画素単位で揃っているかを測る
  build_features.py   # 検出結果から Phase 3 の特徴量CSVを作る
  cv_features.py      # 特徴量で分類し交差検証(Phase 4)。既存CNNと同じ形で出力
  feature_report.py   # どの特徴量がどのラベルの手がかりかをAUCで測る
  explain_date.py     # 日付を1つ指定して天気図・Grad-CAM・確信度を書き出す
  auto_label_era5.py  # ERA5気圧場からの規則ベース自動ラベリング
src/
  dataset.py      # 天気図画像のDataset
  era5_grid.py    # ERA5格子を直接入力にするDataset
  model.py        # EfficientNet / SmallCNN / CoordConv / FeatureFusion
  train.py        # 学習スクリプト
  evaluate.py     # 評価スクリプト
  calibration.py  # 確信度の校正（表示%を実際の的中率に合わせる）
  split.py        # train/val/testの分け方
  labels.py       # ラベル定義
  regions.py      # ラベルごとの「見るべき領域」とGrad-CAMとの突き合わせ
  chartsymbols.py # 天気図の色マスクと記号拾い（学習を使わない抽出）
  chartfeatures.py# 検出結果を分類用の特徴量にする（Phase 3）
  metrics.py      # しきい値の最適化と自明な予測の基準（torch不要）
docs/
  2026-08-21-chart-vs-era5-grid.md  # 天気図とERA5格子の比較。結論と未解決の課題
  2026-08-22-next-chart-only.md     # ERA5を外したあとの計画
  2026-08-25-calibration-order.md   # 校正と平均の順番を実測で決めた記録
  2026-08-25-label-undercount.md    # 保存済みラベルが妥当な答えを取りこぼしている件
  2026-08-25-typhoon-threshold.md   # 台風のしきい値が高すぎた件
  2026-08-25-attention-regions.md   # 教師データ側に「見るべき領域」を持たせた記録
  2026-08-26-stale-run-comparisons.md # 別のラベルで測った結果を比べていた件
  2026-08-26-detection-plan.md      # 物体検出を経由する方式の計画と引き継ぎ
  2026-08-26-detection-prescreen.md # 上の計画の下見。色とテンプレートで取れるか測った
  2026-08-27-features-vs-cnn.md   # 検出した特徴量での分類を既存CNNと比べた結果
                                  # (macro F1 0.408 対 0.641。オホーツク海高気圧だけ勝つ)
runs/             # 交差検証の出力。summary.jsonだけ追跡する
                  # (2026-08-26に過去分を削除。docs/2026-08-26-stale-run-comparisons.md)
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

### Windowsで `pip install` が「ファイル名または拡張子が長すぎます」で失敗する場合

torchは `licenses\third_party\...` の下にきわめて深い階層のファイルを持っている。
リポジトリ自体が深い場所(`Desktop\...\github\vscode\weather-pattern-classification`
のような)にあると、合計がWindowsのパス260文字の上限を超えて展開に失敗する。

**venvだけを浅い場所に置けば解決する。** リポジトリは動かさなくてよい。

```powershell
py -m venv C:\wpcvenv
C:\wpcvenv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`setup-session.ps1` は `C:\wpcvenv` も探すので、そのまま使える。別の場所に置くなら:

```powershell
[Environment]::SetEnvironmentVariable("WPC_VENV", "<venvのパス>", "User")
```

同じ理由で、`Remove-Item -Recurse` でvenvを消せないことがある。その場合は空の
フォルダをミラーして中身を空にしてから消す:

```powershell
New-Item -ItemType Directory -Force -Path "$env:TEMP\emptydir" | Out-Null
robocopy "$env:TEMP\emptydir" venv /MIR /NFL /NDL /NJH /NJS
Remove-Item -Recurse -Force venv
```

大学や社内ネットワークでプロキシを経由する場合、`git` と `pip` はブラウザと違って
プロキシ設定を自動では読まない。ブラウザでGitHubが開けるのに `git push` が
"Connection was reset" になるときはこれが原因:

```powershell
git config --global http.proxy http://<プロキシ>:<ポート>
[Environment]::SetEnvironmentVariable("HTTPS_PROXY", "http://<プロキシ>:<ポート>", "User")
```

プロキシのアドレスは次で確認できる:

```powershell
Get-ItemProperty "HKCU:\Software\Microsoft\Windows\CurrentVersion\Internet Settings" |
    Select-Object ProxyEnable, ProxyServer, AutoConfigURL
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
| `--aux-features` | なし | 検出から作った数値(`scripts/build_features.py` の出力)も答えさせる。位置を**入力**として渡す `--coordconv` が効かなかったのを受けての手 |
| `--aux-weight` | 0 | 上の損失にかける重み。振って決める。大きすぎると本来の10ラベルが犠牲になる |
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

**ラベルを変えたあとの結果は、変える前の結果と比べられない。**
`scripts/compare_runs.py` は foldごとの陽性件数(support)を突き合わせ、
食い違っていれば比較を拒否する。

```
ラベルが食い違っています。このまま比べると、モデルの差とラベルの差が混ざります。

cv_coordconv と loyo_v2 で陽性件数が違うラベル:
  台風                     loyo_v2=[98, 62, 59]  cv_coordconv=[78, 84, 85]
```

実際に、台風ラベルをベストトラックから付け直したあと、付け直す前の実行と
並べて「+0.192の改善」と読んでしまったことがある(差はモデルではなくラベルの
ものだった)。経緯は `docs/2026-08-26-stale-run-comparisons.md`。

`scripts/cross_validate.py` は `summary.json` にラベルファイルの指紋
(SHA-256の先頭16桁)を記録する。陽性の枚数を変えずに中身を差し替えた修正は
supportでは見抜けないので、指紋も併せて見る。


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

## 確信度(表示する%)の校正

### 何が問題だったか

生の出力をそのまま確信度として表示すると、**人が見れば明らかに違う判定に60%前後の
数字が付く**ことがある。モデルの実力の問題ではなく、確率の読み方の問題である。

原因は学習時の `pos_weight`。`src/train.py` は少数ラベルの取りこぼしを防ぐため
`BCEWithLogitsLoss(pos_weight=w)` で学習している。この損失を最小にする出力は、
真の確率 p に対して

```
sigmoid(z) = w·p / (w·p + (1 - p))        逆に解くと    p = sigmoid(z - log w)
```

となり、**構造的に p より大きい**。既定の `--pos-weight-cap 8` が効いているラベルなら

```
表示 60%  →  実体は sigmoid(logit(0.6) - log 8) ≒ 16%
```

でしかない。さらに `w` はラベルごとに違うので、かさ上げの量もラベルごとに違い、
**ラベル間で確信度を比較すること自体が成り立っていなかった**。10ラベルの最大値を
取る `scripts/classify_dates.py` は、この歪みを直接受ける。

### 温度スケーリングでは直らない

以前の `scripts/calibrate.py` はロジットを1つの数Tで割る温度スケーリングだった。
これは全体の自信過剰をならすが、**`pos_weight` のかさ上げは原理的に消せない**。
`sigmoid(z/T)` は T が何であっても `z=0` を必ず 0.5 に写す。しかし `pos_weight=8` の
ラベルでは生の出力 0.5 の実体は `1/(1+8) = 11%` である。割り算では、この
「ロジットを `log w` ぶん平行移動する」歪みを表現できない。

そこでラベルごとに2つの数を当てはめる(Platt scaling):

```
p = sigmoid(a · z + b)
```

`b` が加法シフトなので `pos_weight` の補正を含む(`a=1, b=-log w` がその解析解)。
検証データに陽性が少ないラベルは当てはめが不安定になるため、その場合は解析解だけを
使う(`method: prior`)。実装は `src/calibration.py`。

### 過去の数値は影響を受けない

Platt scaling は `a>0` である限り**単調変換**なので、ラベルごとに閾値を選び直せば
予測は変わらない。`scripts/cross_validate.py` は `--optimize-thresholds` で走っているため、
**報告済みの macro F1 も macro AP も校正の影響を受けない**
(`tests/test_calibration.py::test_optimized_f1_is_unchanged_by_calibration` で担保)。

| 数値 | 校正の影響 |
|---|---|
| 交差検証の macro F1 / macro AP / ラベル別F1 | 変わらない |
| `predict` / `webapp` が表示する確信度 | **大きく変わる** |
| `classify_dates.py` の「その日の気圧配置」 | **変わる**(ラベル間比較のため) |

`src/evaluate.py` も既定では校正を**適用しない**(測って報告するだけ)。
適用したい場合だけ `--calibrated` を付ける。過去の報告値と地続きにしておくため。

### 使い方

```bash
# 学習すると自動で <重み名>.calib.json が作られる(--no-calibration で無効化)
python -m src.train --data-dir <画像> --labels data/labels_v2.csv ...

# 既存の重みに後から校正を付ける / 効き目を確認する(reports/calibration.png も出る)
python -m scripts.calibrate \
    --data-dir <画像> --labels data/labels_v2.csv \
    --weights runs/v2_chart/model_test2023.pt \
    --years 2023 2024 2025 --split-mode loyo --test-year 2023
```

`scripts/predict.py`・`webapp`・`scripts/classify_dates.py`・Grad-CAM は、重みの隣に
`<重み名>.calib.json` があれば自動的に読み込む。無ければ従来どおり生の値を表示し、
未校正である旨を出力する。

### 複数の重みを混ぜるとき

`scripts/classify_dates.py` に重みを複数渡すと、既定では**生の確率を平均してから、
係数を平均した校正で直す**(`--calibration-order after-average`)。ラベルごとの単調
変換なので、未校正のときと順位が一致し、APやAUCが変わらない。

もう一方の `per-model`(モデルごとに校正してから平均)は、各モデルの歪みを個別に
直せる代わりに平均の順位を変える。風替わり167日をベストトラックで測ったところ、
確信度の質(ECE)で `after-average` に劣ったため既定から外した。経緯と数値は
[`docs/2026-08-25-calibration-order.md`](docs/2026-08-25-calibration-order.md)。

### 確信度が低いときは判定を出さない

校正すると、自信の無い判定が正直に低い数字で出る。それを使って、しきい値に届かない
事例は答えを断定しない。

- `scripts/predict.py` / Web UI: どのラベルもしきい値を超えなければ「判定保留」と表示
- `scripts/classify_dates.py`: `判定`列が`要確認`になり、集計でも確定分と分けて数える
  (`--min-confidence` でしきい値を上書き、`--drop-uncertain` で気圧配置列を空にできる)

### 古い校正ファイルの取り違えを防ぐ

校正は重みごとに違うので、**学習し直したら作り直さなければならない**。古い校正ファイルが
新しい重みの隣に残ると、確率の直し方だけが古いまま「静かに間違う」。これを防ぐため、
校正ファイルは重みとラベルCSVの**中身の指紋**(SHA-256の先頭16桁)、`pos_weight` の値と
その出所、分割の引数、作成日時を記録する。

読み込み時に重みの指紋を突き合わせ、食い違えば推論を**止める**(`StaleCalibrationError`)。
黙って古い校正を使うことはない。Webサーバーは起動するがモデルを読み込まず、
`/predict` が503でその理由を返す。

### 時間的リークについて

`temporal` 分割は日付順に train → val → test と切るだけで、**境界の2か所には間隔が無い**
(`--gap-days` が効くのは `loyo` のときだけ)。学習の最終日と検証の初日は1日しか離れておらず、
校正パラメータもECEもその数件のぶんだけ実力より良く出る。`scripts/calibrate.py` は
勝手に除外せず(除外すると過去の実験と分割が変わって比較できなくなる)、件数を出す。

### 校正で消えるもの・消えないもの

消える: 表示上のかさ上げ。校正後は、確信度60%と出た事例のうち実際に約6割が当たる。
消えない: モデルが本当に自信満々で間違えるケース。ただし信頼度図の
「高い確信度の帯で正解率が低い」行として、どのくらい残っているかが件数で見える。

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

### 教師データ側にも「見るべき領域」を持たせる

Grad-CAMはモデルがどこを見たかを示すが、それが正しい場所かどうかを判断するのは
人だった。ラベルは画像1枚に対して名前を並べるだけで、位置の情報を持っていないためである。
この非対称のせいで、「オホーツク海高気圧と答えたのに本州中部を見ている」といった
誤りは、天気図を1枚ずつ目で見て指摘するしかなかった。

`data/regions.csv` はラベルごとに矩形を1つ持つ。相対座標(左上が0,0・右下が1,1)で、
`scripts/preprocess_jma.py --stamp-box` と同じ取り方。

```
label,x0,y0,x1,y1,note
okhotsk_high,0.50,0.00,0.75,0.30,オホーツク海。高気圧の本体がある海域
```

緯度経度ではなく相対座標なのは、前処理の `autocrop_to_content()` が白縁を落とすため
画素と緯度経度の対応表が無く、天気図は正距円筒図法でもないので線形変換で緯度経度に
直すと嘘の精度が付くため。全画像が同じ基準でトリミングされているので、相対座標なら
画像間で揃う。

**測る前に必ず1回、矩形が実際の海域と合っているかを目で確認する。**

```bash
python -m scripts.regions_preview \
    --image ../weather-pattern-classification-data/processed/Js_2025050100.png \
    --out reports/regions_preview.png
```

ずれていたら `data/regions.csv` の x0,y0,x1,y1 を直す。同梱の値は実物の天気図
(2019-06-15、手動アーカイブのJPEG)に重ねて読んだもので、緯度経度から変換したものでは
ない。この天気図は緯線が弧を描くため、目盛りから作った線形の対応式では図の中央で
北にずれる。**2022-10-01以降の気象庁PDF配信の天気図で枠の取り方が同じかは未確認。**

#### 測る

```bash
python -m scripts.attention_check \
    --images-dir ../weather-pattern-classification-data/processed \
    --labels data/labels_v2.csv --weights weights/model.pt \
    --years 2025 --out reports/attention_2025.csv
```

| 列 | 意味 |
|---|---|
| `mass` | Grad-CAMの総和のうち矩形の中にある割合(0〜1) |
| `area` | 矩形が画像に占める面積の割合。注目が一様なときの `mass` の期待値 |
| `lift` | `mass / area`。**1なら画像全体に一様に注目しているのと同じ**で、その気圧配置を位置で捉えられていない。大きいほど良い |
| `peak的中率` | 最も強く見ている点が矩形の中にあった割合(pointing game) |

`mass` だけを見てはいけない。矩形が広いラベル(西高東低は面積45%)は何もしなくても
`mass` が高く出る。判断は `lift` で行う。

`--on` で測る対象を切り替える。

| | 対象 | 何が分かるか |
|---|---|---|
| `--on record`(既定) | 記録にそのラベルがある天気図 | 正解のときに正しい場所を見ているか |
| `--on predicted` | モデルがそのラベルを主張した天気図 | そう答えたとき、どこを見て答えたか。誤検出も含むので、適合率が低いラベルの原因はこちらで見る |

#### 1枚の天気図で見比べる

`show_gradcam(..., show_regions=True)` にすると、ヒートマップの上に矩形が白枠で重なり、
見出しに「枠内 何% / 面積 何%」が出る。

#### 使わない場面

この矩形は**学習には使っていない**。損失にも入力にも影響しないので、
`data/regions.csv` を書き換えてもモデルの出力は変わらない。測るためだけのもの。

## GitHubの画面から実行する(GitHub Actions)

日付を1つ入れて結果を見るだけなら、手元に環境を作らなくてよい。
リポジトリの **Actions** タブ → 「天気図を判定する(日付を指定)」 → **Run workflow**。

| 入力 | 既定 | 意味 |
|---|---|---|
| `date` | (必須) | YYYY-MM-DD |
| `hour` | `0` | 観測時刻(UTC)。0Zは日本時間9時、12Zは21時 |
| `top_k` | `3` | Grad-CAMを描くラベル数 |
| `weights` | `weights/model.pt` | 使う重み |
| `show_regions` | on | 「見るべき領域」の枠を重ねるか |

全ラベルの確信度は実行画面にそのまま出る。Grad-CAMの画像とCSVは
そのページの **Artifacts** からダウンロードする。

実行のたびにリポジトリをクローンし直すので、古いラベルファイルや古い重みが
手元に残っていて結果が変わる、という事故が起きない。

**学習はできない。** GitHubの実行環境にGPUは無く、1ジョブ6時間の上限もある。
学習は手元のPCか、GPUの付いたColabで行う。

同じことを手元で実行する場合:

```bash
python -m scripts.explain_date --date 2025-08-10 --hour 0 \
    --weights weights/model.pt --out-dir reports/explain
```

依存は `requirements-inference.txt` だけでよい(`requirements.txt` は
cartopyやnetCDF4を含み、推論には要らない)。2022-10-01以降の日付はPDF配信なので
popplerが要る。

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
