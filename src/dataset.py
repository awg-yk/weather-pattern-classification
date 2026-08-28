import re
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.labels import LABEL_TO_INDEX, LABELS, parse_labels  # noqa: F401  (従来の import 経路を保つ)
# 観測日時の解釈は src/split.py に移した。分割がそれを要求する当人であり、
# torch を必要としないので木のモデルからも同じものを使える。日付の解釈が
# ずれると分割ごとずれるので、実装は1つだけにする。
from src.split import (  # noqa: F401  (従来の import 経路を保つ)
    DATE_IN_FILENAME,
    index_images_by_stamp,
    parse_datetime,
)


class WeatherMapDataset(Dataset):
    """labels.csv (filename,label,...) と画像ディレクトリからサンプルを返すマルチラベルDataset。

    label列はパイプ区切りで複数ラベルを持てる(例: "winter_pressure_pattern|japan_sea_low")。
    単一ラベルの行もそのまま扱える。
    """

    def __init__(self, images_dir: str, labels_csv: str, transform=None, years=None,
                 features_csv=None, aux_csv=None, aux_columns=None):
        """years に年のリストを渡すと、その年の画像だけを対象にする。

        「2023〜2025年の3年分」のように対象期間を区切って実験するときに使う。
        ここで絞り込んでおくことで、src/split.py が返す行番号と
        Subset のインデックスが常に一致する。

        features_csv に scripts/era5_features.py の出力を渡すと、画像に加えて
        ERA5の数値特徴も返すようになる。特徴量が無い行は学習に使えないため除外する。

        aux_csv に scripts/build_features.py の出力を渡すと、**学習の答えとして
        使う数値**(補助の答え)も返すようになる。features_csv とは向きが逆で、
        こちらは入力ではなく出力側である(`src/model.py` の AuxiliaryTargets)。

        補助の答えが無い行は**除外しない**。NaN のまま返し、損失の側で飛ばす
        (`src/train.py` の masked_mse)。10ラベルの学習は続けられるので、
        検出できなかった日を捨てる理由がない。
        """
        self.images_dir = Path(images_dir)
        self.transform = transform

        df = pd.read_csv(labels_csv)
        rows_in_csv = len(df)
        labels_head = df["filename"].iloc[0] if rows_in_csv else None
        df["parsed_labels"] = df["label"].apply(parse_labels)
        df = df[df["parsed_labels"].apply(len) > 0].reset_index(drop=True)
        if df.empty:
            raise ValueError(
                f"{labels_csv} に、現在のsrc/labels.pyで認識できるラベルを持つ行がありません"
                f"(CSVの行数: {rows_in_csv})。\n"
                "ラベル名を変更した直後であれば scripts/merge_labels.py で移行してください。"
            )
        # 時間ブロック分割(src/split.py)が使う観測日時。
        # 空のリストからは日付型のSeriesが作れないため、明示的に変換しておく
        # (そうしないと後段の .dt が分かりにくいAttributeErrorで落ちる)。
        df["parsed_datetime"] = pd.to_datetime(
            pd.Series(
                [
                    parse_datetime(fn, dt)
                    for fn, dt in zip(df["filename"], df.get("date", [None] * len(df)))
                ],
                index=df.index,
            )
        )
        if df["parsed_datetime"].isna().all():
            raise ValueError(
                f"{labels_csv} のどの行からも観測日時を取り出せませんでした。\n"
                "filename列にYYYYMMDDHH(例: Js_2025010100.png)が含まれている必要があります。\n"
                f"最初の行のfilename: {df['filename'].iloc[0]!r}"
            )

        if years:
            years = {int(y) for y in years}
            before = len(df)
            df = df[df["parsed_datetime"].dt.year.isin(years)].reset_index(drop=True)
            print(f"対象年 {sorted(years)} に限定: {before}件 → {len(df)}件")
            if df.empty:
                raise ValueError(f"指定した年 {sorted(years)} に該当する画像がありません")

        # 画像の実在チェックは学習を始める前にまとめて行う。__getitem__に任せると
        # 1エポック目の途中で落ち、そこまでの計算が無駄になるため。
        available = index_images_by_stamp(self.images_dir)
        resolved = [
            available.get(stamp.strftime("%Y%m%d%H")) for stamp in df["parsed_datetime"]
        ]
        missing = [fn for fn, path in zip(df["filename"], resolved) if path is None]
        if missing:
            by_year = Counter(str(fn)[3:7] for fn in missing)
            raise FileNotFoundError(
                f"labels.csvに載っている画像のうち{len(missing)}件が{self.images_dir}にありません"
                f"(対象{len(df)}件中)。\n"
                f"  年別: {dict(sorted(by_year.items()))}\n"
                f"  例  : {missing[:5]}\n"
                f"  そのディレクトリで見つかった画像: {len(available)}件"
                f"{'(例: ' + next(iter(available.values())).name + ')' if available else ''}\n"
                "scripts/collect_jma.py で取得・変換し直してください。"
            )
        # 照合できた実ファイルのパスを持ち回る。以降はfilenameではなくこれを読む。
        df["image_path"] = resolved

        self.feature_cols = []
        self.features = None
        if features_csv:
            feats = pd.read_csv(features_csv)
            self.feature_cols = [c for c in feats.columns if c not in ("filename", "datetime")]
            before = len(df)
            df = df.merge(
                feats[["filename", *self.feature_cols]], on="filename", how="inner"
            ).reset_index(drop=True)
            print(f"ERA5特徴量({len(self.feature_cols)}個)を結合: {before}件 → {len(df)}件")
            if df.empty:
                raise ValueError(
                    "ラベルとERA5特徴量が1件も結合できませんでした。\n"
                    f"  labels.csvのfilename例: {labels_head!r}\n"
                    f"  特徴量CSVのfilename例 : {feats['filename'].iloc[0]!r}"
                )
            self.features = torch.tensor(
                df[self.feature_cols].to_numpy(dtype="float32"), dtype=torch.float32
            )

        # 補助の答え。**行は減らさない。**画像を1枚も捨てずに済むよう、
        # 見つからない行は NaN にして損失の側で飛ばす
        self.aux_cols = []
        self.aux = None
        self.aux_mean = None
        self.aux_std = None
        if aux_csv:
            table = pd.read_csv(aux_csv)
            if "filename" not in table.columns:
                raise ValueError(f"{aux_csv} に filename 列がありません。")
            self.aux_cols = list(aux_columns) if aux_columns else [
                c for c in table.columns if c not in ("filename", "date", "datetime")
            ]
            missing = [c for c in self.aux_cols if c not in table.columns]
            if missing:
                raise ValueError(f"{aux_csv} に無い列が指定されました: {missing}")
            # filename で突き合わせる。行番号で並べると、ずれても学習は最後まで
            # 走り、別の日の位置を答えとして教えることになる
            values = (table.set_index("filename")
                      .reindex(df["filename"].values)[self.aux_cols]
                      .to_numpy(dtype="float32"))
            found = int((~np.isnan(values).all(axis=1)).sum())
            print(f"補助の答え({len(self.aux_cols)}個)を結合: {len(df)}件中{found}件に値あり")
            if found == 0:
                raise ValueError(
                    "補助の答えが1件も突き合いませんでした。filename を確認すること。\n"
                    f"  labels.csvのfilename例: {labels_head!r}\n"
                    f"  補助CSVのfilename例   : {table['filename'].iloc[0]!r}"
                )
            self.aux = torch.tensor(values, dtype=torch.float32)

        self.df = df

    def set_aux_stats(self, mean, std) -> None:
        """補助の答えを標準化するための平均と標準偏差を渡す。

        **学習データの行だけから求めること**(`compute_aux_stats`)。全行から
        求めると検証・テストの分布が学習に漏れる。`src/era5_grid.py` の
        set_stats と同じ約束である。
        """
        self.aux_mean = torch.as_tensor(mean, dtype=torch.float32)
        self.aux_std = torch.as_tensor(std, dtype=torch.float32).clamp(min=1e-6)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        target = torch.zeros(len(LABELS), dtype=torch.float32)
        for label in row["parsed_labels"]:
            target[LABEL_TO_INDEX[label]] = 1.0

        # ERA5を使わない場合も要素数0のテンソルを返し、返り値の形を常に揃える。
        # 学習・評価のループが分岐を持たずに済む。
        features = self.features[idx] if self.features is not None else torch.empty(0)
        if self.aux is None:
            aux = torch.empty(0)
        else:
            aux = self.aux[idx]
            if self.aux_mean is not None:
                aux = (aux - self.aux_mean) / self.aux_std
        return image, features, target, aux


def compute_aux_stats(dataset, indices) -> tuple:
    """指定した行だけから、補助の答えの平均・標準偏差を求める。

    **学習データの行だけを渡すこと。**全行から求めると、検証・テストの分布が
    学習に漏れる。`src/era5_grid.py` の compute_grid_stats と同じ約束。

    値の無い行(NaN)は飛ばして数える。1つも値の無い列は標準偏差0になるので、
    set_aux_stats 側で下限を入れてある。
    """
    values = dataset.aux[list(indices)].numpy()
    # 全部が欠測の列では nanmean が警告を出す。返り値は下で 0/1 に均すので黙らせる
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(values, axis=0)
        std = np.nanstd(values, axis=0)
    # 1件も値が無い列は NaN になる。0と1にしておけば、その列は常に NaN のまま
    # 損失から飛ばされる
    return np.nan_to_num(mean, nan=0.0), np.nan_to_num(std, nan=1.0)
