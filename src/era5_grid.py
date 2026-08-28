"""ERA5の生格子(圧縮なし)を、天気図画像の代わりにCNNへ入力するデータセット。

scripts/era5_features.py は緯度経度ごとの領域平均などに要約して26個の数値に
圧縮しているが、その過程で「二つ玉低気圧がどこに2つあるか」のような空間的な
形の情報が失われる。天気図の等圧線は結局この気圧場を等値線として描いたもの
であり(前線の記号を除けば)、生の格子は天気図の上位互換のはずだ、という仮説を
検証するために作った。

前線(front_passage / stationary_front)は人間が判断して描いた記号であり、
気圧・気温の格子には含まれない。この入力ではその2ラベルは原理的に不利になる
ことを織り込んだうえで、それ以外のラベルで天気図画像に勝てるかを見る。
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.era5 import open_era5
from src.labels import LABEL_TO_INDEX, LABELS, parse_labels

NUM_CHANNELS = 2  # 海面更正気圧(hPa) ・ 850hPa気温(℃)


def _resize(field: np.ndarray, size: int) -> np.ndarray:
    tensor = torch.from_numpy(field.astype("float32")).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)
    return resized.squeeze(0).squeeze(0).numpy()


def _time_name(da) -> str:
    for name in ("valid_time", "time"):
        if name in da.coords:
            return name
    raise KeyError(f"時刻の座標が見つかりません: {list(da.coords)}")


_warned_about_downsampling = False


def _warn_if_downsampling(native_shape, grid_size: int) -> None:
    """元の格子より粗くリサイズしていたら知らせる。

    grid_sizeが元の格子点数を下回ると、低気圧の中心付近のような細かい構造から
    順に失われる。エラーにはならず精度が下がるだけなので、黙っていると
    「ERA5には情報が無い」と読み違えかねない。年ごとに呼ばれるので一度だけ出す。
    """
    global _warned_about_downsampling
    if _warned_about_downsampling:
        return
    native_y, native_x = native_shape
    if grid_size < min(native_y, native_x):
        print(
            f"注意: ERA5の元の格子は{native_y}×{native_x}点ですが、"
            f"{grid_size}×{grid_size}に縮めています。細かい構造が失われます"
            f"(--grid-size {max(native_y, native_x)} までは情報が増えます)"
        )
    _warned_about_downsampling = True


class ERA5GridDataset(Dataset):
    """labels.csvの各行(=1枚の天気図)に対応する日時のERA5格子を返す。

    __getitem__ の返り値は (格子, 空の特徴量テンソル, ラベル) の3要素。
    WeatherMapDatasetと形を揃えることで、src/train.pyのrun_epochをそのまま使える。

    正規化(平均・標準偏差)は構築時には行わない。検証・テストの分を含めて
    統計を取ると情報が漏れるため、学習用サブセットが決まってからset_stats()で
    与える(src/train.py・src/evaluate.pyを参照)。
    """

    def __init__(self, labels_csv, era5_dir, years=None, grid_size: int = 128, augment: bool = False):
        df = pd.read_csv(labels_csv)
        df["parsed_labels"] = df["label"].apply(parse_labels)
        df = df[df["parsed_labels"].apply(len) > 0].reset_index(drop=True)
        if df.empty:
            raise ValueError(f"{labels_csv} に認識できるラベルを持つ行がありません")

        stamps = df["filename"].str.extract(r"(\d{10})")[0]
        df["parsed_datetime"] = pd.to_datetime(stamps, format="%Y%m%d%H", errors="coerce")
        missing_dates = df["parsed_datetime"].isna()
        if missing_dates.any():
            raise ValueError(
                f"{labels_csv} のうち{int(missing_dates.sum())}行でファイル名から日付を取り出せません。"
                f"例: {df.loc[missing_dates, 'filename'].iloc[0]!r}"
            )

        if years:
            years = {int(y) for y in years}
            before = len(df)
            df = df[df["parsed_datetime"].dt.year.isin(years)].reset_index(drop=True)
            print(f"対象年 {sorted(years)} に限定: {before}件 → {len(df)}件")
            if df.empty:
                raise ValueError(f"指定した年 {sorted(years)} に該当する行がありません")

        self.df = df
        self.feature_cols = []  # WeatherMapDatasetとの互換用(ERA5数値特徴の併用はしない)
        self.grid_size = grid_size
        self.augment = augment
        self.mean = None
        self.std = None

        era5_dir = Path(era5_dir)
        grids = np.empty((len(df), NUM_CHANNELS, grid_size, grid_size), dtype="float32")
        missing_rows = []

        for year in sorted(df["parsed_datetime"].dt.year.unique()):
            mslp_path = era5_dir / f"era5_mslp_{year}.nc"
            t850_path = era5_dir / f"era5_t850_{year}.nc"
            for path in (mslp_path, t850_path):
                if not path.exists():
                    raise SystemExit(
                        f"{path} がありません。\n"
                        f"  python -m scripts.download_era5 --years {year} "
                        f"{'--variable t850 ' if path is t850_path else ''}を実行してください。"
                    )

            with open_era5(mslp_path) as ds:
                mslp = (ds["msl"] / 100.0).load()
            with open_era5(t850_path) as ds:
                t850 = (ds["t"] - 273.15).load()
                if "pressure_level" in t850.dims:
                    t850 = t850.isel(pressure_level=0)

            _warn_if_downsampling(mslp.shape[-2:], grid_size)

            mslp_tname, t850_tname = _time_name(mslp), _time_name(t850)
            mslp_index = {pd.Timestamp(v): i for i, v in enumerate(mslp[mslp_tname].values)}
            t850_index = {pd.Timestamp(v): i for i, v in enumerate(t850[t850_tname].values)}

            for row in df.index[df["parsed_datetime"].dt.year == year]:
                stamp = df.at[row, "parsed_datetime"]
                if stamp not in mslp_index or stamp not in t850_index:
                    missing_rows.append(df.at[row, "filename"])
                    continue
                grids[row, 0] = _resize(mslp.isel({mslp_tname: mslp_index[stamp]}).values, grid_size)
                grids[row, 1] = _resize(t850.isel({t850_tname: t850_index[stamp]}).values, grid_size)

        if missing_rows:
            raise ValueError(
                f"labels.csvの{len(missing_rows)}件に対応するERA5の時刻が見つかりません"
                f"(先頭5件: {missing_rows[:5]})。\n"
                "ダウンロード範囲(年・時刻)を確認してください。"
            )
        if not np.isfinite(grids).all():
            raise ValueError("ERA5の格子に欠損値(NaN/inf)が含まれています。ダウンロードをやり直してください。")

        self.grids = grids  # 正規化前の生の値。set_stats()を呼ぶまでは使えない。

    def set_stats(self, mean, std) -> None:
        """学習用サブセットだけから求めた平均・標準偏差を設定する。

        検証・テストのデータを含めて計算すると、そこにしか無い情報が
        学習側に漏れる(FeatureFusion.set_feature_statsと同じ理由)。
        """
        mean = np.asarray(mean, dtype="float32").reshape(NUM_CHANNELS, 1, 1)
        std = np.asarray(std, dtype="float32").reshape(NUM_CHANNELS, 1, 1)
        self.mean = mean
        self.std = np.clip(std, 1e-6, None)  # 一定値の特徴で0除算にならないようにする

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        if self.mean is None:
            raise RuntimeError(
                "set_stats()を呼ぶ前に読み出そうとしました。"
                "train.py/evaluate.pyが学習用サブセットから統計を計算して渡す必要があります。"
            )
        grid = (self.grids[idx] - self.mean) / self.std
        grid = torch.from_numpy(grid.copy())
        if self.augment:
            # 天気図の色調変動・微小なアフィン変換に相当する軽い摂動。
            # 反転はしない(反転は「西高東低」を「東高西低」に変えてしまう。画像側と同じ理由)。
            grid = grid + torch.randn_like(grid) * 0.02
            shift_y, shift_x = (int(v) for v in torch.randint(-2, 3, (2,)))
            grid = torch.roll(grid, shifts=(shift_y, shift_x), dims=(1, 2))

        target = torch.zeros(len(LABELS), dtype=torch.float32)
        for label in self.df.at[idx, "parsed_labels"]:
            target[LABEL_TO_INDEX[label]] = 1.0
        # 4つ組にそろえる。src/dataset.py と同じ形にしないと同じ学習ループを
        # 通せない。補助の答えは格子側では使わないので空で返す
        return grid, torch.empty(0), target, torch.empty(0)


def compute_grid_stats(dataset: ERA5GridDataset, indices) -> tuple:
    """指定した行だけから、チャンネルごとの平均・標準偏差を計算する。"""
    subset = dataset.grids[indices]
    mean = subset.mean(axis=(0, 2, 3))
    std = subset.std(axis=(0, 2, 3))
    return mean, std
