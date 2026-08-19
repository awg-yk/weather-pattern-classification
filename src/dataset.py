import re
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.labels import LABEL_TO_INDEX, LABELS

# ファイル名に含まれるYYYYMMDDHH。"Js_2001070300_page001.jpg" にも
# "Js_2025010100.png" にも対応する。
_DATE_IN_FILENAME = re.compile(r"(\d{10})")


def parse_labels(label_field: str) -> list:
    """"winter_pressure_pattern|japan_sea_low" のようなパイプ区切りを分解する。"""
    return [l for l in str(label_field).split("|") if l in LABEL_TO_INDEX]


def parse_datetime(filename: str, date_field=None) -> pd.Timestamp:
    """観測日時を取り出す。取れなければ NaT を返す。

    labels.csvのdate列ではなくファイル名を優先して見る。date列は、ファイル名から
    日付を抜き出すロジックを修正する前にラベル付けした行で "page001" のような
    誤った値が入っていることがあるため。ファイル名は常に正しい。
    """
    match = _DATE_IN_FILENAME.search(str(filename))
    raw = match.group(1) if match else str(date_field)
    return pd.to_datetime(raw, format="%Y%m%d%H", errors="coerce")


class WeatherMapDataset(Dataset):
    """labels.csv (filename,label,...) と画像ディレクトリからサンプルを返すマルチラベルDataset。

    label列はパイプ区切りで複数ラベルを持てる(例: "winter_pressure_pattern|japan_sea_low")。
    単一ラベルの行もそのまま扱える。
    """

    def __init__(self, images_dir: str, labels_csv: str, transform=None, years=None):
        """years に年のリストを渡すと、その年の画像だけを対象にする。

        「2023〜2025年の3年分」のように対象期間を区切って実験するときに使う。
        ここで絞り込んでおくことで、src/split.py が返す行番号と
        Subset のインデックスが常に一致する。
        """
        self.images_dir = Path(images_dir)
        self.transform = transform

        df = pd.read_csv(labels_csv)
        df["parsed_labels"] = df["label"].apply(parse_labels)
        df = df[df["parsed_labels"].apply(len) > 0].reset_index(drop=True)
        # 時間ブロック分割(src/split.py)が使う観測日時。
        df["parsed_datetime"] = [
            parse_datetime(fn, dt)
            for fn, dt in zip(df["filename"], df.get("date", [None] * len(df)))
        ]

        if years:
            years = {int(y) for y in years}
            before = len(df)
            df = df[df["parsed_datetime"].dt.year.isin(years)].reset_index(drop=True)
            print(f"対象年 {sorted(years)} に限定: {before}件 → {len(df)}件")
            if df.empty:
                raise ValueError(f"指定した年 {sorted(years)} に該当する画像がありません")

        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self.images_dir / row["filename"]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        target = torch.zeros(len(LABELS), dtype=torch.float32)
        for label in row["parsed_labels"]:
            target[LABEL_TO_INDEX[label]] = 1.0
        return image, target
