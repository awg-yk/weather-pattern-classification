from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.labels import LABEL_TO_INDEX, LABELS


def parse_labels(label_field: str) -> list:
    """"winter_pressure_pattern|japan_sea_low" のようなパイプ区切りを分解する。"""
    return [l for l in str(label_field).split("|") if l in LABEL_TO_INDEX]


class WeatherMapDataset(Dataset):
    """labels.csv (filename,label,...) と画像ディレクトリからサンプルを返すマルチラベルDataset。

    label列はパイプ区切りで複数ラベルを持てる(例: "winter_pressure_pattern|japan_sea_low")。
    単一ラベルの行もそのまま扱える。
    """

    def __init__(self, images_dir: str, labels_csv: str, transform=None):
        self.images_dir = Path(images_dir)
        self.transform = transform

        df = pd.read_csv(labels_csv)
        df["parsed_labels"] = df["label"].apply(parse_labels)
        df = df[df["parsed_labels"].apply(len) > 0].reset_index(drop=True)
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
