from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from src.labels import LABEL_TO_INDEX


class WeatherMapDataset(Dataset):
    """labels.csv (filename,label,...) と画像ディレクトリからサンプルを返すDataset。"""

    def __init__(self, images_dir: str, labels_csv: str, transform=None):
        self.images_dir = Path(images_dir)
        self.transform = transform

        df = pd.read_csv(labels_csv)
        df = df[df["label"].isin(LABEL_TO_INDEX.keys())].reset_index(drop=True)
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self.images_dir / row["filename"]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label_idx = LABEL_TO_INDEX[row["label"]]
        return image, label_idx
