import re
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.labels import LABEL_TO_INDEX, LABELS, parse_labels  # noqa: F401  (従来の import 経路を保つ)

# ファイル名に含まれるYYYYMMDDHH。"Js_2001070300_page001.jpg" にも
# "Js_2025010100.png" にも対応する。
_DATE_IN_FILENAME = re.compile(r"(\d{10})")


def parse_datetime(filename: str, date_field=None) -> pd.Timestamp:
    """観測日時を取り出す。取れなければ NaT を返す。

    labels.csvのdate列ではなくファイル名を優先して見る。date列は、ファイル名から
    日付を抜き出すロジックを修正する前にラベル付けした行で "page001" のような
    誤った値が入っていることがあるため。ファイル名は常に正しい。
    """
    match = _DATE_IN_FILENAME.search(str(filename))
    raw = match.group(1) if match else str(date_field)
    return pd.to_datetime(raw, format="%Y%m%d%H", errors="coerce")


def index_images_by_stamp(images_dir: Path) -> dict:
    """画像ディレクトリを、ファイル名中のYYYYMMDDHH 10桁で引けるようにする。

    labels.csvのfilenameと実ファイル名は、同じ観測時刻を指していても表記が揃わない
    -- 気象庁から取ったものは Js_2025050100.png、国会図書館から取ったものは
    JS_2025050100_page001.jpg のように、接頭辞の大小・接尾辞・拡張子が違う。
    厳密一致で照合すると、画像は手元にあるのに「見つからない」と言って止まる。

    同じ時刻に複数の候補があれば、名前順で最初のものを使う(同じ天気図の別変換版で
    あることを想定している)。
    """
    index = {}
    for path in sorted(images_dir.rglob("*")):
        if not path.is_file():
            continue
        match = _DATE_IN_FILENAME.search(path.name)
        if match:
            index.setdefault(match.group(0), path)
    return index


class WeatherMapDataset(Dataset):
    """labels.csv (filename,label,...) と画像ディレクトリからサンプルを返すマルチラベルDataset。

    label列はパイプ区切りで複数ラベルを持てる(例: "winter_pressure_pattern|japan_sea_low")。
    単一ラベルの行もそのまま扱える。
    """

    def __init__(self, images_dir: str, labels_csv: str, transform=None, years=None,
                 features_csv=None):
        """years に年のリストを渡すと、その年の画像だけを対象にする。

        「2023〜2025年の3年分」のように対象期間を区切って実験するときに使う。
        ここで絞り込んでおくことで、src/split.py が返す行番号と
        Subset のインデックスが常に一致する。

        features_csv に scripts/era5_features.py の出力を渡すと、画像に加えて
        ERA5の数値特徴も返すようになる。特徴量が無い行は学習に使えないため除外する。
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

        self.df = df

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
        return image, features, target
