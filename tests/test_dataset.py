"""ラベルCSVと画像・ERA5特徴量の対応づけのテスト。

行がずれると、別の日の天気図に別の日のラベルを教えることになる。
学習は正常に進み、成績が少し下がるだけなので気づきにくい。
"""

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.dataset import WeatherMapDataset, parse_datetime
from src.labels import LABELS, parse_labels


@pytest.fixture
def workspace(tmp_path):
    """画像・ラベル・ERA5特徴量が揃った一式を作る。"""
    images = tmp_path / "images"
    images.mkdir()
    stamps = pd.date_range("2024-01-01", periods=8, freq="12h")
    names = [f"Js_{t:%Y%m%d%H}.png" for t in stamps]
    for name in names:
        Image.new("RGB", (32, 24)).save(images / name)

    labels = tmp_path / "labels.csv"
    pd.DataFrame({
        "filename": names,
        "label": ["typhoon", "okhotsk_high", "typhoon|okhotsk_high", "unclassified",
                  "typhoon", "typhoon", "typhoon", "typhoon"],
    }).to_csv(labels, index=False)

    features = tmp_path / "features.csv"
    # ファイル名だけを逆順にする。値は順番のまま置くので、
    # 位置で結合してしまうと値がずれて、テストが落ちる。
    pd.DataFrame({
        "filename": list(reversed(names)),
        "datetime": list(reversed(list(stamps))),
        "value": list(range(len(names))),
    }).to_csv(features, index=False)
    return {"images": images, "labels": labels, "features": features, "names": names}


def test_dates_come_from_the_filename_not_the_date_column():
    """date列には過去のバグで'page001'のような値が入った行がある。
    ファイル名は常に正しいので、そちらを優先する。"""
    assert parse_datetime("Js_2001070300_page001.jpg", "page001").year == 2001
    assert parse_datetime("Js_2025010112.png", None).hour == 12


def test_unclassified_rows_are_dropped(workspace):
    dataset = WeatherMapDataset(workspace["images"], workspace["labels"])
    assert len(dataset) == 7, "unclassifiedの行が残っている"


def test_features_are_matched_by_filename_not_by_row_order(workspace):
    """特徴量CSVの並びが違っても、ファイル名で正しく結合できること。"""
    dataset = WeatherMapDataset(
        workspace["images"], workspace["labels"], features_csv=workspace["features"]
    )
    for i in range(len(dataset)):
        name = dataset.df["filename"].iloc[i]
        expected = len(workspace["names"]) - 1 - workspace["names"].index(name)
        _, features, _ = dataset[i]
        assert features.item() == pytest.approx(expected)


def test_targets_match_the_label_column(workspace):
    dataset = WeatherMapDataset(workspace["images"], workspace["labels"])
    for i in range(len(dataset)):
        expected = parse_labels(dataset.df["label"].iloc[i])
        _, _, target = dataset[i]
        got = [LABELS[j] for j, on in enumerate(target.tolist()) if on]
        assert sorted(got) == sorted(expected)


def test_missing_images_are_reported_before_training_starts(workspace):
    """1エポック目の途中で落ちると、そこまでの計算が無駄になる。"""
    (workspace["images"] / workspace["names"][0]).unlink()
    with pytest.raises(FileNotFoundError, match="1件"):
        WeatherMapDataset(workspace["images"], workspace["labels"])


def test_a_labels_file_with_no_usable_rows_says_so(tmp_path):
    labels = tmp_path / "labels.csv"
    pd.DataFrame({"filename": ["Js_2024010100.png"], "label": ["obsolete"]}).to_csv(
        labels, index=False
    )
    with pytest.raises(ValueError, match="認識できるラベル"):
        WeatherMapDataset(tmp_path, labels)


def test_year_filter_keeps_rows_and_features_aligned(workspace):
    dataset = WeatherMapDataset(
        workspace["images"], workspace["labels"],
        features_csv=workspace["features"], years=[2024],
    )
    assert len(dataset) == len(dataset.features)
    assert (dataset.df["parsed_datetime"].dt.year == 2024).all()


def test_images_are_matched_by_timestamp_not_exact_filename(tmp_path):
    """同じ観測時刻を指していれば、表記が違っても突き合わせられること。

    気象庁から取ったものは Js_2025050100.png、国会図書館から取ったものは
    JS_2025050100_page001.jpg。厳密一致で照合すると、画像は手元にあるのに
    「見つからない」と言って学習が始まらない。
    """
    from PIL import Image

    from src.dataset import WeatherMapDataset

    images = tmp_path / "imgs"
    images.mkdir()
    stamps = ["2023010100", "2023010112"]
    for i, stamp in enumerate(stamps):
        prefix = "JS" if i else "Js"
        Image.new("RGB", (16, 16)).save(images / f"{prefix}_{stamp}_page001.jpg")

    labels = tmp_path / "labels.csv"
    pd.DataFrame({
        "filename": [f"Js_{s}.png" for s in stamps],
        "label": ["winter_pressure_pattern"] * len(stamps),
    }).to_csv(labels, index=False)

    dataset = WeatherMapDataset(images, labels)
    assert len(dataset) == len(stamps)
    # 実際に開けるパスが解決されている
    assert dataset.df["image_path"].iloc[0].exists()
