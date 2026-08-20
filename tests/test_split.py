"""データ分割のテスト。

ここが壊れると、学習した画像で評価してしまい、実力より高いスコアが出る。
しかも学習も評価も正常に完了するので、数字を見ても気づけない。
実際に最初の0.75はそれで出た値だった。止まるバグより危険なので、
分割については「重ならないこと」を機械的に確かめておく。
"""

import pandas as pd
import pytest

from src.split import make_splits, min_days_to_other_split

YEARS = (2023, 2024, 2025)


@pytest.fixture
def frame():
    """1日2回(00Z/12Z)、3年分の観測日時を持つDataFrame。"""
    stamps = pd.date_range("2023-01-01", "2025-12-31 12:00", freq="12h")
    return pd.DataFrame(
        {"filename": [f"Js_{t:%Y%m%d%H}.png" for t in stamps], "parsed_datetime": stamps}
    )


def _assert_disjoint(splits):
    train, val, test = (set(splits[k]) for k in ("train", "val", "test"))
    assert not train & val, "学習と検証が重複している"
    assert not train & test, "学習とテストが重複している"
    assert not val & test, "検証とテストが重複している"


@pytest.mark.parametrize("val_mode", ["tail", "spread"])
@pytest.mark.parametrize("test_year", YEARS)
def test_loyo_splits_are_disjoint(frame, val_mode, test_year):
    splits = make_splits(frame, mode="loyo", test_year=test_year, val_mode=val_mode)
    _assert_disjoint(splits)


@pytest.mark.parametrize("val_mode", ["tail", "spread"])
def test_loyo_test_set_is_exactly_the_held_out_year(frame, val_mode):
    splits = make_splits(frame, mode="loyo", test_year=2024, val_mode=val_mode)
    years = frame["parsed_datetime"].dt.year
    assert set(years.iloc[splits["test"]]) == {2024}
    assert 2024 not in set(years.iloc[splits["train"]])
    assert 2024 not in set(years.iloc[splits["val"]])


@pytest.mark.parametrize("val_mode", ["tail", "spread"])
def test_training_data_keeps_its_distance_from_the_test_year(frame, val_mode):
    gap = 3
    splits = make_splits(
        frame, mode="loyo", test_year=2024, gap_days=gap, val_mode=val_mode
    )
    distance = min_days_to_other_split(frame, splits["train"], splits["test"])
    assert distance.min() > gap, "テスト年の直前直後の日が学習に残っている"


def test_spread_validation_covers_every_month(frame):
    """tailは学習期間の末尾しか取らないので、必ず秋冬に偏る。

    夏に集中するラベル(オホーツク海高気圧・太平洋高気圧型)は検証データに
    ほとんど現れず、閾値もモデル選択もその季節を見ないまま決まってしまう。
    """
    spread = make_splits(frame, mode="loyo", test_year=2023, val_mode="spread")
    months = set(frame["parsed_datetime"].iloc[spread["val"]].dt.month)
    assert months == set(range(1, 13)), f"検証データに欠けている月がある: {sorted(months)}"

    tail = make_splits(frame, mode="loyo", test_year=2023, val_mode="tail")
    tail_months = set(frame["parsed_datetime"].iloc[tail["val"]].dt.month)
    assert len(tail_months) < 12, "tailが通年になっている(前提が変わった)"


def test_spread_validation_is_separated_from_training(frame):
    """抜いた週の前後を学習から除いていないと、隣接日で実質的に学習してしまう。"""
    gap = 3
    splits = make_splits(
        frame, mode="loyo", test_year=2023, gap_days=gap, val_mode="spread"
    )
    distance = min_days_to_other_split(frame, splits["train"], splits["val"])
    assert distance.min() > gap


def test_same_day_charts_stay_together(frame):
    """同じ日の00Zと12Zはほぼ同じ絵なので、別々の分割に入ってはいけない。"""
    splits = make_splits(frame, mode="temporal", val_ratio=0.2, test_ratio=0.2)
    days = frame["parsed_datetime"].dt.normalize()
    seen = {}
    for name in ("train", "val", "test"):
        for row in splits[name]:
            day = days.iloc[row]
            assert seen.setdefault(day, name) == name, f"{day:%Y-%m-%d} が分割をまたいでいる"


def test_every_row_is_used_exactly_once(frame):
    splits = make_splits(frame, mode="temporal", val_ratio=0.2, test_ratio=0.2)
    assigned = [row for name in ("train", "val", "test") for row in splits[name]]
    assert sorted(assigned) == list(range(len(frame)))


def test_by_year_gives_each_split_whole_years(frame):
    splits = make_splits(frame, mode="by_year", val_ratio=0.34, test_ratio=0.33)
    years = frame["parsed_datetime"].dt.year
    groups = [set(years.iloc[splits[name]]) for name in ("train", "val", "test")]
    assert all(groups), "空の分割がある"
    assert not groups[0] & groups[1] and not groups[1] & groups[2]


def test_loyo_rejects_a_year_with_no_data(frame):
    with pytest.raises(ValueError, match="2019"):
        make_splits(frame, mode="loyo", test_year=2019)


def test_missing_dates_are_reported_not_ignored():
    """日付が取れない行を黙って捨てると、分割が意図と違うものになる。"""
    df = pd.DataFrame({"filename": ["broken.png"], "parsed_datetime": [pd.NaT]})
    with pytest.raises(ValueError, match="日付"):
        make_splits(df, mode="temporal")
