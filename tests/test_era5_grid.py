"""ERA5の生格子をCNNへ直接入力するデータセットのテスト。

「天気図は前線を除けば気圧場の可視化にすぎない」という仮説を検証するために
作った経路なので、天気図の場合と同じ危険(行のずれ・情報の漏洩)を
別の形で持ち込んでいないかを確かめる。
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.era5_grid import NUM_CHANNELS, ERA5GridDataset, compute_grid_stats


@pytest.fixture
def workspace(tmp_path):
    era5_dir = tmp_path / "era5"
    era5_dir.mkdir()
    lat = np.arange(60, 14.9, -5.0)   # ERA5と同じ降順。テストなので粗くて良い
    lon = np.arange(110, 165.1, 5.0)
    rng = np.random.default_rng(0)

    names = []
    rows = []
    for year in (2023, 2024):
        times = pd.date_range(f"{year}-01-01", periods=6, freq="12h")
        mslp = 1013 + rng.normal(0, 8, (len(times), len(lat), len(lon))).astype("float32")
        t850 = 270 + rng.normal(0, 5, (len(times), len(lat), len(lon))).astype("float32")
        xr.Dataset(
            {"msl": (("valid_time", "latitude", "longitude"), mslp * 100)},
            coords={"valid_time": times, "latitude": lat, "longitude": lon},
        ).to_netcdf(era5_dir / f"era5_mslp_{year}.nc")
        xr.Dataset(
            {"t": (("valid_time", "latitude", "longitude"), t850)},
            coords={"valid_time": times, "latitude": lat, "longitude": lon},
        ).to_netcdf(era5_dir / f"era5_t850_{year}.nc")
        for i, t in enumerate(times):
            name = f"Js_{t:%Y%m%d%H}.png"
            names.append(name)
            rows.append({"filename": name, "label": "typhoon" if i else "unclassified"})

    labels_csv = tmp_path / "labels.csv"
    pd.DataFrame(rows).to_csv(labels_csv, index=False)
    return {"era5_dir": era5_dir, "labels_csv": labels_csv, "names": names}


def test_unclassified_rows_are_dropped(workspace):
    ds = ERA5GridDataset(workspace["labels_csv"], workspace["era5_dir"], grid_size=16)
    assert len(ds) == 10, "unclassifiedの行(年あたり1件)が残っている"


def test_reading_before_set_stats_is_refused(workspace):
    ds = ERA5GridDataset(workspace["labels_csv"], workspace["era5_dir"], grid_size=16)
    with pytest.raises(RuntimeError, match="set_stats"):
        ds[0]


def test_grid_has_two_channels_and_the_requested_size(workspace):
    ds = ERA5GridDataset(workspace["labels_csv"], workspace["era5_dir"], grid_size=20)
    mean, std = compute_grid_stats(ds, list(range(len(ds))))
    ds.set_stats(mean, std)
    grid, features, target, aux = ds[0]
    assert grid.shape == (NUM_CHANNELS, 20, 20)
    assert features.numel() == 0, "26特徴量のFeatureFusion経路とは無関係のはず"
    assert target.shape == (10,)


def test_stats_from_a_subset_differ_from_the_full_set(workspace):
    """検証・テストを含めた統計で正規化すると情報が漏れる。学習用サブセットだけの
    統計と全体の統計が異なることを確かめておけば、取り違えたときにすぐ気づける。"""
    ds = ERA5GridDataset(workspace["labels_csv"], workspace["era5_dir"], grid_size=16)
    full_mean, _ = compute_grid_stats(ds, list(range(len(ds))))
    subset_mean, _ = compute_grid_stats(ds, [0, 1, 2])
    assert not np.allclose(full_mean, subset_mean)


def test_normalisation_centres_the_data_near_zero(workspace):
    ds = ERA5GridDataset(workspace["labels_csv"], workspace["era5_dir"], grid_size=16)
    mean, std = compute_grid_stats(ds, list(range(len(ds))))
    ds.set_stats(mean, std)
    all_values = np.stack([ds[i][0].numpy() for i in range(len(ds))])
    assert abs(all_values.mean()) < 0.5


def test_missing_year_reports_the_download_command(tmp_path):
    labels_csv = tmp_path / "labels.csv"
    pd.DataFrame({"filename": ["Js_2030010100.png"], "label": ["typhoon"]}).to_csv(
        labels_csv, index=False
    )
    with pytest.raises(SystemExit, match="download_era5"):
        ERA5GridDataset(labels_csv, tmp_path / "era5", grid_size=16)


def test_missing_timestamp_in_an_existing_file_is_reported(workspace, tmp_path):
    """ファイルはあるが、その時刻のデータが無い場合(ダウンロード範囲の穴)。"""
    labels_csv = tmp_path / "labels_extra.csv"
    rows = pd.read_csv(workspace["labels_csv"])
    rows = pd.concat([
        rows,
        pd.DataFrame([{"filename": "Js_2023013100.png", "label": "typhoon"}]),
    ])
    rows.to_csv(labels_csv, index=False)
    with pytest.raises(ValueError, match="見つかりません"):
        ERA5GridDataset(labels_csv, workspace["era5_dir"], grid_size=16)
