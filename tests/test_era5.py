"""ERA5の特徴量計算と較正のテスト。

領域の切り出しは緯度の並び順(ERA5は降順)に依存するので、
向きを間違えると空の領域を平均してしまう。値が出てしまうため気づきにくい。
"""

import numpy as np
import pytest
import torch
import xarray as xr

from scripts.calibrate import expected_calibration_error, fit_temperature, reliability
from scripts.era5_features import REGIONS, _subset, geostrophic_wind_north_japan


def _pressure_field(center_lat, center_lon, ascending=False, amplitude=25.0, width=8.0):
    """指定位置に高気圧を置いた気圧場を作る。"""
    lat = np.arange(15, 60.1, 0.25) if ascending else np.arange(60, 14.9, -0.25)
    lon = np.arange(110, 165.1, 0.25)
    grid_lat, grid_lon = np.meshgrid(lat, lon, indexing="ij")
    field = 1013.0 + amplitude * np.exp(
        -(((grid_lat - center_lat) / width) ** 2 + ((grid_lon - center_lon) / (width * 1.5)) ** 2)
    )
    return xr.DataArray(
        field[None, ...],
        dims=("valid_time", "latitude", "longitude"),
        coords={"valid_time": [np.datetime64("2024-06-01T00")], "latitude": lat, "longitude": lon},
    )


@pytest.mark.parametrize("ascending", [False, True])
def test_region_subset_works_for_either_latitude_order(ascending):
    field = _pressure_field(50, 150, ascending=ascending)
    lat0, lat1, lon0, lon1 = REGIONS["okhotsk_south"]
    region = _subset(field, (lat0, lat1), (lon0, lon1))
    assert region.sizes["latitude"] > 0, "緯度の向きを取り違えて空になっている"
    assert float(region.latitude.min()) >= lat0
    assert float(region.latitude.max()) <= lat1


def test_a_high_in_the_south_of_the_okhotsk_sea_drives_a_stronger_easterly():
    """ラベルの定義そのもの。南寄りの高気圧ほど北日本に東寄りの風が入る。

    u が負 = 東寄りの風(やませ)。南部の高気圧で強く、北部では弱い。
    """
    south = geostrophic_wind_north_japan(_pressure_field(48, 150), "latitude", "longitude")
    middle = geostrophic_wind_north_japan(_pressure_field(52, 149), "latitude", "longitude")
    north = geostrophic_wind_north_japan(_pressure_field(57, 148), "latitude", "longitude")

    assert south["u"][0] < middle["u"][0] < north["u"][0] < 0
    assert abs(south["u"][0]) > 3 * abs(north["u"][0])


def test_wind_speed_is_the_magnitude_of_its_components():
    wind = geostrophic_wind_north_japan(_pressure_field(48, 150), "latitude", "longitude")
    assert wind["speed"][0] == pytest.approx(np.hypot(wind["u"][0], wind["v"][0]), rel=1e-5)


def test_temperature_scaling_recovers_a_known_distortion():
    """確信度を3倍に膨らませたモデルから、その3倍を推定できるか。"""
    torch.manual_seed(0)
    logits = torch.randn(4000, 10) * 1.2
    targets = torch.bernoulli(torch.sigmoid(logits))
    temperature = fit_temperature(logits * 3.0, targets)
    assert 2.5 < temperature < 3.5


def test_calibration_changes_the_confidence_but_not_the_prediction():
    """較正はF1などの成績を変えない。目盛りを直すだけ、と説明できる根拠。"""
    torch.manual_seed(0)
    logits = torch.randn(2000, 10) * 2.0
    before = (torch.sigmoid(logits) > 0.5)
    after = (torch.sigmoid(logits / 1.7) > 0.5)
    assert torch.equal(before, after)


def test_calibration_reduces_the_error_it_is_meant_to_reduce():
    torch.manual_seed(0)
    logits = torch.randn(4000, 10) * 1.2
    targets = torch.bernoulli(torch.sigmoid(logits))
    inflated = logits * 3.0
    temperature = fit_temperature(inflated, targets)

    truth = targets.numpy().ravel()
    before = expected_calibration_error(
        reliability(torch.sigmoid(inflated).numpy().ravel(), truth), len(truth)
    )
    after = expected_calibration_error(
        reliability(torch.sigmoid(inflated / temperature).numpy().ravel(), truth), len(truth)
    )
    assert after < before / 2
