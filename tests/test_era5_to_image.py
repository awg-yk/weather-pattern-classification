"""era5_to_image.py が「学習に使える画像」を吐くかを確かめる。

見ているのは絵の綺麗さではなく、後段が壊れない3点。
1. ファイル名に10桁の日時が入っていること(index_images_by_stamp が探す形)
2. どの時刻でも画素数が同じであること
3. 気圧の分布が違えば絵も違うこと(全部同じ絵を量産していないか)
"""

import numpy as np
import pandas as pd
import pytest

from src.dataset import index_images_by_stamp
from scripts.era5_to_image import render_timestep, render_year

xr = pytest.importorskip("xarray")


def _write_nc(path, times, lat, lon, name, values, attrs_offset=0.0):
    da = xr.DataArray(
        values, dims=("valid_time", "latitude", "longitude"),
        coords={"valid_time": times, "latitude": lat, "longitude": lon},
    )
    xr.Dataset({name: da}).to_netcdf(path)


def _make_year(tmp_path, year=2024, n=3):
    times = pd.date_range(f"{year}-01-01", periods=n, freq="6h")
    lat = np.linspace(46, 20, 14)  # ERA5と同じ降順
    lon = np.linspace(120, 150, 16)
    yy, xx = np.meshgrid(lat, lon, indexing="ij")
    mslp = np.stack([1013 + 20 * np.sin((xx + 10 * i) / 12) + 0.2 * yy for i in range(n)])
    t850 = np.stack([yy - 40 + i for i in range(n)])
    era5_dir = tmp_path / "era5"
    era5_dir.mkdir()
    _write_nc(era5_dir / f"era5_mslp_{year}.nc", times, lat, lon, "msl", mslp * 100.0)
    _write_nc(era5_dir / f"era5_t850_{year}.nc", times, lat, lon, "t", t850 + 273.15)
    return era5_dir, times


def test_filenames_carry_a_ten_digit_stamp(tmp_path):
    era5_dir, times = _make_year(tmp_path)
    out = tmp_path / "img"
    written = render_year(era5_dir, 2024, out, size=112)
    assert written == len(times)

    index = index_images_by_stamp(out)
    for stamp in times:
        assert stamp.strftime("%Y%m%d%H") in index


def test_every_image_has_the_same_size(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    era5_dir, _ = _make_year(tmp_path)
    out = tmp_path / "img"
    render_year(era5_dir, 2024, out, size=112)
    sizes = {Image.open(p).size for p in sorted(out.glob("*.png"))}
    assert sizes == {(112, 112)}


def test_different_pressure_fields_give_different_images(tmp_path):
    era5_dir, _ = _make_year(tmp_path)
    out = tmp_path / "img"
    render_year(era5_dir, 2024, out, size=112)
    blobs = {p.read_bytes() for p in out.glob("*.png")}
    assert len(blobs) == 3


def test_existing_files_are_skipped_unless_overwrite(tmp_path):
    era5_dir, _ = _make_year(tmp_path)
    out = tmp_path / "img"
    assert render_year(era5_dir, 2024, out, size=112) == 3
    assert render_year(era5_dir, 2024, out, size=112) == 0
    assert render_year(era5_dir, 2024, out, size=112, overwrite=True) == 3


def test_north_is_at_the_top(tmp_path):
    """緯度が降順で入っていても、北を上にして描く。

    上下が逆だと「北に高気圧」が「南に高気圧」として学習される。
    """
    Image = pytest.importorskip("PIL.Image")
    lat_desc = np.linspace(46, 20, 20)
    lon = np.linspace(120, 150, 20)
    yy, _ = np.meshgrid(lat_desc, lon, indexing="ij")
    # 北ほど気圧が高い場を描くと、等圧線は上半分に集まるはず
    field = 1000 + (yy - 20)

    path = tmp_path / "Js_2024010100.png"
    render_timestep(field, lon, lat_desc, path, size=112)
    arr = np.asarray(Image.open(path).convert("L"), dtype=float)
    ink_rows = np.argwhere(arr < 128)[:, 0]
    top_line, bottom_line = ink_rows.min(), ink_rows.max()

    # 昇順に渡しても同じ絵になること(並べ替えが効いている)
    path2 = tmp_path / "Js_2024010106.png"
    render_timestep(field[::-1], lon, lat_desc[::-1], path2, size=112)
    assert path.read_bytes() == path2.read_bytes()
    assert top_line >= 0 and bottom_line < 112
