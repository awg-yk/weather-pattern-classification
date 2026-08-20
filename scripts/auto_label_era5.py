"""
ERA5海面更正気圧(mslp)の格子データから、気圧配置の特徴量を計算し、
規則ベースで初期ラベル(気圧パターン)を付与する。

これはあくまで初期ラベルの下書きであり、精度を出すには人手での確認・修正が必須。
日本周辺(北緯20-50度, 東経120-150度)を対象とする想定。

使い方:
    python scripts/auto_label_era5.py --nc data/raw/era5/era5_mslp_2024-01-01_2024-01-31.nc \
        --out data/labels_era5_draft.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.era5 import open_era5

# 日本付近をおおまかに東西南北4象限に分けて平均気圧を比較する
WEST_LON = (120, 135)
EAST_LON = (135, 150)
NORTH_LAT = (35, 50)
SOUTH_LAT = (20, 35)


def region_mean(field, lon, lat, lon_range, lat_range):
    lon_mask = (lon >= lon_range[0]) & (lon <= lon_range[1])
    lat_mask = (lat >= lat_range[0]) & (lat <= lat_range[1])
    return float(field[np.ix_(lat_mask, lon_mask)].mean())


def classify(field, lon, lat) -> str:
    west = region_mean(field, lon, lat, WEST_LON, (NORTH_LAT[0], SOUTH_LAT[1]))
    east = region_mean(field, lon, lat, EAST_LON, (NORTH_LAT[0], SOUTH_LAT[1]))
    north = region_mean(field, lon, lat, (WEST_LON[0], EAST_LON[1]), NORTH_LAT)
    south = region_mean(field, lon, lat, (WEST_LON[0], EAST_LON[1]), SOUTH_LAT)

    gradient_ew = west - east
    gradient_ns = north - south
    field_range = float(field.max() - field.min())

    # 等圧線が非常に混んでいる(気圧差が大きい) -> 台風/発達した低気圧の可能性
    if field.min() < 990 and field_range > 40:
        return "typhoon"

    # 西高東低: 西で高圧・東で低圧、南北差より東西差が明瞭
    if gradient_ew > 6 and abs(gradient_ew) > abs(gradient_ns):
        return "winter_pressure_pattern"

    # 南高北低: 南で高圧・北で低圧(太平洋高気圧型)
    if gradient_ns < -4:
        return "pacific_high"

    # 気圧差が小さく穏やか -> 移動性高気圧 or 帯状高気圧の可能性(要人手判別)
    if field_range < 15:
        return "migratory_high"

    return "unclassified"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nc", required=True)
    parser.add_argument("--out", default="data/labels_era5_draft.csv")
    args = parser.parse_args()

    ds = open_era5(args.nc)
    mslp = ds["msl"] / 100.0  # Pa -> hPa
    lon = ds["longitude"].values
    lat = ds["latitude"].values
    time_dim = "valid_time" if "valid_time" in mslp.coords else "time"

    rows = []
    for t in mslp[time_dim].values:
        field = mslp.sel(**{time_dim: t}).values
        label = classify(field, lon, lat)
        ts = pd.to_datetime(t).strftime("%Y%m%d_%H%M")
        rows.append({"filename": f"{ts}.png", "label": label, "source": "era5_auto", "date": ts})
        print(f"{ts}: {label}")

    out_path = Path(args.out)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
