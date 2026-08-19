"""
ERA5の海面更正気圧(mslp)netCDFを、天気図に似た等圧線画像(PNG)に変換する。

使い方:
    python scripts/era5_to_image.py --nc data/raw/era5/era5_mslp_2024-01-01_2024-01-31.nc \
        --out data/processed/era5_images
"""

import argparse
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.era5 import open_era5

CONTOUR_INTERVAL_HPA = 4


def render_timestep(mslp_hpa: np.ndarray, lon, lat, out_path: Path):
    fig = plt.figure(figsize=(6, 6), dpi=150)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.5)

    levels = np.arange(
        np.floor(mslp_hpa.min() / CONTOUR_INTERVAL_HPA) * CONTOUR_INTERVAL_HPA,
        np.ceil(mslp_hpa.max() / CONTOUR_INTERVAL_HPA) * CONTOUR_INTERVAL_HPA,
        CONTOUR_INTERVAL_HPA,
    )
    cs = ax.contour(lon, lat, mslp_hpa, levels=levels, colors="black", linewidths=0.7)
    ax.clabel(cs, inline=True, fontsize=6, fmt="%d")
    ax.axis("off")

    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nc", required=True)
    parser.add_argument("--out", default="data/processed/era5_images")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = open_era5(args.nc)
    mslp = ds["msl"] / 100.0  # Pa -> hPa
    lon = ds["longitude"].values
    lat = ds["latitude"].values

    for t in mslp["valid_time"].values if "valid_time" in mslp.coords else mslp["time"].values:
        field = mslp.sel(**{mslp.dims[0]: t}).values
        ts = pd.to_datetime(t).strftime("%Y%m%d_%H%M")
        out_path = out_dir / f"{ts}.png"
        render_timestep(field, lon, lat, out_path)
        print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
