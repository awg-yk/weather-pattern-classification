"""
ERA5再解析データ（海面更正気圧など）をCopernicus Climate Data Store (CDS) からダウンロードする。

事前準備:
    1. https://cds.climate.copernicus.eu/ でアカウント作成
    2. APIキーを ~/.cdsapirc に設定
       url: https://cds.climate.copernicus.eu/api
       key: <UID>:<API_KEY>
    3. pip install cdsapi

使い方:
    python scripts/download_era5.py --start 2024-01-01 --end 2024-01-31 --out data/raw/era5
"""

import argparse
from pathlib import Path

import cdsapi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--out", default="data/raw/era5")
    parser.add_argument(
        "--area",
        nargs=4,
        type=float,
        default=[50, 120, 20, 150],
        metavar=("NORTH", "WEST", "SOUTH", "EAST"),
        help="日本周辺を含む緯度経度範囲 (デフォルト: 北50 西120 南20 東150)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    client = cdsapi.Client()
    target = out_dir / f"era5_mslp_{args.start}_{args.end}.nc"

    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": ["mean_sea_level_pressure"],
            "date": f"{args.start}/{args.end}",
            "time": ["00:00", "06:00", "12:00", "18:00"],
            "area": args.area,
            "format": "netcdf",
        },
        str(target),
    )
    print(f"saved: {target}")


if __name__ == "__main__":
    main()
