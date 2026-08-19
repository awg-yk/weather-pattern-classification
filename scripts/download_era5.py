"""ERA5再解析データをCopernicus Climate Data Store (CDS)から年ごとにダウンロードする。

天気図の画像だけでは、モデルは「その気圧配置がどこにあるか」を数値としては
受け取れない。ERA5の格子データを併用することで、海面更正気圧の分布と
850hPa気温(前線の判別に効く)を直接与えられるようにする。

取得する時刻は天気図に合わせて00Z/12Zの2回。日付はラベルの対象期間に合わせる。

事前準備と設定ファイルの書き方は scripts/check_era5_access.py を参照。
本番の前に必ずそちらで接続確認をしてから実行すること。

使い方:
    python -m scripts.download_era5 --years 2023 2024 2025
    python -m scripts.download_era5 --years 2023 2024 2025 --variable t850
"""

import argparse
from pathlib import Path

# 日本周辺。オホーツク海高気圧(おおむね北緯45〜60度・東経135〜165度)と
# 太平洋高気圧の中心が範囲外に出ないよう、天気図より広めに取る。
DEFAULT_AREA = [60, 110, 15, 165]  # 北, 西, 南, 東

# 天気図と同じ観測時刻。片方だけにすると画像とERA5の対応が取れなくなる。
TIMES = ["00:00", "12:00"]

VARIABLES = {
    "mslp": {
        "dataset": "reanalysis-era5-single-levels",
        "request": {"variable": ["mean_sea_level_pressure"]},
        "説明": "海面更正気圧(高気圧・低気圧の位置と強さ)",
    },
    "t850": {
        "dataset": "reanalysis-era5-pressure-levels",
        "request": {"variable": ["temperature"], "pressure_level": ["850"]},
        "説明": "850hPa気温(前線帯の判別に使う)",
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, nargs="+", required=True)
    parser.add_argument("--variable", default="mslp", choices=sorted(VARIABLES))
    parser.add_argument("--out", default="data/raw/era5")
    parser.add_argument(
        "--area",
        nargs=4,
        type=float,
        default=DEFAULT_AREA,
        metavar=("NORTH", "WEST", "SOUTH", "EAST"),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="既にあるファイルも取り直す(既定では飛ばすので、中断しても再実行できる)",
    )
    args = parser.parse_args()

    import cdsapi

    spec = VARIABLES[args.variable]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()

    print(f"変数: {args.variable}({spec['説明']})")
    print(f"範囲: 北{args.area[0]} 西{args.area[1]} 南{args.area[2]} 東{args.area[3]}")
    print(f"時刻: {', '.join(TIMES)}\n")

    for year in args.years:
        target = out_dir / f"era5_{args.variable}_{year}.nc"
        if target.exists() and not args.overwrite:
            print(f"{year}年: 既にあるので飛ばします({target})")
            continue

        # 途中で落ちたファイルを本物と誤認しないよう、完了してから名前を付け替える
        partial = target.with_suffix(".nc.part")
        request = {
            "product_type": ["reanalysis"],
            "year": [str(year)],
            "month": [f"{m:02d}" for m in range(1, 13)],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": TIMES,
            "area": args.area,
            "data_format": "netcdf",
            **spec["request"],
        }
        print(f"{year}年: 要求中… (CDSの待ち行列に入るため、数分〜数十分かかります)")
        client.retrieve(spec["dataset"], request, str(partial))
        partial.replace(target)
        print(f"{year}年: 保存しました {target} ({target.stat().st_size / 1e6:.1f} MB)\n")

    print("完了しました。")


if __name__ == "__main__":
    main()
