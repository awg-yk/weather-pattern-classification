"""ERA5の格子データから、気圧配置の判別に使う領域特徴を計算してCSVに書き出す。

天気図の画像だけでは、モデルは「どこに」高気圧・低気圧があるかを数値として
受け取れない。Grad-CAMでも、オホーツク海高気圧の判定でモデルが高気圧本体では
なく下流の等圧線を見ていることが確認できた。

そこで気象学的に意味のある領域ごとの平均気圧(と850hPa気温)を計算し、
「オホーツク海が周囲よりどれだけ高気圧か」のような量を直接与えられるようにする。
特徴量の一つ一つに気象学的な意味があるので、効いた理由を説明できる。

出力はラベルCSVと同じ filename 列を持つので、そのまま結合できる。

使い方:
    python -m scripts.era5_features --era5-dir data/raw/era5 --years 2023 2024 2025 \
        --out data/era5_features.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.era5 import open_era5

# 気圧配置の定義に対応する領域(南, 北, 西, 東)。
# ラベルごとに「そこが高いか低いか」が決め手になる場所を選んでいる。
REGIONS = {
    "okhotsk": (45, 60, 135, 160),      # オホーツク海高気圧
    "japan_sea": (35, 45, 130, 140),    # 日本海低気圧
    "south_coast": (25, 35, 130, 145),  # 南岸低気圧
    "continent": (45, 60, 110, 130),    # シベリア高気圧(西高東低の西)
    "pacific": (20, 35, 145, 165),      # 太平洋高気圧
    "japan": (30, 45, 128, 146),        # 日本付近(基準)
}


def _subset(da, lat_range, lon_range):
    """緯度が降順・昇順どちらで格納されていても切り出せるようにする。"""
    lat_name = "latitude" if "latitude" in da.coords else "lat"
    lon_name = "longitude" if "longitude" in da.coords else "lon"
    lat = da[lat_name].values
    lo, hi = lat_range
    lat_slice = slice(hi, lo) if lat[0] > lat[-1] else slice(lo, hi)
    return da.sel({lat_name: lat_slice, lon_name: slice(*lon_range)})


def _time_name(ds):
    for name in ("valid_time", "time"):
        if name in ds.coords:
            return name
    raise KeyError(f"時刻の座標が見つかりません: {list(ds.coords)}")


def mslp_features(ds) -> pd.DataFrame:
    """海面更正気圧から、領域ごとの気圧偏差と最強の高低気圧の位置を出す。"""
    msl = ds["msl"] / 100.0  # Pa -> hPa
    tname = _time_name(ds)
    lat_name = "latitude" if "latitude" in msl.coords else "lat"
    lon_name = "longitude" if "longitude" in msl.coords else "lon"

    # 領域全体の平均を基準にした偏差にする。季節による気圧全体の上下を打ち消し、
    # 「周囲と比べて高いか低いか」という気圧配置の情報だけを残すため。
    domain_mean = msl.mean(dim=[lat_name, lon_name])

    out = {"datetime": pd.to_datetime(msl[tname].values)}
    out["mslp_domain_mean"] = domain_mean.values
    for name, (lat0, lat1, lon0, lon1) in REGIONS.items():
        region = _subset(msl, (lat0, lat1), (lon0, lon1)).mean(dim=[lat_name, lon_name])
        out[f"mslp_{name}_anom"] = (region - domain_mean).values

    # 西高東低の指標。冬型では大陸が高く、日本の東の海上が低い。
    out["mslp_west_minus_east"] = out["mslp_continent_anom"] - out["mslp_pacific_anom"]

    # 領域内で最も低い/高い格子点の位置。低気圧・高気圧の中心がどこにあるかを直接表す。
    flat = msl.stack(pt=(lat_name, lon_name))
    for tag, idx in (("low", flat.argmin("pt")), ("high", flat.argmax("pt"))):
        picked = flat["pt"][idx.values]
        out[f"mslp_{tag}_lat"] = np.array([p[0] for p in picked.values], dtype="float32")
        out[f"mslp_{tag}_lon"] = np.array([p[1] for p in picked.values], dtype="float32")
    out["mslp_min"] = flat.min("pt").values
    out["mslp_max"] = flat.max("pt").values

    return pd.DataFrame(out)


def t850_features(ds) -> pd.DataFrame:
    """850hPa気温から、前線の判別に使う温度傾度を出す。

    前線は温度が水平方向に急変する場所なので、傾度の大きさがそのまま前線の
    強さの指標になる。日本付近での最大傾度と、その緯度を特徴量にする。
    """
    t = ds["t"] - 273.15  # K -> ℃
    if "pressure_level" in t.dims:
        t = t.isel(pressure_level=0)
    tname = _time_name(ds)
    lat_name = "latitude" if "latitude" in t.coords else "lat"
    lon_name = "longitude" if "longitude" in t.coords else "lon"

    japan = _subset(t, (REGIONS["japan"][0], REGIONS["japan"][1]),
                    (REGIONS["japan"][2], REGIONS["japan"][3]))
    # 緯度方向の温度差(南北の温度傾度)。前線帯で大きくなる。
    grad = japan.differentiate(lat_name).pipe(abs)

    out = {"datetime": pd.to_datetime(t[tname].values)}
    out["t850_japan_mean"] = japan.mean(dim=[lat_name, lon_name]).values
    out["t850_japan_max_grad"] = grad.max(dim=[lat_name, lon_name]).values
    # 傾度が最大になる緯度 = 前線がどのあたりにあるか
    out["t850_front_lat"] = grad.mean(dim=lon_name).idxmax(dim=lat_name).values.astype("float32")
    return pd.DataFrame(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--era5-dir", default="data/raw/era5")
    parser.add_argument("--years", type=int, nargs="+", required=True)
    parser.add_argument("--out", default="data/era5_features.csv")
    parser.add_argument(
        "--filename-format",
        default="Js_{:%Y%m%d%H}.png",
        help="ラベルCSVのfilename列に合わせる書式",
    )
    args = parser.parse_args()

    era5_dir = Path(args.era5_dir)
    frames = []
    for year in args.years:
        mslp_path = era5_dir / f"era5_mslp_{year}.nc"
        if not mslp_path.exists():
            raise SystemExit(
                f"{mslp_path} がありません。\n"
                f"  python -m scripts.download_era5 --years {year} を実行してください。"
            )
        with open_era5(mslp_path) as ds:
            frame = mslp_features(ds)
        print(f"{year}年 mslp: {len(frame)}時刻")

        t850_path = era5_dir / f"era5_t850_{year}.nc"
        if t850_path.exists():
            with open_era5(t850_path) as ds:
                t_frame = t850_features(ds)
            print(f"{year}年 t850: {len(t_frame)}時刻")
            frame = frame.merge(t_frame, on="datetime", how="left")
        else:
            print(f"{year}年 t850: ファイルがないため前線の特徴は省略します")
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True).sort_values("datetime").reset_index(drop=True)
    df.insert(0, "filename", [args.filename_format.format(t) for t in df["datetime"]])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\n書き出しました: {out_path}({len(df)}行 / 特徴量{len(df.columns) - 2}個)")
    print("\n--- 統計 ---")
    print(df.drop(columns=["filename", "datetime"]).describe().T[["mean", "std", "min", "max"]])


if __name__ == "__main__":
    main()
