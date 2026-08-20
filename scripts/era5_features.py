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
    # オホーツク海は南北に分ける。高気圧が南寄りにあれば北日本に北東気流が
    # 入りやすく、北寄りだと入りにくい。ひとまとめに平均すると、この違いが
    # 消えてしまう(北緯58度の高気圧と47度の高気圧が同じ値になる)。
    "okhotsk_south": (45, 52, 140, 160),
    "okhotsk_north": (52, 60, 135, 160),
    "japan_sea": (35, 45, 130, 140),    # 日本海低気圧
    "south_coast": (25, 35, 130, 145),  # 南岸低気圧
    "continent": (45, 60, 110, 130),    # シベリア高気圧(西高東低の西)
    "pacific": (20, 35, 145, 165),      # 太平洋高気圧
    "japan": (30, 45, 128, 146),        # 日本付近(基準)
    "north_japan": (38, 46, 139, 147),  # 北日本(北東気流が入るかを見る場所)
}

# 高気圧の中心が南北どちらにあるかを見る範囲(オホーツク海全体)
OKHOTSK_BOX = (45, 60, 135, 160)


def _subset(da, lat_range, lon_range):
    """緯度が降順・昇順どちらで格納されていても切り出せるようにする。"""
    lat_name = "latitude" if "latitude" in da.coords else "lat"
    lon_name = "longitude" if "longitude" in da.coords else "lon"
    lat = da[lat_name].values
    lo, hi = lat_range
    lat_slice = slice(hi, lo) if lat[0] > lat[-1] else slice(lo, hi)
    return da.sel({lat_name: lat_slice, lon_name: slice(*lon_range)})


# 地衡風の計算に使う定数
OMEGA = 7.2921e-5      # 地球の自転角速度 [1/s]
AIR_DENSITY = 1.225    # 空気密度 [kg/m^3]
METERS_PER_DEGREE = 111_000.0


def geostrophic_wind_north_japan(msl, lat_name: str, lon_name: str) -> dict:
    """北日本の地衡風(等圧線に平行に吹く風)を気圧場から計算する。

    オホーツク海高気圧のラベルは「北東気流が北日本に入りそうな高気圧」という
    予報官の判断で付けられている。その判断に対応する量を直接計算する。

    地衡風は気圧傾度力とコリオリ力の釣り合いで決まり、等圧線に平行に吹く。
    北半球では高気圧の周りを時計回りに回るので、日本の北東に高気圧があれば
    北日本には北東からの風が入る。

        u = -(1/(f·ρ))·∂p/∂y   (東向き成分)
        v =  (1/(f·ρ))·∂p/∂x   (北向き成分)

    返すのは東西成分u・南北成分v・風速の3つ。北東成分のような合成量は作らない。
    合成の仕方を決め打ちすると情報が落ちるため(実際、北向き成分を混ぜると
    オホーツク海の南北による違いが消えてしまう)、成分のまま渡して重み付けは
    モデルに任せる。

    やませに相当するのは u が大きく負(東寄りの風)の状態。高気圧が
    オホーツク海の南部にあるときに強く現れ、北部にあるときは弱い。
    """
    lat0, lat1, lon0, lon1 = REGIONS["north_japan"]
    # 傾度を取るため、対象領域より少し広く切り出す
    field = _subset(msl, (lat0 - 3, lat1 + 3), (lon0 - 3, lon1 + 3)) * 100.0  # hPa -> Pa

    lat = field[lat_name]
    # 緯度・経度[度]を距離[m]に直してから微分する
    dp_dy = field.differentiate(lat_name) / METERS_PER_DEGREE
    dp_dx = field.differentiate(lon_name) / (METERS_PER_DEGREE * np.cos(np.deg2rad(lat)))
    coriolis = 2 * OMEGA * np.sin(np.deg2rad(lat))

    u = -dp_dy / (coriolis * AIR_DENSITY)
    v = dp_dx / (coriolis * AIR_DENSITY)

    u = _subset(u, (lat0, lat1), (lon0, lon1)).mean(dim=[lat_name, lon_name])
    v = _subset(v, (lat0, lat1), (lon0, lon1)).mean(dim=[lat_name, lon_name])
    return {
        "u": u.values.astype("float32"),
        "v": v.values.astype("float32"),
        "speed": np.hypot(u.values, v.values).astype("float32"),
    }


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

    # オホーツク海の高気圧が南寄りか北寄りか。正なら南寄り。
    out["mslp_okhotsk_south_minus_north"] = (
        out["mslp_okhotsk_south_anom"] - out["mslp_okhotsk_north_anom"]
    )

    # オホーツク海の範囲内で最も気圧が高い格子点の緯度 = 高気圧中心の南北位置
    okhotsk = _subset(msl, (OKHOTSK_BOX[0], OKHOTSK_BOX[1]), (OKHOTSK_BOX[2], OKHOTSK_BOX[3]))
    ok_flat = okhotsk.stack(pt=(lat_name, lon_name))
    ok_center = ok_flat["pt"][ok_flat.argmax("pt").values]
    out["mslp_okhotsk_center_lat"] = np.array([p[0] for p in ok_center.values], dtype="float32")
    out["mslp_okhotsk_center_lon"] = np.array([p[1] for p in ok_center.values], dtype="float32")
    out["mslp_okhotsk_center_value"] = ok_flat.max("pt").values

    # 北日本に北東気流が入っているか。風のデータを落とさなくても、地上気圧から
    # 地衡風(等圧線に平行に吹く風)として計算できる。
    wind = geostrophic_wind_north_japan(msl, lat_name, lon_name)
    out["wind_north_japan_u"] = wind["u"]   # 負なら東寄りの風(やませ)
    out["wind_north_japan_v"] = wind["v"]   # 負なら北寄りの風
    out["wind_north_japan_speed"] = wind["speed"]

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
