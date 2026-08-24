"""ERA5の格子を、天気図と同じ画像としてCNNに入力するためにPNGへ描画する。

なぜこれが要るか
----------------
「天気図画像 0.636 対 ERA5格子 0.503」という差には2つの要因が混ざっている。

1. 情報量の差 : 天気図には前線・高低気圧の記号など、人間が解析した結果が
   描き込まれている。ERA5の気圧・気温の格子にはそれが無い。
2. 表現形式の差 : 天気図はEfficientNet＋ImageNet事前学習の経路を通るが、
   ERA5格子は2チャンネルの数値テンソルで、事前学習が効かない。

この2つを分離していない。ERA5を等圧線画像として描き、天気図とまったく同じ
経路(EfficientNet・224×224・同じ前処理)に通せば要因2が揃うので、そこに残る
差が要因1、つまり純粋な情報量の差になる。

描画の約束
----------
* ファイル名は ``Js_YYYYMMDDHH.png``。src/dataset.py の index_images_by_stamp が
  10桁の数字で画像を探すので、この形でないと1枚も見つからない。
* 等圧線の高さは全画像で固定(``CONTOUR_LEVELS``)。画像ごとに min/max から
  決めると、同じ位置の線が画像ごとに違う気圧を意味することになり、
  「線の本数が多い＝気圧傾度が大きい」という手がかりが壊れる。
* 画布の大きさも固定する。bbox_inches="tight" は余白を詰めるため、時刻ごとに
  画素数が変わり、緯度経度と画素位置の対応がずれる。

使い方:
    python -m scripts.era5_to_image --era5-dir data/raw/era5 --years 2023 2024 2025 \
        --out data/processed/era5_images
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.era5 import open_era5  # noqa: E402

CONTOUR_INTERVAL_HPA = 4
# 地上気圧の現実的な範囲。全画像で共通の高さを使う(上の「描画の約束」を参照)。
CONTOUR_LEVELS = np.arange(940, 1064, CONTOUR_INTERVAL_HPA)
# 850hPa気温の陰影。格子入力の2チャンネル目と同じ情報を画像にも載せるため。
T850_LEVELS = np.arange(-40, 41, 4)


def _time_name(da) -> str:
    for name in ("valid_time", "time"):
        if name in da.coords:
            return name
    raise KeyError(f"時刻の座標が見つかりません: {list(da.coords)}")


def _ascending(field: np.ndarray, lat: np.ndarray):
    """緯度が北→南の順に入っている場合に、南→北へ並べ替える。

    ERA5の latitude は降順で入っている。そのまま描くと北が下になり、
    天気図と上下が逆の画像ができる。
    """
    if lat[0] > lat[-1]:
        return field[..., ::-1, :], lat[::-1]
    return field, lat


def render_timestep(
    mslp_hpa: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    out_path: Path,
    t850_c: np.ndarray = None,
    size: int = 448,
    dpi: int = 100,
    coastlines: bool = False,
) -> None:
    """1時刻分を、天気図に似た等圧線画像として size×size 画素のPNGに描く。"""
    mslp_hpa, lat_up = _ascending(np.asarray(mslp_hpa), np.asarray(lat))

    inches = size / dpi
    fig = plt.figure(figsize=(inches, inches), dpi=dpi)
    if coastlines:
        import cartopy.crs as ccrs

        ax = fig.add_axes([0, 0, 1, 1], projection=ccrs.PlateCarree())
        ax.coastlines(linewidth=0.4, color="0.5")
    else:
        ax = fig.add_axes([0, 0, 1, 1])

    ax.set_facecolor("white")
    if t850_c is not None:
        t850_up, _ = _ascending(np.asarray(t850_c), np.asarray(lat))
        ax.contourf(lon, lat_up, t850_up, levels=T850_LEVELS, cmap="coolwarm", extend="both", alpha=0.55)
    ax.contour(lon, lat_up, mslp_hpa, levels=CONTOUR_LEVELS, colors="black", linewidths=0.7)

    ax.set_xlim(float(np.min(lon)), float(np.max(lon)))
    ax.set_ylim(float(np.min(lat_up)), float(np.max(lat_up)))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)  # bbox_inches は指定しない(画素数を固定するため)
    plt.close(fig)


def render_year(
    era5_dir: Path,
    year: int,
    out_dir: Path,
    with_t850: bool = True,
    size: int = 448,
    coastlines: bool = False,
    overwrite: bool = False,
) -> int:
    """1年分のncを読んで、時刻ごとにPNGを書き出す。書いた枚数を返す。"""
    mslp_path = era5_dir / f"era5_mslp_{year}.nc"
    if not mslp_path.exists():
        raise SystemExit(
            f"{mslp_path} がありません。\n"
            f"  python -m scripts.download_era5 --years {year} を実行してください。"
        )
    t850_path = era5_dir / f"era5_t850_{year}.nc"
    if with_t850 and not t850_path.exists():
        raise SystemExit(
            f"{t850_path} がありません。\n"
            f"  python -m scripts.download_era5 --years {year} --variable t850 を実行するか、\n"
            f"  --no-t850 を付けて気圧だけで描いてください。"
        )

    with open_era5(mslp_path) as ds:
        mslp = (ds["msl"] / 100.0).load()
        lon = ds["longitude"].values
        lat = ds["latitude"].values
    t850 = None
    if with_t850:
        with open_era5(t850_path) as ds:
            t850 = (ds["t"] - 273.15).load()
            if "pressure_level" in t850.dims:
                t850 = t850.isel(pressure_level=0)

    tname = _time_name(mslp)
    t850_index = {}
    if t850 is not None:
        t_tname = _time_name(t850)
        t850_index = {pd.Timestamp(v): i for i, v in enumerate(t850[t_tname].values)}

    written = 0
    for i, value in enumerate(mslp[tname].values):
        stamp = pd.Timestamp(value)
        out_path = out_dir / f"Js_{stamp.strftime('%Y%m%d%H')}.png"
        if out_path.exists() and not overwrite:
            continue
        field = mslp.isel({tname: i}).values
        temp = None
        if t850 is not None:
            if stamp not in t850_index:
                raise SystemExit(
                    f"{stamp} の850hPa気温がありません。ダウンロード範囲を確認するか、"
                    "--no-t850 を付けてください。"
                )
            temp = t850.isel({t_tname: t850_index[stamp]}).values
        render_timestep(
            field, lon, lat, out_path, t850_c=temp, size=size, coastlines=coastlines
        )
        written += 1
    return written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--era5-dir", default="data/raw/era5")
    parser.add_argument("--years", nargs="+", type=int, required=True)
    parser.add_argument("--out", default="data/processed/era5_images")
    parser.add_argument("--size", type=int, default=448, help="出力の一辺の画素数")
    parser.add_argument(
        "--no-t850", dest="with_t850", action="store_false",
        help="850hPa気温の陰影を描かず、等圧線だけにする",
    )
    parser.add_argument(
        "--coastlines", action="store_true",
        help="海岸線を描く(cartopyが必要。初回は地図データの取得が走る)",
    )
    parser.add_argument("--overwrite", action="store_true", help="既にあるPNGも描き直す")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for year in args.years:
        written = render_year(
            Path(args.era5_dir), year, out_dir,
            with_t850=args.with_t850, size=args.size,
            coastlines=args.coastlines, overwrite=args.overwrite,
        )
        print(f"{year}年: {written}枚を書き出しました")
        total += written
    print(f"合計 {total}枚 → {out_dir}")
    if total == 0:
        print("(すべて既にありました。描き直すには --overwrite を付けてください)")


if __name__ == "__main__":
    main()
