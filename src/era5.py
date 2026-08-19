"""ERA5のnetCDFを、パスに日本語が含まれていても開けるようにする共通処理。

netCDF4はC言語のライブラリを内部で使っており、Windowsでは非ASCIIのパスを
開けない(ファイルは存在するのに FileNotFoundError になる)。
本プロジェクトの置き場所が「02 自分の研究」のように日本語を含む場合に必ず踏む。

そこでパスを渡す方式が失敗したときは、Python側で開いたファイルオブジェクトを
h5netcdfに渡す方式に切り替える。この経路ではライブラリがパス文字列を扱わないため、
フォルダ名に何が入っていても影響を受けない。
"""

from pathlib import Path

import xarray as xr


def open_era5(path) -> xr.Dataset:
    """ERA5のnetCDFを開く。パスに日本語が含まれていても動く。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} がありません")

    try:
        return xr.open_dataset(path)
    except (FileNotFoundError, OSError) as first_error:
        # パスをライブラリに渡さない経路で開き直す
        try:
            handle = path.open("rb")
        except OSError:
            raise first_error
        try:
            ds = xr.open_dataset(handle, engine="h5netcdf")
        except Exception:
            handle.close()
            raise SystemExit(
                f"{path} を開けませんでした。\n"
                f"  最初の試行: {type(first_error).__name__}: {first_error}\n"
                "  パスに日本語が含まれる場合に起きることがあります。\n"
                "  pip install h5netcdf h5py を実行しても直らないときは、\n"
                "  プロジェクトをASCII文字だけのフォルダに移してください。"
            )
        # xarrayが閉じられるときに、こちらで開いたファイルも閉じる
        ds.set_close(handle.close)
        return ds
