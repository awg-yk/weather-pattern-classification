"""CDS(Copernicus Climate Data Store)につながるかを、最小のリクエストで確かめる。

本番のダウンロードは数十分かかるため、設定の不備はここで出し切っておきたい。
1日・1時刻・狭い範囲だけを要求するので、成功すれば数十秒〜数分で終わる。

事前準備:
    1. https://cds.climate.copernicus.eu/ でアカウントを作る
    2. ログインした状態で右上の名前 → "Your profile" を開き、
       Personal Access Token を控える
    3. ~/.cdsapirc (Windowsなら C:\\Users\\<ユーザー名>\\.cdsapirc) を作る:

           url: https://cds.climate.copernicus.eu/api
           key: <Personal Access Token>

       ※ 2024年のCDS刷新で、URLは末尾が /api になり(以前は /api/v2)、
         keyも "UID:APIキー" 形式から トークンのみ に変わっている。
         古い形式の情報が検索で出てくるので注意。
    4. 使いたいデータセットのページを開き、Download → Terms of use に同意する
       (同意しないと、認証が通っていてもリクエストが弾かれる)

使い方:
    python -m scripts.check_era5_access
"""

import argparse
import sys
from pathlib import Path

# 動作確認に使う最小のリクエスト(日本付近の1日1時刻ぶんだけ)
CHECK_DATE = "2024-01-01"
CHECK_AREA = [46, 128, 42, 132]  # 北, 西, 南, 東


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/raw/era5/_access_check.nc")
    parser.add_argument(
        "--dataset",
        default="single-levels",
        choices=["single-levels", "pressure-levels"],
        help="single-levels=海面更正気圧 / pressure-levels=850hPa気温。両方試すこと",
    )
    args = parser.parse_args()

    try:
        import cdsapi
    except ImportError:
        raise SystemExit("cdsapiが入っていません。 pip install cdsapi を実行してください。")

    rc = Path.home() / ".cdsapirc"
    print(f"設定ファイル: {rc} ({'あり' if rc.exists() else 'なし'})")
    if rc.exists():
        for line in rc.read_text().splitlines():
            if line.startswith("url:"):
                url = line.split(":", 1)[1].strip()
                print(f"  url: {url}")
                if url.rstrip("/").endswith("/v2"):
                    print("  ⚠ 古いURLです。https://cds.climate.copernicus.eu/api に直してください。")
            elif line.startswith("key:"):
                key = line.split(":", 1)[1].strip()
                print(f"  key: {'*' * min(len(key), 12)} ({len(key)}文字)")
                if ":" in key:
                    print("  ⚠ 'UID:APIキー' 形式は旧仕様です。トークンだけを書いてください。")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.dataset == "single-levels":
        name = "reanalysis-era5-single-levels"
        request = {
            "product_type": ["reanalysis"],
            "variable": ["mean_sea_level_pressure"],
            "year": [CHECK_DATE[:4]],
            "month": [CHECK_DATE[5:7]],
            "day": [CHECK_DATE[8:10]],
            "time": ["00:00"],
            "area": CHECK_AREA,
            "data_format": "netcdf",
        }
    else:
        name = "reanalysis-era5-pressure-levels"
        request = {
            "product_type": ["reanalysis"],
            "variable": ["temperature"],
            "pressure_level": ["850"],
            "year": [CHECK_DATE[:4]],
            "month": [CHECK_DATE[5:7]],
            "day": [CHECK_DATE[8:10]],
            "time": ["00:00"],
            "area": CHECK_AREA,
            "data_format": "netcdf",
        }

    print(f"\n{name} に最小のリクエストを送ります…")
    try:
        cdsapi.Client().retrieve(name, request, str(out_path))
    except Exception as e:
        print(f"\n失敗しました: {type(e).__name__}: {e}\n", file=sys.stderr)
        raise SystemExit(_hint(e))

    size_kb = out_path.stat().st_size / 1024
    print(f"\n成功しました: {out_path} ({size_kb:.1f} KB)")

    try:
        import xarray as xr
    except ImportError:
        print("(xarrayが無いため中身の確認は省略します)")
        return
    ds = xr.open_dataset(out_path)
    print("\n--- 中身 ---")
    print(ds)


def _hint(error: Exception) -> str:
    """よくある失敗に対して、次に何をすればよいかを示す。"""
    text = str(error).lower()
    if "licence" in text or "license" in text or "required licences" in text:
        return (
            "データセットの利用条件に同意していません。\n"
            "  CDSのサイトでデータセットのページを開き、Download タブの一番下にある\n"
            "  Terms of use に同意してから、もう一度実行してください。"
        )
    if "401" in text or "403" in text or "authentication" in text or "authorization" in text:
        return (
            "認証に失敗しました。~/.cdsapirc の url と key を確認してください。\n"
            "  url: https://cds.climate.copernicus.eu/api\n"
            "  key: (Personal Access Token のみ。UID:キー 形式は旧仕様)"
        )
    if "certificate" in text or "ssl" in text:
        return (
            "SSL証明書の検証に失敗しました。大学のネットワークが通信を検査している場合に起きます。\n"
            "  pip install pip-system-certs を実行し、PowerShellを開き直してください。\n"
            "  (証明書の検証を無効にするのは、通信内容を保護できなくなるので避けてください)"
        )
    if "proxy" in text or "connection" in text or "timed out" in text:
        return (
            "接続できませんでした。プロキシの設定が必要かもしれません。\n"
            '  $env:HTTPS_PROXY = "http://10.0.0.1:8080"\n'
            '  $env:HTTP_PROXY  = "http://10.0.0.1:8080"\n'
            "  を設定してから、もう一度実行してください。"
        )
    return "上のエラー内容を貼ってください。"


if __name__ == "__main__":
    main()
