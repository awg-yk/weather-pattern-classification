"""
国立国会図書館デジタルコレクション(NDL)から、気象庁JSMAPアーカイブより古い
天気図(2000年〜2022年9月ごろ)を取得するヘルパー。

仕組み:
    1. NDLサーチのSRU検索API(https://ndlsearch.ndl.go.jp/api/sru)で、
       「天気図 {和暦}年{月}月　日本域地上天気図」というタイトルを検索し、
       その月の資料のpid(永続的識別子)を取得する。
    2. https://dl.ndl.go.jp/api/item/search/info:ndljp/pid/{pid} というJSON API
       (デジタルコレクションのビューアが内部で使っているAPI)から、その月に含まれる
       各日のPDFファイル情報を取得する(ファイル名は気象庁と同じ
       "JS_YYYYMMDDHH.pdf" 形式)。
    3. 該当日のPDFをダウンロードしてPNGに変換する。

NDLのデジタル化資料は1日1回(00Z)のみで、気象庁の最近のアーカイブのような
12Zは無いことを確認済み。

Colabのノートブックセルで以下のように使う:

    import sys
    sys.path.append("/content/weather-pattern-classification")
    from scripts.collect_ndl import fetch_ndl_chart

    image_path = fetch_ndl_chart(2000, 1, 1)
"""

import re
from pathlib import Path

import requests

from scripts.collect_jma import pdf_to_png

SRU_ENDPOINT = "https://ndlsearch.ndl.go.jp/api/sru"
ITEM_API_ENDPOINT = "https://dl.ndl.go.jp/api/item/search/info:ndljp/pid/{pid}"
DEFAULT_CACHE_DIR = Path("data/raw/ndl_fetch")


def to_japanese_era_title(year: int, month: int) -> str:
    """「天気図 {和暦}年{月}月　日本域地上天気図」というNDL上のタイトル文字列を作る。

    和暦の変わり目(1989年1月, 2019年5月)をまたぐ場合に注意。
    昭和・平成・令和のみ対応(それ以前は非対応)。
    """
    if (year, month) >= (2019, 5):
        era, era_start = "令和", 2018
    elif (year, month) >= (1989, 1):
        era, era_start = "平成", 1988
    elif (year, month) >= (1926, 12):
        era, era_start = "昭和", 1925
    else:
        raise ValueError(f"未対応の年です(昭和より前): {year}")

    era_year = year - era_start
    era_year_str = "元" if era_year == 1 else str(era_year)
    # "月"と"日本域"の間は全角スペース(NDL上の実際の表記に合わせる)
    return f"天気図 {era}{era_year_str}年{month}月　日本域地上天気図"


def resolve_ndl_pid(year: int, month: int) -> int:
    """指定した年月の巻号のpidをNDLサーチのSRU APIで検索する。"""
    title = to_japanese_era_title(year, month)
    params = {
        "operation": "searchRetrieve",
        "query": f'title="{title}"',
        "recordSchema": "dcndl",
    }
    resp = requests.get(SRU_ENDPOINT, params=params, timeout=30)
    resp.raise_for_status()

    match = re.search(r"info:ndljp/pid/(\d+)", resp.text)
    if not match:
        raise FileNotFoundError(f"NDLで該当月が見つかりません: {year}年{month}月 (検索タイトル: {title})")
    return int(match.group(1))


def get_ndl_day_urls(pid: int) -> dict:
    """該当pidのアイテムAPIから、"JS_YYYYMMDDHH" -> PDFの直リンクURL の対応表を作る。"""
    resp = requests.get(ITEM_API_ENDPOINT.format(pid=pid), timeout=30)
    resp.raise_for_status()
    data = resp.json()

    mapping = {}
    for bundle in data.get("item", {}).get("contentsBundles") or []:
        for content in bundle.get("contents") or []:
            sort_key = content.get("sortKey", "")
            m = re.match(r"(JS_\d{10})\.pdf", sort_key)
            path = content.get("publicPath") or content.get("path")
            if m and path:
                mapping[m.group(1)] = f"https://dl.ndl.go.jp/{path}"
    return mapping


def fetch_ndl_chart(year: int, month: int, day: int, hour: int = 0, cache_dir: str = str(DEFAULT_CACHE_DIR)) -> Path:
    """指定日の天気図をNDLから取得してPNGに変換し、そのパスを返す。"""
    cache_dir = Path(cache_dir)
    pdf_dir = cache_dir / "pdf"
    png_dir = cache_dir / "png"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    key = f"JS_{year:04d}{month:02d}{day:02d}{hour:02d}"
    png_path = png_dir / f"{key}.png"
    if png_path.exists():
        return png_path

    pid = resolve_ndl_pid(year, month)
    day_urls = get_ndl_day_urls(pid)
    if key not in day_urls:
        raise FileNotFoundError(
            f"{key} がNDLの資料(pid={pid})内に見つかりません。"
            "NDLのデジタル化資料は1日1回(00Z)のみなので、--hour 0 以外は存在しません。"
        )

    pdf_url = day_urls[key]
    pdf_path = pdf_dir / f"{key}.pdf"
    # PDF本体はヘッダーだけでは401になることがある。トップページに一度アクセスして
    # Cookie(セッション)を受け取ってから、同じセッションでPDFを取得する。
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        ),
        "Referer": f"https://dl.ndl.go.jp/pid/{pid}",
    }
    session = requests.Session()
    session.headers.update(headers)
    session.get("https://dl.ndl.go.jp/", timeout=30)  # Cookie取得のためのウォームアップ
    session.get(f"https://dl.ndl.go.jp/pid/{pid}", timeout=30)

    resp = session.get(pdf_url, timeout=30)
    resp.raise_for_status()
    pdf_path.write_bytes(resp.content)

    pdf_to_png(pdf_path, png_path)
    pdf_path.unlink(missing_ok=True)
    return png_path
