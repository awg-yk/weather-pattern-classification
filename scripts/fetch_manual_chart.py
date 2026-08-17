"""
2000年〜2022年9月分の天気図(国立国会図書館デジタルコレクションから手動で収集し、
JPEGに変換したもの)を、GitHubの `jpeg-data` ブランチから直接取得するヘルパー。

ファイルは data/raw/ndl_manual/jpg/{prefix}_{YYYYMMDDHH}_page001.jpg という名前で、
prefixは "Js" または "JS"(収集時期によって表記揺れがあるため両方試す)。
"""

from datetime import date
from pathlib import Path

import requests

RAW_BASE_URL = (
    "https://raw.githubusercontent.com/awg-yk/weather-pattern-classification/"
    "jpeg-data/data/raw/ndl_manual/jpg"
)

MANUAL_ARCHIVE_START_DATE = date(2000, 1, 1)

DEFAULT_CACHE_DIR = Path("data/raw/ndl_manual_fetch")

_PREFIX_CANDIDATES = ("Js", "JS")


def _candidate_urls(target_date: date, hour: int) -> list[str]:
    ts = target_date.strftime("%Y%m%d") + f"{hour:02d}"
    return [f"{RAW_BASE_URL}/{prefix}_{ts}_page001.jpg" for prefix in _PREFIX_CANDIDATES]


def manual_chart_exists(target_date: date, hour: int) -> bool:
    """ダウンロードはせず、その日付・時刻の天気図がGitHub上の手動アーカイブに
    存在するかだけ確認する。"""
    if hour not in (0, 12):
        return False
    for url in _candidate_urls(target_date, hour):
        try:
            resp = requests.head(url, timeout=15, allow_redirects=True)
            if resp.status_code == 200:
                return True
        except requests.RequestException:
            continue
    return False


def fetch_manual_chart(
    target_date: date, hour: int = 0, cache_dir: str = str(DEFAULT_CACHE_DIR)
) -> Path:
    """GitHub上の手動アーカイブ(jpeg-dataブランチ)から天気図JPEGを取得し、
    ローカルにキャッシュしてそのパスを返す。"""
    if hour not in (0, 12):
        raise ValueError("hourは0か12を指定してください")

    cache_dir_path = Path(cache_dir)
    cache_dir_path.mkdir(parents=True, exist_ok=True)

    ts = target_date.strftime("%Y%m%d") + f"{hour:02d}"
    local_path = cache_dir_path / f"Js_{ts}_page001.jpg"
    if local_path.exists():
        return local_path

    last_status = None
    for url in _candidate_urls(target_date, hour):
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            local_path.write_bytes(resp.content)
            return local_path
        last_status = resp.status_code

    raise FileNotFoundError(
        f"{target_date.isoformat()} {hour}Z の天気図は手動アーカイブに見つかりません"
        f"(最後のHTTPステータス: {last_status})。"
        f"{MANUAL_ARCHIVE_START_DATE.isoformat()}〜2022-09-30の範囲・0Zまたは12Zのみ対応しています。"
    )
