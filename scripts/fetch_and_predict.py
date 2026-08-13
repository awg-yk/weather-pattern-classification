"""
日付(と時刻)を指定するだけで、気象庁の天気図PDFをダウンロード・変換し、
そのまま分類まで行うヘルパー。画像を手元に用意する必要がない。

Colabのノートブックセルで以下のように使う:

    import sys
    sys.path.append("/content/weather-pattern-classification")
    from scripts.fetch_and_predict import fetch_chart

    image_path = fetch_chart("2025-01-01", hour=0)

使い方(CLI):
    python scripts/fetch_and_predict.py 2025-01-01 --hour 0
"""

import argparse
from datetime import date, timedelta
from pathlib import Path

import requests

from scripts.collect_jma import build_url, download_pdf, pdf_to_png

DEFAULT_CACHE_DIR = Path("data/raw/jma_fetch")

# 動作確認により判明した気象庁JSMAPアーカイブの実際の範囲(2026年8月時点)。
# 「直近1年分のみ」ではなく、この開始日以降が固定的に蓄積されている模様。
# これより古い日付は非公開(2000年〜2022年9月分は国立国会図書館デジタルコレクション
# https://dl.ndl.go.jp/pid/12896309 を別途参照する必要がある)。
EARLIEST_KNOWN_DATE = date(2022, 10, 1)


def chart_exists(target_date: date, hour: int) -> bool:
    """ダウンロードはせず、その日付・時刻の天気図が実際に存在するかだけ確認する。"""
    if target_date < EARLIEST_KNOWN_DATE:
        from scripts.collect_ndl import get_ndl_day_urls, resolve_ndl_pid

        if hour != 0:
            return False
        try:
            pid = resolve_ndl_pid(target_date.year, target_date.month)
            key = f"JS_{target_date.strftime('%Y%m%d')}{hour:02d}"
            return key in get_ndl_day_urls(pid)
        except (requests.RequestException, FileNotFoundError):
            return False

    url = build_url(target_date, hour)
    try:
        resp = requests.head(url, timeout=15, allow_redirects=True)
        if resp.status_code == 200:
            return True
        if resp.status_code == 405:  # HEAD未対応のサーバー向けフォールバック
            resp = requests.get(url, timeout=15, stream=True)
            resp.close()
            return resp.status_code == 200
        return False
    except requests.RequestException:
        return False


def find_latest_available_date(hour: int = 0, search_back_days: int = 150) -> date | None:
    """今日から遡って、実際に存在する最新の日付を探す(アーカイブは公開まで
    約3ヶ月のタイムラグがあるため、'今日'を指定しても404になることが多い)。

    見つからなければNoneを返す。
    """
    d = date.today()
    for _ in range(search_back_days):
        if chart_exists(d, hour):
            return d
        d -= timedelta(days=1)
    return None


def fetch_chart(date_str: str, hour: int = 0, cache_dir: str = str(DEFAULT_CACHE_DIR)) -> Path:
    """指定した日付・時刻の天気図PDFをダウンロードしてPNGに変換し、そのパスを返す。

    2022-10-01以降は気象庁JSMAPアーカイブ、それより古い日付は国立国会図書館
    デジタルコレクション(NDL)から自動で取得する(NDL側は00Zのみ存在)。
    既にダウンロード済みならキャッシュを再利用する。
    """
    target_date = date.fromisoformat(date_str)

    if target_date < EARLIEST_KNOWN_DATE:
        from scripts.collect_ndl import fetch_ndl_chart

        if hour != 0:
            raise ValueError("NDL側のデータは00Zのみ存在します(--hour 0 を指定してください)")
        return fetch_ndl_chart(
            target_date.year, target_date.month, target_date.day, hour=0, cache_dir=cache_dir
        )

    cache_dir = Path(cache_dir)
    pdf_dir = cache_dir / "pdf"
    png_dir = cache_dir / "png"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    ts = target_date.strftime("%Y%m%d") + f"{hour:02d}"
    png_path = png_dir / f"Js_{ts}.png"
    if png_path.exists():
        return png_path

    pdf_path = pdf_dir / f"Js_{ts}.pdf"
    url = build_url(target_date, hour)
    if not download_pdf(url, pdf_path):
        raise FileNotFoundError(
            f"天気図が見つかりません(404): {url}\n"
            "JSMAPは00Z・12Z(日本時間9時・21時)のみ存在します。"
            "またまだ公開されていない新しい日付は404になります。"
        )

    pdf_to_png(pdf_path, png_path)
    pdf_path.unlink(missing_ok=True)
    return png_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("date", help="YYYY-MM-DD")
    parser.add_argument("--hour", type=int, default=0, choices=[0, 12], help="UTC時刻(0または12)")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    args = parser.parse_args()

    png_path = fetch_chart(args.date, hour=args.hour, cache_dir=args.cache_dir)
    print(png_path)


if __name__ == "__main__":
    main()
