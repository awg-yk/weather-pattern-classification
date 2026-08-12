"""
気象庁の地上天気図画像を日付範囲を指定して収集するスクリプト。

気象庁の実況天気図・過去天気図の画像URLは仕様が変わることがあるため、
実行前に https://www.jma.go.jp/bosai/weather_map/ 等で現在のURL形式を確認し、
BASE_URL / build_url() を必要に応じて修正すること。

利用規約(気象庁ウェブサイト利用規約)を必ず確認し、許可された範囲・頻度で利用すること。
大量の連続アクセスはサーバー負荷になるため、リクエスト間に十分なインターバルを空ける。

使い方:
    python scripts/collect_jma.py --start 2024-01-01 --end 2024-01-31 --out data/raw/jma
"""

import argparse
import time
from datetime import date, timedelta
from pathlib import Path

import requests

BASE_URL = "https://www.jma.go.jp/bosai/weather_map/data/png/{yyyymmddhh}.png"
REQUEST_INTERVAL_SEC = 2.0


def daterange(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def build_url(target_date: date, hour: int = 9) -> str:
    yyyymmddhh = target_date.strftime("%Y%m%d") + f"{hour:02d}"
    return BASE_URL.format(yyyymmddhh=yyyymmddhh)


def download(url: str, out_path: Path) -> bool:
    resp = requests.get(url, timeout=30)
    if resp.status_code != 200 or not resp.content:
        return False
    out_path.write_bytes(resp.content)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--hour", type=int, default=9, help="天気図の観測時刻(JST時)")
    parser.add_argument("--out", default="data/raw/jma")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    for d in daterange(start, end):
        url = build_url(d, args.hour)
        out_path = out_dir / f"{d.isoformat()}_{args.hour:02d}.png"
        if out_path.exists():
            continue
        ok = download(url, out_path)
        print(f"{d.isoformat()} -> {'OK' if ok else 'FAILED'} ({url})")
        time.sleep(REQUEST_INTERVAL_SEC)


if __name__ == "__main__":
    main()
