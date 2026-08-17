"""
気象庁「保存用天気図」(日本域天気図 JSMAP) をPDFでダウンロードし、PNG画像に変換するスクリプト。

URLパターン(確認済み・2026年4月時点):
    https://www.data.jma.go.jp/yoho/data/wxchart/archive/{yyyy}_{mm}/PDFDATA/JSMAP/Js_{yyyymmddHH}.pdf

観測時刻は動作確認の結果、00Z・12Z (日本時間9時・21時) のみ存在することを確認済み
(06Z・18Z は404)。他の時刻を試したい場合は --hours で指定できるが、存在しない
組み合わせは404になるため失敗はスキップして続行する。

このシリーズは配色が統一されている(海岸線・経緯度線=赤茶色、等圧線=黒、
温暖前線=赤、寒冷前線=青、閉塞前線=ピンク)ため、カラーのまま学習データとして使える。

利用規約(気象庁ウェブサイト利用規約)を必ず確認し、許可された範囲・頻度で利用すること。
大量の連続アクセスはサーバー負荷になるため、リクエスト間に十分なインターバルを空ける。

事前準備:
    pip install pdf2image
    poppler-utils (pdftoppmコマンド) がシステムに必要
    - Debian/Ubuntu: apt-get install poppler-utils
    - Mac: brew install poppler

使い方:
    python scripts/collect_jma.py --start 2026-04-01 --end 2026-04-30 --out data/raw/jma
"""

import argparse
import time
from datetime import date, timedelta
from pathlib import Path

import requests
from pdf2image import convert_from_path

BASE_URL = "https://www.data.jma.go.jp/yoho/data/wxchart/archive/{yyyy}_{mm}/PDFDATA/JSMAP/Js_{yyyymmddhh}.pdf"
REQUEST_INTERVAL_SEC = 2.0
HOURS_UTC = [0, 12]  # 動作確認済み(日本時間9時・21時の地上天気図)


def daterange(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def build_url(target_date: date, hour: int) -> str:
    return BASE_URL.format(
        yyyy=target_date.strftime("%Y"),
        mm=target_date.strftime("%m"),
        yyyymmddhh=target_date.strftime("%Y%m%d") + f"{hour:02d}",
    )


def download_pdf(url: str, out_path: Path) -> bool:
    resp = requests.get(url, timeout=30)
    if resp.status_code != 200 or not resp.content:
        return False
    out_path.write_bytes(resp.content)
    return True


def pdf_to_png(pdf_path: Path, png_path: Path, dpi: int = 200) -> None:
    pages = convert_from_path(str(pdf_path), dpi=dpi)
    pages[0].save(png_path, "PNG")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--hours", type=int, nargs="+", default=HOURS_UTC, help="取得する観測時刻(UTC)")
    parser.add_argument("--out", default="data/raw/jma")
    parser.add_argument("--keep-pdf", action="store_true", help="変換後もPDFを残す")
    args = parser.parse_args()

    out_dir = Path(args.out)
    pdf_dir = out_dir / "pdf"
    png_dir = out_dir / "png"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    for d in daterange(start, end):
        for hour in args.hours:
            ts = d.strftime("%Y%m%d") + f"{hour:02d}"
            url = build_url(d, hour)
            pdf_path = pdf_dir / f"Js_{ts}.pdf"
            png_path = png_dir / f"Js_{ts}.png"

            if png_path.exists():
                continue

            ok = download_pdf(url, pdf_path)
            if not ok:
                print(f"{ts} -> FAILED (404 or empty) ({url})")
                continue

            try:
                pdf_to_png(pdf_path, png_path)
                print(f"{ts} -> OK ({png_path})")
            except Exception as e:
                print(f"{ts} -> PDF->PNG変換失敗: {e}")
            finally:
                if not args.keep_pdf:
                    pdf_path.unlink(missing_ok=True)

            time.sleep(REQUEST_INTERVAL_SEC)


if __name__ == "__main__":
    main()
