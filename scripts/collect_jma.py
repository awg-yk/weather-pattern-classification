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


def pdf_to_png(pdf_path: Path, png_path: Path, dpi: int = 200, poppler_path=None) -> None:
    """PDFの1ページ目をPNGにする。

    poppler_pathは、popplerのbinフォルダをPATHに通していない場合に直接指定するためのもの
    (Windowsでは環境変数をいじるより手軽なことが多い)。
    """
    pages = convert_from_path(str(pdf_path), dpi=dpi, poppler_path=poppler_path)
    pages[0].save(png_path, "PNG")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--hours", type=int, nargs="+", default=HOURS_UTC, help="取得する観測時刻(UTC)")
    parser.add_argument("--out", default="data/raw/jma")
    parser.add_argument("--keep-pdf", action="store_true", help="変換後もPDFを残す")
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="PDFのダウンロードだけ行い、PNGへの変換はしない。"
        "poppler(pdftoppm)の準備前に取得だけ先に進めたいときに使う。"
        "同じ--outで--download-onlyなしで再実行すれば、変換だけ後から実行できる",
    )
    parser.add_argument(
        "--poppler-path",
        default=None,
        help=r"popplerのbinフォルダ(例: C:\poppler\Library\bin)。"
        "PATHに通していない場合に指定する",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    pdf_dir = out_dir / "pdf"
    png_dir = out_dir / "png"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    downloaded, convert_failures = 0, 0

    for d in daterange(start, end):
        for hour in args.hours:
            ts = d.strftime("%Y%m%d") + f"{hour:02d}"
            url = build_url(d, hour)
            pdf_path = pdf_dir / f"Js_{ts}.pdf"
            png_path = png_dir / f"Js_{ts}.png"

            if png_path.exists():
                continue

            # 前回ダウンロード済みのPDFが残っていれば再取得しない。
            # (変換だけ失敗していた場合、通信をやり直さず変換から再開できる)
            reused = pdf_path.exists() and pdf_path.stat().st_size > 0
            if reused:
                print(f"{ts} -> ダウンロード済みのPDFを再利用")
            else:
                ok = download_pdf(url, pdf_path)
                if not ok:
                    print(f"{ts} -> FAILED (404 or empty) ({url})")
                    continue
                downloaded += 1

            if not args.download_only:
                try:
                    pdf_to_png(pdf_path, png_path, poppler_path=args.poppler_path)
                    print(f"{ts} -> OK ({png_path})")
                    # 変換できた場合だけPDFを消す。失敗したのに消すと、
                    # ダウンロードし直しになってしまう。
                    if not args.keep_pdf:
                        pdf_path.unlink(missing_ok=True)
                except Exception as e:
                    convert_failures += 1
                    print(f"{ts} -> PDF->PNG変換失敗(PDFは保持): {e}")
            else:
                print(f"{ts} -> DL ({pdf_path})")

            # 待つのはサーバーへの負荷を抑えるため。既存PDFを使った回は
            # 通信していないので待たない(変換のやり直しが無駄に遅くなる)。
            if not reused:
                time.sleep(REQUEST_INTERVAL_SEC)

    print(f"\n新規ダウンロード: {downloaded}件")
    if args.download_only:
        print(
            f"PDFは {pdf_dir} にあります。PNGへの変換は、同じ--outを指定して\n"
            "--download-only なしで再実行してください(再ダウンロードはしません)。"
        )
    if convert_failures:
        print(
            f"\n{convert_failures}件がPNGに変換できませんでした。PDFは {pdf_dir} に残してあります。\n"
            "poppler(pdftoppm)が入っていない可能性があります。導入後に\n"
            f"  python -m scripts.collect_jma --start {args.start} --end {args.end} --out {args.out}\n"
            "を再実行すると、ダウンロード済みのPDFを使って変換だけやり直します。"
        )


if __name__ == "__main__":
    main()
