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
from datetime import date
from pathlib import Path

from scripts.collect_jma import build_url, download_pdf, pdf_to_png

DEFAULT_CACHE_DIR = Path("data/raw/jma_fetch")


def fetch_chart(date_str: str, hour: int = 0, cache_dir: str = str(DEFAULT_CACHE_DIR)) -> Path:
    """指定した日付・時刻の天気図PDFをダウンロードしてPNGに変換し、そのパスを返す。

    既にダウンロード済みならキャッシュを再利用する。存在しない日付・時刻の場合は
    FileNotFoundErrorを送出する(JSMAPは00Z・12Zのみ存在。他の時刻は404になりやすい)。
    """
    target_date = date.fromisoformat(date_str)
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
            "また直近1年より古い/新しい日付は非公開の可能性があります。"
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
