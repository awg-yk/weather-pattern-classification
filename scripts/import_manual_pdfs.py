"""
手動でダウンロードした天気図PDF(NDLの「一括ダウンロード」など)をまとめてPNGに変換する。

想定する使い方:
    1. NDLのデジタルコレクション(https://dl.ndl.go.jp/pid/12896309)で、
       月ごとの巻号ページを開き、「一括ダウンロード」でその月のPDF一式(ZIP)を取得する
    2. ダウンロードしたZIPを展開する(サブフォルダに分かれていてもよい。
       このスクリプトはフォルダ内を再帰的に探索する)
    3. このスクリプトを実行してPNGに変換する

ファイル名は "JS_YYYYMMDDHH.pdf" 形式(気象庁・NDLどちらも共通)を前提とする。
変換後はscripts/preprocess_jma.pyで余白クロップ・日時スタンプ消去を行うこと。

使い方:
    python scripts/import_manual_pdfs.py --in-dir ~/Downloads/tenkizu_2000 --out-dir data/raw/ndl_manual/png
"""

import argparse
from pathlib import Path

from scripts.collect_jma import pdf_to_png


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", required=True, help="ダウンロードしたPDFが入っているフォルダ(再帰的に探索)")
    parser.add_argument("--out-dir", required=True, help="変換後のPNGを保存するフォルダ")
    parser.add_argument("--dpi", type=int, default=200, help="PNG変換時の解像度")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(in_dir.rglob("*.pdf"))
    if not pdf_files:
        print(f"PDFファイルが見つかりません: {in_dir}")
        return

    print(f"{len(pdf_files)}件のPDFが見つかりました。変換します。")

    converted, skipped, failed = 0, 0, 0
    for pdf_path in pdf_files:
        png_path = out_dir / (pdf_path.stem + ".png")
        if png_path.exists():
            skipped += 1
            continue
        try:
            pdf_to_png(pdf_path, png_path, dpi=args.dpi)
            converted += 1
            print(f"OK: {pdf_path.name} -> {png_path.name}")
        except Exception as e:
            failed += 1
            print(f"失敗: {pdf_path.name} ({e})")

    print()
    print(f"変換: {converted}件 / スキップ(既に変換済み): {skipped}件 / 失敗: {failed}件")
    print(f"保存先: {out_dir}")
    print()
    print("次は前処理(余白クロップ・日時スタンプ消去)を行ってください:")
    print(f"  python scripts/preprocess_jma.py --in-dir {out_dir} --out-dir data/processed/ndl_manual")


if __name__ == "__main__":
    main()
