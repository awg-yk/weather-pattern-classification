# ============================================================
# Colabの「何もないセル」にこのファイルの中身を丸ごと貼り付けて実行してください。
# ランタイムを再起動した後も、このセルを最初から実行し直せば復帰できます。
# ============================================================

import subprocess
from pathlib import Path

REPO_URL = "https://github.com/awg-yk/weather-pattern-classification.git"
BRANCH = "claude/weather-chart-classification-4b6in1"
REPO_DIR = Path("/content/weather-pattern-classification")
DRIVE_DATA_DIR = Path("/content/drive/MyDrive/weather-pattern-classification-data")

# 1. Google Driveをマウント
from google.colab import drive
drive.mount("/content/drive")

# 2. リポジトリを取得(なければclone、あればpullで最新化)
if REPO_DIR.exists():
    subprocess.run(["git", "-C", str(REPO_DIR), "fetch", "origin", BRANCH], check=True)
    subprocess.run(["git", "-C", str(REPO_DIR), "checkout", BRANCH], check=True)
    subprocess.run(["git", "-C", str(REPO_DIR), "pull", "origin", BRANCH], check=True)
else:
    subprocess.run(["git", "clone", "-b", BRANCH, REPO_URL, str(REPO_DIR)], check=True)

# 3. 依存パッケージをインストール
subprocess.run(["pip", "install", "-q", "-r", str(REPO_DIR / "requirements.txt")], check=True)

# 4. Drive上の作業ディレクトリを用意
processed_dir = DRIVE_DATA_DIR / "processed"
labels_csv = DRIVE_DATA_DIR / "labels.csv"
processed_dir.mkdir(parents=True, exist_ok=True)

n_images = len(list(processed_dir.glob("*.png")))
print(f"processed images: {n_images} 枚 ({processed_dir})")
if n_images == 0:
    print("!! 前処理済み画像が見つかりません。先にダウンロード・前処理を実行してください。")

# 5. ラベリングツールを起動
import sys
sys.path.append(str(REPO_DIR))

import importlib
import src.labels
import scripts.label_tool
importlib.reload(src.labels)
importlib.reload(scripts.label_tool)
from scripts.label_tool import run_labeling_session

run_labeling_session(
    images_dir=str(processed_dir),
    labels_csv=str(labels_csv),
)

print(
    "\n最初の1クリック後、別セルで次を実行して保存されているか必ず確認してください:\n"
    f'  !tail -3 "{labels_csv}"'
)
